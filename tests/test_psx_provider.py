"""Tests for src.data.psx_provider (Phase 7).

No real network access is used or required: connect()/get_*_observation()
are exercised against a mocked urllib.request.urlopen, so these tests run
fully offline and never touch any real endpoint, licensed or otherwise.
"""

from __future__ import annotations

import json
from unittest.mock import MagicMock, patch

import pytest

from src.data.base_provider import ProviderNotConnectedError
from src.data.psx_provider import (
    PSXConnectionError,
    PSXLicenseRequiredError,
    PSXProvider,
)


def _fake_response(payload: dict) -> MagicMock:
    """Build a mock object usable as the `with urlopen(...) as response:` target."""
    response = MagicMock()
    response.read.return_value = json.dumps(payload).encode("utf-8")
    response.__enter__.return_value = response
    response.__exit__.return_value = False
    return response


def test_connect_without_credentials_raises_license_required():
    provider = PSXProvider()  # no base_url/api_key
    with pytest.raises(PSXLicenseRequiredError) as exc_info:
        provider.connect()
    assert "marketdatarequest@psx.com.pk" in str(exc_info.value)
    assert "PSX_DATA_API_BASE_URL" in str(exc_info.value)


def test_is_configured_reflects_credential_presence():
    assert PSXProvider().is_configured() is False
    assert PSXProvider(base_url="https://x", api_key="k").is_configured() is True
    assert PSXProvider(base_url="https://x").is_configured() is False  # missing key
    assert PSXProvider(api_key="k").is_configured() is False  # missing base_url


def test_from_env_reads_environment_variables(monkeypatch):
    monkeypatch.setenv("PSX_DATA_API_BASE_URL", "https://vendor.example.com/api")
    monkeypatch.setenv("PSX_DATA_API_KEY", "my-key")
    monkeypatch.setenv("PSX_DATA_API_SECRET", "my-secret")

    provider = PSXProvider.from_env()
    assert provider.is_configured() is True
    assert provider._base_url == "https://vendor.example.com/api"
    assert provider._api_key == "my-key"
    assert provider._api_secret == "my-secret"


def test_from_env_missing_variables_yields_unconfigured_provider(monkeypatch):
    monkeypatch.delenv("PSX_DATA_API_BASE_URL", raising=False)
    monkeypatch.delenv("PSX_DATA_API_KEY", raising=False)
    provider = PSXProvider.from_env()
    assert provider.is_configured() is False


def test_provider_identity_is_not_synthetic():
    provider = PSXProvider(base_url="https://x", api_key="k")
    assert provider.provider_name == "psx"
    assert provider.is_synthetic is False


def test_methods_raise_before_connect():
    provider = PSXProvider(base_url="https://x", api_key="k")
    with pytest.raises(ProviderNotConnectedError):
        provider.get_index_observation("KSE100")
    with pytest.raises(ProviderNotConnectedError):
        provider.get_stock_observation("THCCL")


@patch("urllib.request.urlopen")
def test_connect_succeeds_when_health_check_responds(mock_urlopen):
    mock_urlopen.return_value = _fake_response({"status": "ok"})
    provider = PSXProvider(base_url="https://vendor.example.com/api", api_key="k")

    provider.connect()

    assert provider.is_connected() is True
    called_request = mock_urlopen.call_args[0][0]
    assert called_request.full_url == "https://vendor.example.com/api/health"
    assert called_request.get_header("Authorization") == "Bearer k"


@patch("urllib.request.urlopen")
def test_connect_raises_connection_error_when_endpoint_unreachable(mock_urlopen):
    import urllib.error

    mock_urlopen.side_effect = urllib.error.URLError("connection refused")
    provider = PSXProvider(base_url="https://vendor.example.com/api", api_key="k")

    with pytest.raises(PSXConnectionError):
        provider.connect()
    assert provider.is_connected() is False


@patch("urllib.request.urlopen")
def test_get_index_observation_parses_response(mock_urlopen):
    mock_urlopen.side_effect = [
        _fake_response({"status": "ok"}),  # health check during connect()
        _fake_response(
            {
                "value": 45210.5,
                "change": 12.3,
                "change_percent": 0.03,
                "high": 45300.0,
                "low": 45100.0,
                "volume": 500000,
                "market_status": "OPEN",
                "timestamp": "2026-09-14T09:30:00+05:00",
            }
        ),
    ]
    provider = PSXProvider(base_url="https://vendor.example.com/api", api_key="k")
    provider.connect()

    obs = provider.get_index_observation("KSE100")

    assert obs.provider == "psx"
    assert obs.symbol == "KSE100"
    assert obs.value == 45210.5
    assert obs.is_synthetic is False
    assert obs.market_status == "OPEN"
    assert obs.timestamp.utcoffset().total_seconds() == 5 * 3600


@patch("urllib.request.urlopen")
def test_get_stock_observation_parses_response(mock_urlopen):
    mock_urlopen.side_effect = [
        _fake_response({"status": "ok"}),
        _fake_response(
            {
                "last_price": 80.25,
                "open": 80.0,
                "high": 80.5,
                "low": 79.8,
                "previous_close": 79.9,
                "volume": 12345,
                "market_status": "OPEN",
                "timestamp": "2026-09-14T09:31:00+05:00",
            }
        ),
    ]
    provider = PSXProvider(base_url="https://vendor.example.com/api", api_key="k")
    provider.connect()

    obs = provider.get_stock_observation("THCCL")

    assert obs.provider == "psx"
    assert obs.symbol == "THCCL"
    assert obs.last_price == 80.25
    assert obs.is_synthetic is False


@patch("urllib.request.urlopen")
def test_missing_fields_in_response_become_none_not_fabricated(mock_urlopen):
    mock_urlopen.side_effect = [
        _fake_response({"status": "ok"}),
        _fake_response({"last_price": 80.25}),  # vendor didn't supply anything else
    ]
    provider = PSXProvider(base_url="https://vendor.example.com/api", api_key="k")
    provider.connect()

    obs = provider.get_stock_observation("THCCL")

    assert obs.last_price == 80.25
    assert obs.open is None
    assert obs.bid is None
    assert obs.vwap is None


def test_secret_header_only_sent_when_configured():
    provider_without_secret = PSXProvider(base_url="https://x", api_key="k")
    provider_with_secret = PSXProvider(base_url="https://x", api_key="k", api_secret="s")

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = _fake_response({"status": "ok"})
        provider_without_secret.connect()
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("X-api-secret") is None

    with patch("urllib.request.urlopen") as mock_urlopen:
        mock_urlopen.return_value = _fake_response({"status": "ok"})
        provider_with_secret.connect()
        req = mock_urlopen.call_args[0][0]
        assert req.get_header("X-api-secret") == "s"
