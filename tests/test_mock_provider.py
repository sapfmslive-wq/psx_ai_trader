"""Tests for src.data.base_provider and src.data.mock_provider (Phase 4)."""

from __future__ import annotations

import pytest

from src.data.base_provider import ProviderNotConnectedError
from src.data.mock_provider import MockProvider


def test_methods_raise_before_connect():
    provider = MockProvider(seed=1)
    with pytest.raises(ProviderNotConnectedError):
        provider.get_index_observation("KSE100")
    with pytest.raises(ProviderNotConnectedError):
        provider.get_stock_observation("THCCL")


def test_connect_and_disconnect_toggle_is_connected():
    provider = MockProvider(seed=1)
    assert provider.is_connected() is False
    provider.connect()
    assert provider.is_connected() is True
    provider.disconnect()
    assert provider.is_connected() is False


def test_context_manager_connects_and_disconnects():
    provider = MockProvider(seed=1)
    with provider as p:
        assert p.is_connected() is True
        obs = p.get_index_observation("KSE100")
        assert obs.is_synthetic is True
    assert provider.is_connected() is False


def test_provider_identity_and_synthetic_labeling():
    provider = MockProvider(seed=1)
    assert provider.provider_name == "mock"
    assert provider.is_synthetic is True

    provider.connect()
    index_obs = provider.get_index_observation("KSE100")
    stock_obs = provider.get_stock_observation("THCCL")

    assert index_obs.provider == "mock"
    assert index_obs.is_synthetic is True
    assert index_obs.market_status == "SYNTHETIC_OPEN"

    assert stock_obs.provider == "mock"
    assert stock_obs.is_synthetic is True
    assert stock_obs.market_status == "SYNTHETIC_OPEN"


def test_index_observation_has_plausible_values():
    provider = MockProvider(seed=42)
    provider.connect()
    obs = provider.get_index_observation("KSE100")

    assert obs.value is not None and obs.value > 0
    assert obs.high >= obs.value
    assert obs.low <= obs.value
    assert obs.volume is not None and obs.volume >= 0
    assert obs.symbol == "KSE100"


def test_stock_high_low_track_the_running_extremes():
    provider = MockProvider(seed=7)
    provider.connect()

    seen_prices = []
    for _ in range(20):
        obs = provider.get_stock_observation("THCCL")
        seen_prices.append(obs.last_price)
        assert obs.high >= max(seen_prices)
        assert obs.low <= min(seen_prices)


def test_stock_cumulative_volume_never_decreases():
    provider = MockProvider(seed=7)
    provider.connect()

    previous_volume = -1.0
    for _ in range(15):
        obs = provider.get_stock_observation("THCCL")
        assert obs.volume >= previous_volume
        previous_volume = obs.volume


def test_same_seed_produces_identical_sequences():
    provider_a = MockProvider(seed=123)
    provider_b = MockProvider(seed=123)
    provider_a.connect()
    provider_b.connect()

    for _ in range(10):
        obs_a = provider_a.get_index_observation("KSE100")
        obs_b = provider_b.get_index_observation("KSE100")
        assert obs_a.value == obs_b.value
        assert obs_a.change == obs_b.change


def test_different_seeds_produce_different_sequences():
    provider_a = MockProvider(seed=1)
    provider_b = MockProvider(seed=2)
    provider_a.connect()
    provider_b.connect()

    values_a = [provider_a.get_index_observation("KSE100").value for _ in range(5)]
    values_b = [provider_b.get_index_observation("KSE100").value for _ in range(5)]
    assert values_a != values_b


def test_unknown_symbol_falls_back_to_default_base_price_without_crashing():
    provider = MockProvider(seed=1)
    provider.connect()

    index_obs = provider.get_index_observation("SOME_NEW_INDEX")
    stock_obs = provider.get_stock_observation("SOME_NEW_STOCK")

    assert index_obs.value is not None
    assert stock_obs.last_price is not None


def test_custom_base_prices_are_respected_on_first_call():
    provider = MockProvider(
        seed=1,
        volatility=0.0,  # no randomness: first call should equal the base price exactly
        index_base_values={"KSE100": 12345.0},
        stock_base_prices={"THCCL": 55.5},
    )
    provider.connect()

    index_obs = provider.get_index_observation("KSE100")
    stock_obs = provider.get_stock_observation("THCCL")

    assert index_obs.value == 12345.0
    assert stock_obs.last_price == 55.5
    assert stock_obs.open == 55.5
    assert stock_obs.previous_close == 55.5
