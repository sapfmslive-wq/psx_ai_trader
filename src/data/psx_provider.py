"""PSXProvider — the real-PSX-data adapter (Phase 7, Sections 3 and 116).

RESEARCH SUMMARY (done before writing any code here, per Section 3/116):

Pakistan Stock Exchange (PSX, psx.com.pk — not to be confused with Nasdaq's
unrelated "PSX", the Philadelphia Stock Exchange) does not publish a free,
self-service, developer-facing API for real-time or licensed historical
market data. What actually exists, as of this research:

1. A free public "Data Portal" at dps.psx.com.pk. Individual investors can
   browse current-session market data, end-of-day data, announcements, and
   a stock screener with no registration or payment. HOWEVER, the portal
   itself carries an explicit legal notice (checked directly on
   dps.psx.com.pk): "any dissemination, transmission, sale, and commercial
   use of Market Data feed (on a live or delayed basis) ... through any
   mode of communication, including ... electronic communications ...
   without acquiring respective rights/license from the PSX is strictly
   prohibited," with PSX reserving the right to pursue civil/criminal
   proceedings. This is a page meant for a human to look at, not an
   endpoint this project is authorized to poll and store data from.

2. Real-time and licensed historical feeds (Level 1, Level 2/tick-by-tick,
   end-of-day) are sold as commercial products through PSX's own "Data
   Services Vending" program (psx.com.pk/psx/product-and-services/
   data-services-vending) and through PSX-authorized third-party data
   vendors. The documented contact for licensing is
   marketdatarequest@psx.com.pk.

3. Multiple community/open-source projects scrape dps.psx.com.pk directly
   without a license. This project does not do that: Section 3 explicitly
   prohibits scraping a website aggressively or bypassing licensing/access
   controls, "regardless of how the request is framed" or how common the
   practice is elsewhere.

CONCLUSION AND WHAT THIS MODULE ACTUALLY DOES:

`PSXProvider` is a fully-implemented `DataProvider`, but it is a **licensed-
feed adapter**, not a scraper. It refuses to connect at all unless the
person running this project has their own licensed data source configured
(either a direct PSX Data Services Vending arrangement, or a subscription
with a PSX-authorized third-party vendor that exposes an HTTP API). Without
that configuration, `connect()` raises `PSXLicenseRequiredError` explaining
exactly what's needed and where to get it — it never silently falls back to
scraping the public portal.

Because no specific vendor's exact API shape is known to this project (each
commercial vendor documents its own, different schema), this class defines
a minimal, generic HTTP contract the *configured* endpoint is expected to
implement:

    GET {base_url}/health              -> 200 OK  (used by connect() to verify reachability)
    GET {base_url}/index/{symbol}      -> JSON matching IndexObservation's fields
    GET {base_url}/stock/{symbol}      -> JSON matching StockObservation's fields

In practice this means: either your licensed vendor already speaks this
shape, or you put a small translation shim in front of their real API that
does. That shim is outside this project's scope — it depends entirely on
which vendor you license data from.

Required environment variables (see `.env.example`):
    PSX_DATA_API_BASE_URL   e.g. https://your-licensed-vendor.example.com/api
    PSX_DATA_API_KEY        sent as an Authorization: Bearer header
    PSX_DATA_API_SECRET     optional, sent as X-Api-Secret if your vendor needs it
"""

from __future__ import annotations

import datetime as dt
import json
import os
import urllib.error
import urllib.parse
import urllib.request

from src.data.base_provider import DataProvider, IndexObservation, StockObservation

LICENSE_REQUIRED_MESSAGE = (
    "PSXProvider requires a licensed data source — it does not, and will not, "
    "scrape PSX's public Data Portal (dps.psx.com.pk), whose terms explicitly "
    "prohibit automated dissemination of its market data without a license.\n\n"
    "To use real PSX data, either:\n"
    "  1) Obtain a data license directly from PSX's Data Services Vending "
    "team (marketdatarequest@psx.com.pk), or\n"
    "  2) Subscribe to a PSX-authorized third-party data vendor that exposes "
    "an HTTP API,\n"
    "then set these in your .env file:\n"
    "  PSX_DATA_API_BASE_URL=<your vendor's API base URL>\n"
    "  PSX_DATA_API_KEY=<your API key>\n"
    "  PSX_DATA_API_SECRET=<optional, if your vendor requires it>\n\n"
    "Until then, use the mock provider (data.provider: mock) or the CSV "
    "importer for research and testing."
)


class PSXLicenseRequiredError(RuntimeError):
    """Raised when PSXProvider is used without a configured licensed endpoint."""


class PSXConnectionError(RuntimeError):
    """Raised when the configured licensed endpoint is unreachable or returns
    something this adapter can't parse."""


class PSXProvider(DataProvider):
    provider_name = "psx"
    is_synthetic = False

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        api_secret: str | None = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        super().__init__()
        self._base_url = base_url.rstrip("/") if base_url else None
        self._api_key = api_key
        self._api_secret = api_secret
        self._timeout_seconds = timeout_seconds

    @classmethod
    def from_env(cls) -> "PSXProvider":
        """Build a provider from PSX_DATA_API_* environment variables.

        Does not raise if they're missing — that's `connect()`'s job, so the
        error message is seen at the moment a connection is actually
        attempted, not at construction time.
        """
        return cls(
            base_url=os.getenv("PSX_DATA_API_BASE_URL") or None,
            api_key=os.getenv("PSX_DATA_API_KEY") or None,
            api_secret=os.getenv("PSX_DATA_API_SECRET") or None,
        )

    def is_configured(self) -> bool:
        return bool(self._base_url and self._api_key)

    def connect(self) -> None:
        if not self.is_configured():
            raise PSXLicenseRequiredError(LICENSE_REQUIRED_MESSAGE)
        # Verify the configured endpoint is actually reachable before
        # declaring ourselves connected — never assume it silently.
        self._request("GET", "/health")
        self._connected = True

    def disconnect(self) -> None:
        self._connected = False

    def _request(self, method: str, path: str) -> dict:
        assert self._base_url is not None  # guaranteed by is_configured() check in connect()
        url = f"{self._base_url}{path}"
        request = urllib.request.Request(url, method=method)
        request.add_header("Authorization", f"Bearer {self._api_key}")
        if self._api_secret:
            request.add_header("X-Api-Secret", self._api_secret)

        try:
            with urllib.request.urlopen(request, timeout=self._timeout_seconds) as response:
                body = response.read().decode("utf-8")
                return json.loads(body) if body else {}
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise PSXConnectionError(
                f"Request to configured PSX data endpoint failed ({url}): {exc}"
            ) from exc
        except json.JSONDecodeError as exc:
            raise PSXConnectionError(
                f"Configured PSX data endpoint returned invalid JSON ({url}): {exc}"
            ) from exc

    @staticmethod
    def _parse_timestamp(raw: object) -> dt.datetime | None:
        if raw is None:
            return None
        try:
            parsed = dt.datetime.fromisoformat(str(raw))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=dt.timezone.utc)
        return parsed

    def get_index_observation(self, symbol: str) -> IndexObservation:
        self._require_connected()
        data = self._request("GET", f"/index/{urllib.parse.quote(symbol)}")
        now = dt.datetime.now(dt.timezone.utc)
        return IndexObservation(
            provider=self.provider_name,
            symbol=symbol,
            timestamp=self._parse_timestamp(data.get("timestamp")) or now,
            value=data.get("value"),
            change=data.get("change"),
            change_percent=data.get("change_percent"),
            high=data.get("high"),
            low=data.get("low"),
            volume=data.get("volume"),
            market_status=data.get("market_status"),
            source_timestamp=self._parse_timestamp(data.get("source_timestamp")),
            is_synthetic=False,
        )

    def get_stock_observation(self, symbol: str) -> StockObservation:
        self._require_connected()
        data = self._request("GET", f"/stock/{urllib.parse.quote(symbol)}")
        now = dt.datetime.now(dt.timezone.utc)
        return StockObservation(
            provider=self.provider_name,
            symbol=symbol,
            timestamp=self._parse_timestamp(data.get("timestamp")) or now,
            last_price=data.get("last_price"),
            open=data.get("open"),
            high=data.get("high"),
            low=data.get("low"),
            previous_close=data.get("previous_close"),
            change=data.get("change"),
            change_percent=data.get("change_percent"),
            volume=data.get("volume"),
            bid=data.get("bid"),
            bid_volume=data.get("bid_volume"),
            ask=data.get("ask"),
            ask_volume=data.get("ask_volume"),
            turnover=data.get("turnover"),
            vwap=data.get("vwap"),
            market_status=data.get("market_status"),
            source_timestamp=self._parse_timestamp(data.get("source_timestamp")),
            is_synthetic=False,
        )
