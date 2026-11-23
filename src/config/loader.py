"""Configuration loading and validation.

Loads `config/settings.yaml`, `config/symbols.yaml`, and `.env`, validates
every value, and returns an immutable `AppConfig`. Nothing downstream should
ever call `yaml.safe_load` or read `.env` directly — this module is the one
and only place configuration is parsed, so validation rules live in exactly
one location.

Design choices, and why:

- Validation collects *all* problems before raising, instead of failing on
  the first one. Fixing a config file one error at a time is miserable;
  this reports everything wrong in one pass.
- The LIVE-trading safety switch is checked here, at config-load time, not
  only in Phase 26's order-placement code. If `trading.mode` is set to LIVE
  in `settings.yaml` but `LIVE_TRADING_ENABLED` is not `true` in `.env`,
  config loading itself fails. A user cannot accidentally get closer to
  live trading by editing only one of the two files.
- `.env` is read without mutating `os.environ` globally (via
  `dotenv.dotenv_values`), so loading config in a test or a second process
  never leaks state.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml
from dotenv import dotenv_values

from src.utils.file_utils import resolve_path

_ALLOWED_PROVIDERS = frozenset({"mock", "csv", "historical", "psx", "broker"})
_ALLOWED_TRADING_MODES = frozenset({"RESEARCH", "BACKTEST", "PAPER", "LIVE"})
_ALLOWED_LOG_LEVELS = frozenset({"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"})
_HORIZON_PATTERN = re.compile(r"^\d+[mhd]$")


class ConfigValidationError(Exception):
    """Raised when one or more configuration values fail validation.

    `errors` holds every problem found in a single load attempt, so the
    user can fix the config file once instead of playing whack-a-mole.
    """

    def __init__(self, errors: list[str]):
        self.errors = errors
        message = "Configuration validation failed:\n" + "\n".join(
            f"  - {e}" for e in errors
        )
        super().__init__(message)


@dataclass(frozen=True)
class MarketConfig:
    timezone: str


@dataclass(frozen=True)
class DataConfig:
    provider: str


@dataclass(frozen=True)
class CollectorConfig:
    reconnect_seconds: int
    stale_timeout_seconds: int
    poll_interval_seconds: int
    max_consecutive_failures: int


@dataclass(frozen=True)
class PredictionConfig:
    horizons: tuple[str, ...]


@dataclass(frozen=True)
class RiskConfig:
    max_risk_per_trade: float
    max_positions: int


@dataclass(frozen=True)
class TradingConfig:
    mode: str
    live_trading_enabled: bool


@dataclass(frozen=True)
class DashboardConfig:
    host: str
    port: int
    allow_external: bool


@dataclass(frozen=True)
class LoggingConfig:
    level: str


@dataclass(frozen=True)
class DatabaseConfig:
    path: str
    echo: bool


@dataclass(frozen=True)
class QualityConfig:
    max_change_percent: float
    stale_seconds: int
    future_timestamp_tolerance_seconds: int


@dataclass(frozen=True)
class CandleConfig:
    volume_convention: str  # "cumulative" or "per_tick"


@dataclass(frozen=True)
class IndicatorsConfig:
    sma_periods: tuple[int, ...]
    ema_periods: tuple[int, ...]
    rsi_period: int
    macd_fast_period: int
    macd_slow_period: int
    macd_signal_period: int
    atr_period: int
    adx_period: int
    bollinger_period: int
    bollinger_num_std: float
    roc_period: int
    momentum_period: int
    stochastic_k_period: int
    stochastic_d_period: int
    volume_ma_period: int


@dataclass(frozen=True)
class FeaturesConfig:
    return_horizons: tuple[int, ...]  # in bars of whatever timeframe features are built at
    high_low_breakout_period: int
    trend_fast_sma_period: int
    trend_slow_sma_period: int
    volatility_period: int


@dataclass(frozen=True)
class LabelsConfig:
    horizons: tuple[int, ...]  # in bars of whatever timeframe labels are built at
    neutral_threshold_pct: float
    take_profit_pct: float
    stop_loss_pct: float


@dataclass(frozen=True)
class BaselinesConfig:
    ma_fast_period: int
    ma_slow_period: int
    rsi_oversold: float
    rsi_overbought: float
    train_fraction: float
    random_seed: int


@dataclass(frozen=True)
class ModelsConfig:
    train_fraction: float
    validation_fraction: float
    random_seed: int
    enabled: tuple[str, ...]
    min_rows: int
    permutation_repeats: int


@dataclass(frozen=True)
class CostsConfig:
    commission_rate_pct: float
    commission_min_per_share: float
    commission_max_pct: float
    day_trade_one_side_only: bool
    sales_tax_on_commission_pct: float
    cdc_per_share: float
    other_levies_pct_per_side: float
    cgt_apply: bool
    tax_filer: bool
    cgt_filer_rate_pct: float
    cgt_non_filer_rate_pct: float

    @property
    def cgt_rate_pct(self) -> float:
        return self.cgt_filer_rate_pct if self.tax_filer else self.cgt_non_filer_rate_pct


@dataclass(frozen=True)
class BacktestConfig:
    initial_cash: float
    position_fraction: float
    execution_delay_bars: int
    slippage_bps: float
    max_participation_pct: float
    stop_loss_pct: float
    take_profit_pct: float
    min_price: float
    min_avg_volume: float
    avg_volume_bars: int
    risk_free_rate_annual_pct: float
    liquidate_at_end: bool
    min_trades_for_confidence: int


@dataclass(frozen=True)
class SymbolsConfig:
    indices: tuple[str, ...]
    stocks: tuple[str, ...]
    watchlist: tuple[str, ...]


@dataclass(frozen=True)
class AppConfig:
    market: MarketConfig
    data: DataConfig
    collector: CollectorConfig
    prediction: PredictionConfig
    risk: RiskConfig
    trading: TradingConfig
    dashboard: DashboardConfig
    logging: LoggingConfig
    database: DatabaseConfig
    quality: QualityConfig
    candles: CandleConfig
    indicators: IndicatorsConfig
    features: FeaturesConfig
    labels: LabelsConfig
    baselines: BaselinesConfig
    models: ModelsConfig
    backtest: BacktestConfig
    costs: CostsConfig
    symbols: SymbolsConfig


def _load_yaml(path: Path, errors: list[str]) -> dict:
    if not path.is_file():
        errors.append(f"Config file not found: {path}")
        return {}
    try:
        with path.open("r", encoding="utf-8") as fh:
            data = yaml.safe_load(fh)
    except yaml.YAMLError as exc:
        errors.append(f"Failed to parse {path}: {exc}")
        return {}
    if not isinstance(data, dict):
        errors.append(f"{path} must contain a YAML mapping at the top level.")
        return {}
    return data


def _require_section(raw: dict, name: str, errors: list[str]) -> dict:
    section = raw.get(name)
    if not isinstance(section, dict):
        errors.append(f"Missing or invalid section '{name}' in settings.yaml.")
        return {}
    return section


def _parse_bool_env(raw_value: str | None, var_name: str, errors: list[str]) -> bool:
    if raw_value is None:
        return False  # fail safe: absent means disabled
    normalized = raw_value.strip().lower()
    if normalized not in ("true", "false"):
        errors.append(
            f"{var_name} must be 'true' or 'false' (got: {raw_value!r})."
        )
        return False
    return normalized == "true"


def _validate_market(section: dict, errors: list[str]) -> MarketConfig:
    timezone = section.get("timezone")
    if not isinstance(timezone, str) or not timezone:
        errors.append("market.timezone must be a non-empty string.")
        return MarketConfig(timezone="UTC")
    try:
        ZoneInfo(timezone)
    except ZoneInfoNotFoundError:
        errors.append(f"market.timezone '{timezone}' is not a known IANA timezone.")
    return MarketConfig(timezone=timezone)


def _validate_data(section: dict, errors: list[str]) -> DataConfig:
    provider = section.get("provider")
    if provider not in _ALLOWED_PROVIDERS:
        errors.append(
            f"data.provider must be one of {sorted(_ALLOWED_PROVIDERS)} "
            f"(got: {provider!r})."
        )
        provider = "mock"
    return DataConfig(provider=provider)


def _validate_positive_int(value, field_name: str, errors: list[str]) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        errors.append(f"{field_name} must be a positive integer (got: {value!r}).")
        return 1
    return value


def _validate_collector(section: dict, errors: list[str]) -> CollectorConfig:
    reconnect = _validate_positive_int(
        section.get("reconnect_seconds"), "collector.reconnect_seconds", errors
    )
    stale = _validate_positive_int(
        section.get("stale_timeout_seconds"), "collector.stale_timeout_seconds", errors
    )
    poll_interval = _validate_positive_int(
        section.get("poll_interval_seconds"), "collector.poll_interval_seconds", errors
    )
    max_failures = _validate_positive_int(
        section.get("max_consecutive_failures"), "collector.max_consecutive_failures", errors
    )
    return CollectorConfig(
        reconnect_seconds=reconnect,
        stale_timeout_seconds=stale,
        poll_interval_seconds=poll_interval,
        max_consecutive_failures=max_failures,
    )


def _validate_prediction(section: dict, errors: list[str]) -> PredictionConfig:
    horizons = section.get("horizons")
    if not isinstance(horizons, list) or not horizons:
        errors.append("prediction.horizons must be a non-empty list.")
        return PredictionConfig(horizons=())
    valid_horizons: list[str] = []
    for h in horizons:
        if isinstance(h, str) and _HORIZON_PATTERN.match(h):
            valid_horizons.append(h)
        else:
            errors.append(
                f"prediction.horizons entry {h!r} is invalid; expected format "
                f"like '5m', '1h', '1d'."
            )
    return PredictionConfig(horizons=tuple(valid_horizons))


def _validate_risk(section: dict, errors: list[str]) -> RiskConfig:
    max_risk = section.get("max_risk_per_trade")
    if not isinstance(max_risk, (int, float)) or isinstance(max_risk, bool) or not (0 < max_risk <= 1):
        errors.append(
            f"risk.max_risk_per_trade must be a number in (0, 1] (got: {max_risk!r})."
        )
        max_risk = 0.01
    max_positions = _validate_positive_int(
        section.get("max_positions"), "risk.max_positions", errors
    )
    return RiskConfig(max_risk_per_trade=float(max_risk), max_positions=max_positions)


def _validate_trading(
    section: dict, env: dict[str, str], errors: list[str]
) -> TradingConfig:
    mode = section.get("mode")
    if mode not in _ALLOWED_TRADING_MODES:
        errors.append(
            f"trading.mode must be one of {sorted(_ALLOWED_TRADING_MODES)} "
            f"(got: {mode!r})."
        )
        mode = "RESEARCH"

    live_enabled = _parse_bool_env(env.get("LIVE_TRADING_ENABLED"), "LIVE_TRADING_ENABLED", errors)

    if mode == "LIVE" and not live_enabled:
        errors.append(
            "trading.mode is 'LIVE' in settings.yaml but LIVE_TRADING_ENABLED "
            "is not 'true' in .env. Live trading stays disabled until both "
            "agree — this is intentional and must not be bypassed."
        )

    return TradingConfig(mode=mode, live_trading_enabled=live_enabled)


def _validate_dashboard(section: dict, errors: list[str]) -> DashboardConfig:
    host = section.get("host")
    port = section.get("port")
    allow_external = section.get("allow_external", False)

    if not isinstance(host, str) or not host:
        errors.append("dashboard.host must be a non-empty string.")
        host = "127.0.0.1"

    if not isinstance(port, int) or isinstance(port, bool) or not (1 <= port <= 65535):
        errors.append(f"dashboard.port must be an integer in 1-65535 (got: {port!r}).")
        port = 8000

    if not isinstance(allow_external, bool):
        errors.append("dashboard.allow_external must be true or false.")
        allow_external = False

    if host not in ("127.0.0.1", "localhost") and not allow_external:
        errors.append(
            f"dashboard.host is '{host}' (not localhost) but "
            f"dashboard.allow_external is false. Per project safety rules the "
            f"dashboard binds to localhost only unless this is explicitly "
            f"acknowledged."
        )

    return DashboardConfig(host=host, port=port, allow_external=allow_external)


def _validate_logging(section: dict, errors: list[str]) -> LoggingConfig:
    level = section.get("level")
    if level not in _ALLOWED_LOG_LEVELS:
        errors.append(
            f"logging.level must be one of {sorted(_ALLOWED_LOG_LEVELS)} "
            f"(got: {level!r})."
        )
        level = "INFO"
    return LoggingConfig(level=level)


def _validate_database(section: dict, errors: list[str]) -> DatabaseConfig:
    path = section.get("path")
    if not isinstance(path, str) or not path.strip():
        errors.append("database.path must be a non-empty string.")
        path = "database/psx.db"
    elif not path.endswith(".db"):
        errors.append(f"database.path must end with '.db' (got: {path!r}).")

    echo = section.get("echo", False)
    if not isinstance(echo, bool):
        errors.append("database.echo must be true or false.")
        echo = False

    return DatabaseConfig(path=path, echo=echo)


def _validate_quality(section: dict, errors: list[str]) -> QualityConfig:
    max_change_percent = section.get("max_change_percent")
    if (
        not isinstance(max_change_percent, (int, float))
        or isinstance(max_change_percent, bool)
        or max_change_percent <= 0
    ):
        errors.append(
            f"quality.max_change_percent must be a positive number "
            f"(got: {max_change_percent!r})."
        )
        max_change_percent = 10.0

    stale_seconds = _validate_positive_int(
        section.get("stale_seconds"), "quality.stale_seconds", errors
    )

    future_tolerance = section.get("future_timestamp_tolerance_seconds")
    if (
        not isinstance(future_tolerance, int)
        or isinstance(future_tolerance, bool)
        or future_tolerance < 0
    ):
        errors.append(
            f"quality.future_timestamp_tolerance_seconds must be a "
            f"non-negative integer (got: {future_tolerance!r})."
        )
        future_tolerance = 5

    return QualityConfig(
        max_change_percent=float(max_change_percent),
        stale_seconds=stale_seconds,
        future_timestamp_tolerance_seconds=future_tolerance,
    )


_ALLOWED_VOLUME_CONVENTIONS = frozenset({"cumulative", "per_tick"})


def _validate_candles(section: dict, errors: list[str]) -> CandleConfig:
    volume_convention = section.get("volume_convention")
    if volume_convention not in _ALLOWED_VOLUME_CONVENTIONS:
        errors.append(
            f"candles.volume_convention must be one of {sorted(_ALLOWED_VOLUME_CONVENTIONS)} "
            f"(got: {volume_convention!r})."
        )
        volume_convention = "cumulative"
    return CandleConfig(volume_convention=volume_convention)


def _validate_period_list(raw: object, field_name: str, errors: list[str]) -> tuple[int, ...]:
    if not isinstance(raw, list) or not raw:
        errors.append(f"{field_name} must be a non-empty list of positive integers.")
        return ()
    periods = []
    for item in raw:
        if not isinstance(item, int) or isinstance(item, bool) or item <= 0:
            errors.append(f"{field_name}: every entry must be a positive integer (got: {item!r}).")
            continue
        periods.append(item)
    if periods and len(set(periods)) != len(periods):
        errors.append(f"{field_name} contains duplicate periods.")
    return tuple(sorted(set(periods)))


def _validate_indicators(section: dict, errors: list[str]) -> IndicatorsConfig:
    sma_periods = _validate_period_list(section.get("sma_periods"), "indicators.sma_periods", errors)
    ema_periods = _validate_period_list(section.get("ema_periods"), "indicators.ema_periods", errors)

    def _int_field(name: str) -> int:
        return _validate_positive_int(section.get(name), f"indicators.{name}", errors)

    rsi_period = _int_field("rsi_period")
    macd_fast_period = _int_field("macd_fast_period")
    macd_slow_period = _int_field("macd_slow_period")
    macd_signal_period = _int_field("macd_signal_period")
    atr_period = _int_field("atr_period")
    adx_period = _int_field("adx_period")
    bollinger_period = _int_field("bollinger_period")
    roc_period = _int_field("roc_period")
    momentum_period = _int_field("momentum_period")
    stochastic_k_period = _int_field("stochastic_k_period")
    stochastic_d_period = _int_field("stochastic_d_period")
    volume_ma_period = _int_field("volume_ma_period")

    if (
        macd_fast_period
        and macd_slow_period
        and macd_fast_period != 1
        and macd_fast_period >= macd_slow_period
    ):
        errors.append(
            f"indicators.macd_fast_period ({macd_fast_period}) must be less than "
            f"macd_slow_period ({macd_slow_period})."
        )

    bollinger_num_std = section.get("bollinger_num_std")
    if (
        not isinstance(bollinger_num_std, (int, float))
        or isinstance(bollinger_num_std, bool)
        or bollinger_num_std <= 0
    ):
        errors.append(
            f"indicators.bollinger_num_std must be a positive number (got: {bollinger_num_std!r})."
        )
        bollinger_num_std = 2.0

    return IndicatorsConfig(
        sma_periods=sma_periods,
        ema_periods=ema_periods,
        rsi_period=rsi_period,
        macd_fast_period=macd_fast_period,
        macd_slow_period=macd_slow_period,
        macd_signal_period=macd_signal_period,
        atr_period=atr_period,
        adx_period=adx_period,
        bollinger_period=bollinger_period,
        bollinger_num_std=float(bollinger_num_std),
        roc_period=roc_period,
        momentum_period=momentum_period,
        stochastic_k_period=stochastic_k_period,
        stochastic_d_period=stochastic_d_period,
        volume_ma_period=volume_ma_period,
    )


def _validate_features(section: dict, errors: list[str]) -> FeaturesConfig:
    return_horizons = _validate_period_list(
        section.get("return_horizons"), "features.return_horizons", errors
    )

    def _int_field(name: str) -> int:
        return _validate_positive_int(section.get(name), f"features.{name}", errors)

    high_low_breakout_period = _int_field("high_low_breakout_period")
    trend_fast_sma_period = _int_field("trend_fast_sma_period")
    trend_slow_sma_period = _int_field("trend_slow_sma_period")
    volatility_period = _int_field("volatility_period")

    if (
        trend_fast_sma_period
        and trend_slow_sma_period
        and trend_fast_sma_period != 1
        and trend_fast_sma_period >= trend_slow_sma_period
    ):
        errors.append(
            f"features.trend_fast_sma_period ({trend_fast_sma_period}) must be less than "
            f"trend_slow_sma_period ({trend_slow_sma_period})."
        )

    return FeaturesConfig(
        return_horizons=return_horizons,
        high_low_breakout_period=high_low_breakout_period,
        trend_fast_sma_period=trend_fast_sma_period,
        trend_slow_sma_period=trend_slow_sma_period,
        volatility_period=volatility_period,
    )


def _validate_labels(section: dict, errors: list[str]) -> LabelsConfig:
    horizons = _validate_period_list(section.get("horizons"), "labels.horizons", errors)

    def _number(name: str, allow_zero: bool) -> float:
        value = section.get(name)
        bad = (
            not isinstance(value, (int, float))
            or isinstance(value, bool)
            or value < 0
            or (value == 0 and not allow_zero)
        )
        if bad:
            kind = "non-negative" if allow_zero else "positive"
            errors.append(f"labels.{name} must be a {kind} number (got: {value!r}).")
            return 1.0
        return float(value)

    return LabelsConfig(
        horizons=horizons,
        neutral_threshold_pct=_number("neutral_threshold_pct", allow_zero=True),
        take_profit_pct=_number("take_profit_pct", allow_zero=False),
        stop_loss_pct=_number("stop_loss_pct", allow_zero=False),
    )


def _validate_baselines(section: dict, errors: list[str]) -> BaselinesConfig:
    fast = _validate_positive_int(section.get("ma_fast_period"), "baselines.ma_fast_period", errors)
    slow = _validate_positive_int(section.get("ma_slow_period"), "baselines.ma_slow_period", errors)
    if fast != 1 and slow and fast >= slow:
        errors.append(
            f"baselines.ma_fast_period ({fast}) must be less than ma_slow_period ({slow})."
        )

    def _num(name: str):
        value = section.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            errors.append(f"baselines.{name} must be a number (got: {value!r}).")
            return None
        return float(value)

    oversold, overbought = _num("rsi_oversold"), _num("rsi_overbought")
    if oversold is not None and overbought is not None:
        if not (0 < oversold < overbought < 100):
            errors.append(
                f"baselines: need 0 < rsi_oversold ({oversold}) < rsi_overbought ({overbought}) < 100."
            )
    train_fraction = _num("train_fraction")
    if train_fraction is not None and not (0 < train_fraction < 1):
        errors.append(f"baselines.train_fraction must be between 0 and 1 exclusive (got: {train_fraction}).")

    seed = section.get("random_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        errors.append(f"baselines.random_seed must be a non-negative integer (got: {seed!r}).")
        seed = 0

    return BaselinesConfig(
        ma_fast_period=fast,
        ma_slow_period=slow,
        rsi_oversold=oversold if oversold is not None else 30.0,
        rsi_overbought=overbought if overbought is not None else 70.0,
        train_fraction=train_fraction if train_fraction is not None else 0.7,
        random_seed=seed,
    )


_ALLOWED_MODEL_NAMES = ("logistic", "random_forest", "gradient_boosting", "hist_gradient_boosting")


def _validate_models(section: dict, errors: list[str]) -> ModelsConfig:
    def _fraction(name: str) -> float:
        value = section.get(name)
        if not isinstance(value, (int, float)) or isinstance(value, bool) or not (0 < value < 1):
            errors.append(f"models.{name} must be a number between 0 and 1 exclusive (got: {value!r}).")
            return 0.5
        return float(value)

    train_fraction = _fraction("train_fraction")
    validation_fraction = _fraction("validation_fraction")
    if train_fraction + validation_fraction > 0.9:
        errors.append(
            f"models: train_fraction + validation_fraction ({train_fraction + validation_fraction:.2f}) "
            f"must leave at least 10% of the data for the untouched test segment."
        )

    seed = section.get("random_seed")
    if not isinstance(seed, int) or isinstance(seed, bool) or seed < 0:
        errors.append(f"models.random_seed must be a non-negative integer (got: {seed!r}).")
        seed = 0

    enabled_raw = section.get("enabled")
    enabled: list[str] = []
    if not isinstance(enabled_raw, list) or not enabled_raw:
        errors.append("models.enabled must be a non-empty list of model names.")
    else:
        for name in enabled_raw:
            if name not in _ALLOWED_MODEL_NAMES:
                errors.append(f"models.enabled: unknown model {name!r}; allowed: {list(_ALLOWED_MODEL_NAMES)}.")
            elif name in enabled:
                errors.append(f"models.enabled: duplicate model {name!r}.")
            else:
                enabled.append(name)

    min_rows = _validate_positive_int(section.get("min_rows"), "models.min_rows", errors)
    if min_rows < 50:
        errors.append(f"models.min_rows must be at least 50 (got: {min_rows}).")
    permutation_repeats = _validate_positive_int(
        section.get("permutation_repeats"), "models.permutation_repeats", errors
    )
    return ModelsConfig(
        train_fraction=train_fraction,
        validation_fraction=validation_fraction,
        random_seed=seed,
        enabled=tuple(enabled),
        min_rows=min_rows,
        permutation_repeats=permutation_repeats,
    )


def _num_field(section: dict, name: str, label: str, errors: list[str], *, minimum: float = 0.0,
               maximum: float | None = None, exclusive_min: bool = False, default: float = 0.0) -> float:
    value = section.get(name)
    ok = isinstance(value, (int, float)) and not isinstance(value, bool)
    if ok:
        ok = (value > minimum) if exclusive_min else (value >= minimum)
        if ok and maximum is not None:
            ok = value <= maximum
    if not ok:
        bound = f"{'>' if exclusive_min else '>='} {minimum}" + (f" and <= {maximum}" if maximum is not None else "")
        errors.append(f"{label}.{name} must be a number {bound} (got: {value!r}).")
        return default
    return float(value)


def _bool_field(section: dict, name: str, label: str, errors: list[str]) -> bool:
    value = section.get(name)
    if not isinstance(value, bool):
        errors.append(f"{label}.{name} must be true or false (got: {value!r}).")
        return False
    return value


def _validate_backtest(section: dict, errors: list[str]) -> BacktestConfig:
    L = "backtest"
    delay = section.get("execution_delay_bars")
    if not isinstance(delay, int) or isinstance(delay, bool) or delay < 1:
        errors.append(
            f"backtest.execution_delay_bars must be an integer >= 1 (got: {delay!r}). "
            f"0 would fill an order at the same bar that produced the signal — look-ahead."
        )
        delay = 1
    return BacktestConfig(
        initial_cash=_num_field(section, "initial_cash", L, errors, exclusive_min=True, default=1.0),
        position_fraction=_num_field(section, "position_fraction", L, errors, maximum=1.0, exclusive_min=True, default=1.0),
        execution_delay_bars=delay,
        slippage_bps=_num_field(section, "slippage_bps", L, errors, maximum=1000.0),
        max_participation_pct=_num_field(section, "max_participation_pct", L, errors, maximum=100.0),
        stop_loss_pct=_num_field(section, "stop_loss_pct", L, errors, maximum=100.0),
        take_profit_pct=_num_field(section, "take_profit_pct", L, errors, maximum=1000.0),
        min_price=_num_field(section, "min_price", L, errors),
        min_avg_volume=_num_field(section, "min_avg_volume", L, errors),
        avg_volume_bars=_validate_positive_int(section.get("avg_volume_bars"), "backtest.avg_volume_bars", errors),
        risk_free_rate_annual_pct=_num_field(section, "risk_free_rate_annual_pct", L, errors, maximum=100.0),
        liquidate_at_end=_bool_field(section, "liquidate_at_end", L, errors),
        min_trades_for_confidence=_validate_positive_int(
            section.get("min_trades_for_confidence"), "backtest.min_trades_for_confidence", errors
        ),
    )


def _validate_costs(raw: dict, errors: list[str]) -> CostsConfig:
    brokerage = raw.get("brokerage") if isinstance(raw.get("brokerage"), dict) else {}
    if not brokerage:
        errors.append("costs.yaml: missing or invalid section 'brokerage'.")
    cgt = raw.get("capital_gains_tax") if isinstance(raw.get("capital_gains_tax"), dict) else {}
    if not cgt:
        errors.append("costs.yaml: missing or invalid section 'capital_gains_tax'.")

    commission_rate = _num_field(brokerage, "commission_rate_pct", "costs.brokerage", errors, maximum=100.0)
    commission_max = _num_field(brokerage, "commission_max_pct", "costs.brokerage", errors, maximum=100.0, default=2.5)
    if commission_rate > commission_max:
        errors.append(
            f"costs.brokerage: commission_rate_pct ({commission_rate}) must not exceed "
            f"commission_max_pct ({commission_max})."
        )
    return CostsConfig(
        commission_rate_pct=commission_rate,
        commission_min_per_share=_num_field(brokerage, "commission_min_per_share", "costs.brokerage", errors),
        commission_max_pct=commission_max,
        day_trade_one_side_only=_bool_field(brokerage, "day_trade_one_side_only", "costs.brokerage", errors),
        sales_tax_on_commission_pct=_num_field(raw, "sales_tax_on_commission_pct", "costs", errors, maximum=100.0),
        cdc_per_share=_num_field(raw, "cdc_per_share", "costs", errors),
        other_levies_pct_per_side=_num_field(raw, "other_levies_pct_per_side", "costs", errors, maximum=100.0),
        cgt_apply=_bool_field(cgt, "apply", "costs.capital_gains_tax", errors),
        tax_filer=_bool_field(cgt, "tax_filer", "costs.capital_gains_tax", errors),
        cgt_filer_rate_pct=_num_field(cgt, "filer_rate_pct", "costs.capital_gains_tax", errors, maximum=100.0),
        cgt_non_filer_rate_pct=_num_field(cgt, "non_filer_rate_pct", "costs.capital_gains_tax", errors, maximum=100.0),
    )


def _validate_symbol_list(raw: object, field_name: str, errors: list[str]) -> tuple[str, ...]:
    if not isinstance(raw, list) or not raw:
        errors.append(f"symbols.yaml: '{field_name}' must be a non-empty list.")
        return ()
    if not all(isinstance(s, str) and s for s in raw):
        errors.append(f"symbols.yaml: every entry in '{field_name}' must be a non-empty string.")
        return tuple(str(s) for s in raw if isinstance(s, str) and s)

    normalized = [s.upper() for s in raw]
    if normalized != raw:
        errors.append(
            f"symbols.yaml: entries in '{field_name}' must be upper-case "
            f"(e.g. use 'KSE100', not 'kse100')."
        )
    if len(set(normalized)) != len(normalized):
        errors.append(f"symbols.yaml: '{field_name}' contains duplicate entries.")

    return tuple(normalized)


def _validate_symbols(raw: dict, errors: list[str]) -> SymbolsConfig:
    indices = _validate_symbol_list(raw.get("indices"), "indices", errors)
    stocks = _validate_symbol_list(raw.get("stocks"), "stocks", errors)
    watchlist = _validate_symbol_list(raw.get("watchlist"), "watchlist", errors)

    known = set(indices) | set(stocks)
    unknown_watchlist_entries = [w for w in watchlist if w not in known]
    if unknown_watchlist_entries:
        errors.append(
            f"symbols.yaml: watchlist entries {unknown_watchlist_entries} do not "
            f"appear in 'indices' or 'stocks'."
        )

    return SymbolsConfig(indices=indices, stocks=stocks, watchlist=watchlist)


def load_config(
    settings_path: Path | None = None,
    symbols_path: Path | None = None,
    env_path: Path | None = None,
    costs_path: Path | None = None,
) -> AppConfig:
    """Load, validate, and return the full application configuration.

    Args:
        settings_path: Override for `config/settings.yaml` (used by tests).
        symbols_path: Override for `config/symbols.yaml` (used by tests).
        env_path: Override for `.env` (used by tests).
        costs_path: Override for `config/costs.yaml` (used by tests).

    Returns:
        A fully validated, immutable `AppConfig`.

    Raises:
        ConfigValidationError: if any value in either YAML file, or the
            relevant environment variables, fails validation. The exception
            carries the full list of problems found, not just the first one.
    """
    errors: list[str] = []

    settings_path = settings_path or resolve_path("config/settings.yaml")
    symbols_path = symbols_path or resolve_path("config/symbols.yaml")
    env_path = env_path or resolve_path(".env")
    costs_path = costs_path or resolve_path("config/costs.yaml")

    settings_raw = _load_yaml(settings_path, errors)
    symbols_raw = _load_yaml(symbols_path, errors)
    costs_raw = _load_yaml(costs_path, errors)
    env_values = dotenv_values(env_path) if env_path.is_file() else {}

    if errors:
        # If the files themselves are missing/unparsable, section-level
        # validation below would just produce a wall of confusing follow-on
        # errors. Stop here with the root cause instead.
        raise ConfigValidationError(errors)

    market = _validate_market(_require_section(settings_raw, "market", errors), errors)
    data = _validate_data(_require_section(settings_raw, "data", errors), errors)
    collector = _validate_collector(_require_section(settings_raw, "collector", errors), errors)
    prediction = _validate_prediction(_require_section(settings_raw, "prediction", errors), errors)
    risk = _validate_risk(_require_section(settings_raw, "risk", errors), errors)
    trading = _validate_trading(_require_section(settings_raw, "trading", errors), env_values, errors)
    dashboard = _validate_dashboard(_require_section(settings_raw, "dashboard", errors), errors)
    logging_cfg = _validate_logging(_require_section(settings_raw, "logging", errors), errors)
    database = _validate_database(_require_section(settings_raw, "database", errors), errors)
    quality = _validate_quality(_require_section(settings_raw, "quality", errors), errors)
    candles = _validate_candles(_require_section(settings_raw, "candles", errors), errors)
    indicators = _validate_indicators(_require_section(settings_raw, "indicators", errors), errors)
    features = _validate_features(_require_section(settings_raw, "features", errors), errors)
    labels = _validate_labels(_require_section(settings_raw, "labels", errors), errors)
    baselines = _validate_baselines(_require_section(settings_raw, "baselines", errors), errors)
    models = _validate_models(_require_section(settings_raw, "models", errors), errors)
    backtest = _validate_backtest(_require_section(settings_raw, "backtest", errors), errors)
    costs = _validate_costs(costs_raw, errors)
    symbols = _validate_symbols(symbols_raw, errors)

    if errors:
        raise ConfigValidationError(errors)

    return AppConfig(
        market=market,
        data=data,
        collector=collector,
        prediction=prediction,
        risk=risk,
        trading=trading,
        dashboard=dashboard,
        logging=logging_cfg,
        database=database,
        quality=quality,
        candles=candles,
        indicators=indicators,
        features=features,
        labels=labels,
        baselines=baselines,
        models=models,
        backtest=backtest,
        costs=costs,
        symbols=symbols,
    )
