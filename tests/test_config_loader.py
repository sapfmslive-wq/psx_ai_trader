"""Tests for src.config.loader (Phase 2)."""

from __future__ import annotations

from pathlib import Path

import pytest

from src.config.loader import ConfigValidationError, load_config

VALID_SETTINGS = """
market:
  timezone: Asia/Karachi
data:
  provider: mock
collector:
  reconnect_seconds: 5
  stale_timeout_seconds: 30
  poll_interval_seconds: 15
  max_consecutive_failures: 10
prediction:
  horizons: ["5m", "15m", "30m"]
risk:
  max_risk_per_trade: 0.01
  max_positions: 5
trading:
  mode: RESEARCH
dashboard:
  host: 127.0.0.1
  port: 8000
  allow_external: false
logging:
  level: INFO
database:
  path: database/psx.db
  echo: false
quality:
  max_change_percent: 10.0
  stale_seconds: 300
  future_timestamp_tolerance_seconds: 5
candles:
  volume_convention: cumulative
indicators:
  sma_periods: [5, 10, 20]
  ema_periods: [5, 10, 20]
  rsi_period: 14
  macd_fast_period: 12
  macd_slow_period: 26
  macd_signal_period: 9
  atr_period: 14
  adx_period: 14
  bollinger_period: 20
  bollinger_num_std: 2.0
  roc_period: 12
  momentum_period: 10
  stochastic_k_period: 14
  stochastic_d_period: 3
  volume_ma_period: 20
features:
  return_horizons: [1, 5, 15, 60]
  high_low_breakout_period: 20
  trend_fast_sma_period: 20
  trend_slow_sma_period: 50
  volatility_period: 20
labels:
  horizons: [5, 15, 30]
  neutral_threshold_pct: 0.1
  take_profit_pct: 1.0
  stop_loss_pct: 0.5
baselines:
  ma_fast_period: 10
  ma_slow_period: 30
  rsi_oversold: 30
  rsi_overbought: 70
  train_fraction: 0.7
  random_seed: 42
models:
  train_fraction: 0.6
  validation_fraction: 0.2
  random_seed: 42
  enabled: [logistic, random_forest, gradient_boosting, hist_gradient_boosting]
  min_rows: 300
  permutation_repeats: 5
backtest:
  initial_cash: 1000000
  position_fraction: 1.0
  execution_delay_bars: 1
  slippage_bps: 10.0
  max_participation_pct: 10.0
  stop_loss_pct: 0.0
  take_profit_pct: 0.0
  min_price: 0.0
  min_avg_volume: 0.0
  avg_volume_bars: 20
  risk_free_rate_annual_pct: 0.0
  liquidate_at_end: true
  min_trades_for_confidence: 30
"""

VALID_SYMBOLS = """
indices:
  - KSE100
  - KMI30
stocks:
  - THCCL
watchlist:
  - THCCL
"""


def _write(tmp_path: Path, name: str, content: str) -> Path:
    path = tmp_path / name
    path.write_text(content, encoding="utf-8")
    return path


VALID_COSTS = (Path(__file__).resolve().parent.parent / "config" / "costs.yaml").read_text(encoding="utf-8")


def _load(tmp_path: Path, settings: str = VALID_SETTINGS, symbols: str = VALID_SYMBOLS, env: str = "",
          costs: str | None = None):
    settings_path = _write(tmp_path, "settings.yaml", settings)
    symbols_path = _write(tmp_path, "symbols.yaml", symbols)
    env_path = _write(tmp_path, ".env", env)
    costs_path = _write(tmp_path, "costs.yaml", costs) if costs is not None else None
    return load_config(settings_path=settings_path, symbols_path=symbols_path, env_path=env_path,
                       costs_path=costs_path)


def test_valid_config_loads_and_parses_expected_values(tmp_path):
    config = _load(tmp_path)

    assert config.market.timezone == "Asia/Karachi"
    assert config.data.provider == "mock"
    assert config.collector.reconnect_seconds == 5
    assert config.collector.poll_interval_seconds == 15
    assert config.collector.max_consecutive_failures == 10
    assert config.prediction.horizons == ("5m", "15m", "30m")
    assert config.risk.max_risk_per_trade == 0.01
    assert config.trading.mode == "RESEARCH"
    assert config.trading.live_trading_enabled is False
    assert config.dashboard.host == "127.0.0.1"
    assert config.symbols.indices == ("KSE100", "KMI30")
    assert config.symbols.stocks == ("THCCL",)
    assert config.symbols.watchlist == ("THCCL",)
    assert config.database.path == "database/psx.db"
    assert config.database.echo is False
    assert config.candles.volume_convention == "cumulative"
    assert config.indicators.sma_periods == (5, 10, 20)
    assert config.indicators.rsi_period == 14
    assert config.indicators.bollinger_num_std == 2.0
    assert config.features.return_horizons == (1, 5, 15, 60)
    assert config.features.trend_fast_sma_period == 20
    assert config.labels.horizons == (5, 15, 30)
    assert config.labels.take_profit_pct == 1.0
    assert config.baselines.ma_slow_period == 30
    assert config.baselines.random_seed == 42
    assert config.models.enabled == ("logistic", "random_forest", "gradient_boosting", "hist_gradient_boosting")
    assert config.models.min_rows == 300
    assert config.backtest.execution_delay_bars == 1
    assert config.backtest.slippage_bps == 10.0
    assert config.quality.max_change_percent == 10.0
    assert config.quality.stale_seconds == 300
    assert config.quality.future_timestamp_tolerance_seconds == 5


def test_missing_settings_file_raises(tmp_path):
    symbols_path = _write(tmp_path, "symbols.yaml", VALID_SYMBOLS)
    env_path = _write(tmp_path, ".env", "")
    with pytest.raises(ConfigValidationError) as exc_info:
        load_config(
            settings_path=tmp_path / "does_not_exist.yaml",
            symbols_path=symbols_path,
            env_path=env_path,
        )
    assert any("not found" in e for e in exc_info.value.errors)


def test_invalid_timezone_is_rejected(tmp_path):
    bad_settings = VALID_SETTINGS.replace("Asia/Karachi", "Mars/Olympus_Mons")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("timezone" in e for e in exc_info.value.errors)


def test_invalid_data_provider_is_rejected(tmp_path):
    bad_settings = VALID_SETTINGS.replace("provider: mock", "provider: some_random_thing")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("data.provider" in e for e in exc_info.value.errors)


def test_invalid_horizon_format_is_rejected(tmp_path):
    bad_settings = VALID_SETTINGS.replace('["5m", "15m", "30m"]', '["five minutes"]')
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("horizons" in e for e in exc_info.value.errors)


def test_risk_out_of_range_is_rejected(tmp_path):
    bad_settings = VALID_SETTINGS.replace("max_risk_per_trade: 0.01", "max_risk_per_trade: 1.5")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("max_risk_per_trade" in e for e in exc_info.value.errors)


def test_live_mode_without_env_flag_is_rejected(tmp_path):
    bad_settings = VALID_SETTINGS.replace("mode: RESEARCH", "mode: LIVE")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings, env="LIVE_TRADING_ENABLED=false")
    assert any("LIVE" in e for e in exc_info.value.errors)


def test_live_mode_with_env_flag_true_is_accepted(tmp_path):
    good_settings = VALID_SETTINGS.replace("mode: RESEARCH", "mode: LIVE")
    config = _load(tmp_path, settings=good_settings, env="LIVE_TRADING_ENABLED=true")
    assert config.trading.mode == "LIVE"
    assert config.trading.live_trading_enabled is True


def test_dashboard_non_localhost_without_allow_external_is_rejected(tmp_path):
    bad_settings = VALID_SETTINGS.replace(
        "host: 127.0.0.1\n  port: 8000\n  allow_external: false",
        "host: 0.0.0.0\n  port: 8000\n  allow_external: false",
    )
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("dashboard" in e.lower() for e in exc_info.value.errors)


def test_duplicate_symbol_entries_are_rejected(tmp_path):
    bad_symbols = VALID_SYMBOLS.replace("stocks:\n  - THCCL", "stocks:\n  - THCCL\n  - THCCL")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, symbols=bad_symbols)
    assert any("duplicate" in e.lower() for e in exc_info.value.errors)


def test_watchlist_entry_not_in_symbols_is_rejected(tmp_path):
    bad_symbols = VALID_SYMBOLS.replace("watchlist:\n  - THCCL", "watchlist:\n  - UNKNOWN")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, symbols=bad_symbols)
    assert any("watchlist" in e.lower() for e in exc_info.value.errors)


def test_multiple_errors_are_aggregated_in_one_exception(tmp_path):
    bad_settings = (
        VALID_SETTINGS
        .replace("provider: mock", "provider: nonsense")
        .replace("max_risk_per_trade: 0.01", "max_risk_per_trade: 5")
    )
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert len(exc_info.value.errors) >= 2


def test_database_path_without_db_extension_is_rejected(tmp_path):
    bad_settings = VALID_SETTINGS.replace("path: database/psx.db", "path: database/psx.txt")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("database.path" in e for e in exc_info.value.errors)


def test_database_echo_must_be_boolean(tmp_path):
    bad_settings = VALID_SETTINGS.replace("echo: false", "echo: not_a_bool")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("database.echo" in e for e in exc_info.value.errors)


def test_collector_poll_interval_must_be_positive_int(tmp_path):
    bad_settings = VALID_SETTINGS.replace("poll_interval_seconds: 15", "poll_interval_seconds: 0")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("collector.poll_interval_seconds" in e for e in exc_info.value.errors)


def test_collector_max_consecutive_failures_must_be_positive_int(tmp_path):
    bad_settings = VALID_SETTINGS.replace("max_consecutive_failures: 10", "max_consecutive_failures: -1")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("collector.max_consecutive_failures" in e for e in exc_info.value.errors)


def test_candles_volume_convention_must_be_recognized(tmp_path):
    bad_settings = VALID_SETTINGS.replace("volume_convention: cumulative", "volume_convention: nonsense")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("candles.volume_convention" in e for e in exc_info.value.errors)


def test_indicators_sma_periods_must_be_positive_ints(tmp_path):
    bad_settings = VALID_SETTINGS.replace("sma_periods: [5, 10, 20]", "sma_periods: [5, -10, 20]")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("indicators.sma_periods" in e for e in exc_info.value.errors)


def test_indicators_sma_periods_rejects_empty_list(tmp_path):
    bad_settings = VALID_SETTINGS.replace("sma_periods: [5, 10, 20]", "sma_periods: []")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("indicators.sma_periods" in e for e in exc_info.value.errors)


def test_indicators_rsi_period_must_be_positive_int(tmp_path):
    bad_settings = VALID_SETTINGS.replace("rsi_period: 14", "rsi_period: 0")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("indicators.rsi_period" in e for e in exc_info.value.errors)


def test_indicators_macd_fast_must_be_less_than_slow(tmp_path):
    bad_settings = VALID_SETTINGS.replace("macd_fast_period: 12", "macd_fast_period: 30")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("macd_fast_period" in e for e in exc_info.value.errors)


def test_indicators_bollinger_num_std_must_be_positive(tmp_path):
    bad_settings = VALID_SETTINGS.replace("bollinger_num_std: 2.0", "bollinger_num_std: -1")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("indicators.bollinger_num_std" in e for e in exc_info.value.errors)


def test_features_return_horizons_must_be_positive_ints(tmp_path):
    bad_settings = VALID_SETTINGS.replace("return_horizons: [1, 5, 15, 60]", "return_horizons: [1, 0]")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("features.return_horizons" in e for e in exc_info.value.errors)


def test_baselines_ma_fast_must_be_less_than_slow(tmp_path):
    bad = VALID_SETTINGS.replace("ma_fast_period: 10", "ma_fast_period: 40")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad)
    assert any("ma_fast_period" in e for e in exc_info.value.errors)


def test_baselines_rsi_thresholds_must_be_ordered(tmp_path):
    bad = VALID_SETTINGS.replace("rsi_oversold: 30", "rsi_oversold: 80")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad)
    assert any("rsi_oversold" in e for e in exc_info.value.errors)


def test_baselines_train_fraction_must_be_between_zero_and_one(tmp_path):
    for bad_value in ("0", "1", "1.5"):
        bad = VALID_SETTINGS.replace("train_fraction: 0.7", f"train_fraction: {bad_value}")
        with pytest.raises(ConfigValidationError) as exc_info:
            _load(tmp_path, settings=bad)
        assert any("train_fraction" in e for e in exc_info.value.errors)


def test_models_fractions_must_leave_a_test_segment(tmp_path):
    bad = VALID_SETTINGS.replace("train_fraction: 0.6", "train_fraction: 0.85")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad)
    assert any("untouched test segment" in e for e in exc_info.value.errors)


def test_models_unknown_or_duplicate_model_names_are_rejected(tmp_path):
    for bad_list in ("[logistic, wizardry]", "[logistic, logistic]", "[]"):
        bad = VALID_SETTINGS.replace(
            "enabled: [logistic, random_forest, gradient_boosting, hist_gradient_boosting]", f"enabled: {bad_list}"
        )
        with pytest.raises(ConfigValidationError) as exc_info:
            _load(tmp_path, settings=bad)
        assert any("models.enabled" in e for e in exc_info.value.errors)


def test_models_min_rows_has_a_floor(tmp_path):
    bad = VALID_SETTINGS.replace("min_rows: 300", "min_rows: 10")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad)
    assert any("models.min_rows" in e for e in exc_info.value.errors)


def test_shipped_costs_file_loads_with_the_documented_sourced_defaults(tmp_path):
    c = _load(tmp_path).costs  # no costs override: reads the real config/costs.yaml
    assert (c.commission_rate_pct, c.commission_min_per_share, c.commission_max_pct) == (0.15, 0.03, 2.5)
    assert c.day_trade_one_side_only is False  # conservative: charge both sides
    assert c.sales_tax_on_commission_pct == 16.0 and c.cdc_per_share == 0.005
    assert c.other_levies_pct_per_side == 0.0  # deliberately unmodeled; no source gave a figure
    assert (c.cgt_filer_rate_pct, c.cgt_non_filer_rate_pct) == (15.0, 20.0)


def test_cgt_rate_follows_filer_status(tmp_path):
    assert _load(tmp_path).costs.cgt_rate_pct == 15.0
    non_filer = VALID_COSTS.replace("tax_filer: true", "tax_filer: false")
    assert _load(tmp_path, costs=non_filer).costs.cgt_rate_pct == 20.0


def test_costs_commission_rate_may_not_exceed_the_regulatory_ceiling(tmp_path):
    bad = VALID_COSTS.replace("commission_rate_pct: 0.15", "commission_rate_pct: 3.0")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, costs=bad)
    assert any("must not exceed commission_max_pct" in e for e in exc_info.value.errors)


def test_costs_reject_negative_numbers_and_non_boolean_flags(tmp_path):
    for old, new, needle in (
        ("cdc_per_share: 0.005", "cdc_per_share: -1", "cdc_per_share"),
        ("sales_tax_on_commission_pct: 16.0", "sales_tax_on_commission_pct: 150", "sales_tax_on_commission_pct"),
        ("day_trade_one_side_only: false", "day_trade_one_side_only: maybe", "day_trade_one_side_only"),
        ("apply: true", "apply: 1", "capital_gains_tax.apply"),
    ):
        bad = VALID_COSTS.replace(old, new)
        assert bad != VALID_COSTS, old
        with pytest.raises(ConfigValidationError) as exc_info:
            _load(tmp_path, costs=bad)
        assert any(needle in e for e in exc_info.value.errors), (old, exc_info.value.errors)


def test_costs_missing_sections_are_reported(tmp_path):
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, costs="cdc_per_share: 0.005\n")
    errors = " ".join(exc_info.value.errors)
    assert "brokerage" in errors and "capital_gains_tax" in errors


def test_backtest_execution_delay_must_be_at_least_one_bar(tmp_path):
    for bad_value in ("0", "-1", "1.5"):
        bad = VALID_SETTINGS.replace("execution_delay_bars: 1", f"execution_delay_bars: {bad_value}")
        with pytest.raises(ConfigValidationError) as exc_info:
            _load(tmp_path, settings=bad)
        assert any("execution_delay_bars" in e and "look-ahead" in e for e in exc_info.value.errors), bad_value


def test_backtest_rejects_bad_sizing_slippage_and_flags(tmp_path):
    for old, new, needle in (
        ("position_fraction: 1.0", "position_fraction: 1.5", "position_fraction"),
        ("position_fraction: 1.0", "position_fraction: 0", "position_fraction"),
        ("slippage_bps: 10.0", "slippage_bps: -1", "slippage_bps"),
        ("initial_cash: 1000000", "initial_cash: 0", "initial_cash"),
        ("max_participation_pct: 10.0", "max_participation_pct: 150", "max_participation_pct"),
        ("liquidate_at_end: true", "liquidate_at_end: sometimes", "liquidate_at_end"),
    ):
        bad = VALID_SETTINGS.replace(old, new)
        assert bad != VALID_SETTINGS, old
        with pytest.raises(ConfigValidationError) as exc_info:
            _load(tmp_path, settings=bad)
        assert any(needle in e for e in exc_info.value.errors), (old, exc_info.value.errors)


def test_labels_horizons_must_be_positive_ints(tmp_path):
    bad = VALID_SETTINGS.replace("horizons: [5, 15, 30]", "horizons: [5, -1]")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad)
    assert any("labels.horizons" in e for e in exc_info.value.errors)


def test_labels_take_profit_must_be_positive(tmp_path):
    bad = VALID_SETTINGS.replace("take_profit_pct: 1.0", "take_profit_pct: 0")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad)
    assert any("labels.take_profit_pct" in e for e in exc_info.value.errors)


def test_labels_neutral_threshold_may_be_zero_but_not_negative(tmp_path):
    ok = VALID_SETTINGS.replace("neutral_threshold_pct: 0.1", "neutral_threshold_pct: 0")
    assert _load(tmp_path, settings=ok).labels.neutral_threshold_pct == 0.0
    bad = VALID_SETTINGS.replace("neutral_threshold_pct: 0.1", "neutral_threshold_pct: -0.1")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad)
    assert any("labels.neutral_threshold_pct" in e for e in exc_info.value.errors)


def test_features_trend_fast_must_be_less_than_slow(tmp_path):
    bad_settings = VALID_SETTINGS.replace("trend_fast_sma_period: 20", "trend_fast_sma_period: 60")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("trend_fast_sma_period" in e for e in exc_info.value.errors)


def test_quality_max_change_percent_must_be_positive(tmp_path):
    bad_settings = VALID_SETTINGS.replace("max_change_percent: 10.0", "max_change_percent: -5.0")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("quality.max_change_percent" in e for e in exc_info.value.errors)


def test_quality_stale_seconds_must_be_positive_int(tmp_path):
    bad_settings = VALID_SETTINGS.replace("stale_seconds: 300", "stale_seconds: 0")
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("quality.stale_seconds" in e for e in exc_info.value.errors)


def test_quality_future_timestamp_tolerance_must_be_non_negative_int(tmp_path):
    bad_settings = VALID_SETTINGS.replace(
        "future_timestamp_tolerance_seconds: 5", "future_timestamp_tolerance_seconds: -1"
    )
    with pytest.raises(ConfigValidationError) as exc_info:
        _load(tmp_path, settings=bad_settings)
    assert any("quality.future_timestamp_tolerance_seconds" in e for e in exc_info.value.errors)
