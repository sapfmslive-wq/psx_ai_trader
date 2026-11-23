"""Indicator engine (Phase 11): computes every configured indicator from a
symbol's candles and stores the results as `Feature` rows.

Two layers, same split as every other module in this project:

    candles_to_dataframe() - pure: turns a list of Candle ORM rows into a
                              plain pandas DataFrame (open/high/low/close/
                              volume indexed by open_time). No config, no
                              indicator math.
    compute_indicators()    - pure: given that DataFrame and an
                              IndicatorsConfig, returns one wide DataFrame
                              with every indicator as its own column,
                              named consistently (e.g. "sma_20", "rsi_14",
                              "macd_line", "bb_upper_20").
    compute_and_store_indicators() - the only function that touches the
                              database: reads candles, calls the two pure
                              functions above, writes each (timestamp,
                              feature_name) pair as a Feature row.

`feature_version` is a module-level constant, bumped whenever a formula or
default changes in a way that would make old and new values for the same
feature_name not directly comparable (Section 34's model/feature
versioning discipline, applied here for the first time since a real
feature now exists).
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from src.indicators import momentum, trend, volatility
from src.indicators import volume as vol_ind

FEATURE_VERSION = "indicators_v1"


def candles_to_dataframe(candles: list) -> pd.DataFrame:  # noqa: ANN001 — list[Candle]
    """Turn Candle ORM rows (already ordered by open_time) into a plain
    DataFrame with columns open/high/low/close/volume, indexed by
    open_time. No database dependency — `candles` can be any object with
    those attributes, which is what makes this independently testable."""
    if not candles:
        return pd.DataFrame(columns=["open", "high", "low", "close", "volume"])
    return pd.DataFrame(
        {
            "open": [c.open for c in candles],
            "high": [c.high for c in candles],
            "low": [c.low for c in candles],
            "close": [c.close for c in candles],
            "volume": [c.volume for c in candles],
        },
        index=pd.DatetimeIndex([c.open_time for c in candles], name="open_time"),
    )


def compute_indicators(df: pd.DataFrame, config) -> pd.DataFrame:  # noqa: ANN001 — IndicatorsConfig
    """Compute every configured indicator from a candle DataFrame.

    Returns a wide DataFrame, one column per indicator/period, indexed the
    same way as `df`. Columns whose underlying candle volume is entirely
    missing (None) are simply all-NaN — never fabricated.
    """
    if df.empty:
        return pd.DataFrame(index=df.index)

    close, high, low, volume = df["close"], df["high"], df["low"], df["volume"]
    columns: dict[str, pd.Series] = {}

    for period in config.sma_periods:
        columns[f"sma_{period}"] = trend.sma(close, period)
    for period in config.ema_periods:
        columns[f"ema_{period}"] = trend.ema(close, period)

    macd_df = trend.macd(close, config.macd_fast_period, config.macd_slow_period, config.macd_signal_period)
    columns["macd_line"] = macd_df["macd"]
    columns["macd_signal"] = macd_df["signal"]
    columns["macd_histogram"] = macd_df["histogram"]

    adx_df = trend.adx(high, low, close, config.adx_period)
    columns[f"adx_{config.adx_period}"] = adx_df["adx"]
    columns[f"plus_di_{config.adx_period}"] = adx_df["plus_di"]
    columns[f"minus_di_{config.adx_period}"] = adx_df["minus_di"]

    columns[f"rsi_{config.rsi_period}"] = momentum.rsi(close, config.rsi_period)
    columns[f"roc_{config.roc_period}"] = momentum.roc(close, config.roc_period)
    columns[f"momentum_{config.momentum_period}"] = momentum.momentum(close, config.momentum_period)

    stoch_df = momentum.stochastic(high, low, close, config.stochastic_k_period, config.stochastic_d_period)
    columns[f"stoch_k_{config.stochastic_k_period}"] = stoch_df["percent_k"]
    columns[f"stoch_d_{config.stochastic_d_period}"] = stoch_df["percent_d"]

    columns[f"atr_{config.atr_period}"] = volatility.atr(high, low, close, config.atr_period)

    bb_df = volatility.bollinger_bands(close, config.bollinger_period, config.bollinger_num_std)
    columns[f"bb_middle_{config.bollinger_period}"] = bb_df["middle"]
    columns[f"bb_upper_{config.bollinger_period}"] = bb_df["upper"]
    columns[f"bb_lower_{config.bollinger_period}"] = bb_df["lower"]

    columns[f"rolling_volatility_{config.bollinger_period}"] = volatility.rolling_volatility(
        close, config.bollinger_period
    )

    columns["obv"] = vol_ind.obv(close, volume)
    columns[f"volume_ma_{config.volume_ma_period}"] = vol_ind.volume_moving_average(
        volume, config.volume_ma_period
    )
    vwap_series = vol_ind.vwap(high, low, close, volume)
    columns["vwap"] = vwap_series
    columns["vwap_distance_pct"] = vol_ind.price_distance_from_vwap(close, vwap_series)

    return pd.DataFrame(columns, index=df.index)


def compute_and_store_indicators(
    config,  # noqa: ANN001 — AppConfig
    session_factory,  # noqa: ANN001
    symbol: str,
    timeframe: str,
    start: dt.datetime,
    end: dt.datetime,
) -> dict[str, int]:
    """Read candles for [start, end), compute every configured indicator,
    and store each (timestamp, feature_name) as a Feature row.

    Returns:
        {"written": N, "duplicate": M} — counts across every feature/timestamp
        combination attempted.
    """
    from src.database.repository import get_candles, insert_feature

    with session_factory() as session:
        candles = get_candles(session, symbol, timeframe, start, end)

    df = candles_to_dataframe(candles)
    indicators_df = compute_indicators(df, config.indicators)

    written = duplicate = 0
    with session_factory() as session:
        for feature_name in indicators_df.columns:
            series = indicators_df[feature_name]
            for timestamp, value in series.items():
                if pd.isna(value):
                    continue  # never store a fabricated "unknown" as a real value
                inserted = insert_feature(
                    session,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=timestamp.to_pydatetime(),
                    feature_name=feature_name,
                    feature_value=float(value),
                    feature_version=FEATURE_VERSION,
                )
                if inserted:
                    written += 1
                else:
                    duplicate += 1

    return {"written": written, "duplicate": duplicate}
