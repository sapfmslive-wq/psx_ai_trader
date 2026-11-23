"""Feature engine (Phase 12): aligns a stock's candles with KSE-100/KMI-30
candles, computes market-regime and stock features, and stores them as
`Feature` rows.

Alignment is exact-timestamp only (`reindex`, no forward-fill): if an index
has no candle at a timestamp where the stock does, every index-derived
value at that timestamp is NaN — an honest gap, never a fabricated
carry-forward. NaN values are never stored.

`FEATURE_VERSION` is separate from the indicators' version: the two feature
groups evolve independently, and a formula change in one must not
invalidate stored values from the other.
"""

from __future__ import annotations

import datetime as dt

import pandas as pd

from src.features import market_regime, stock_features
from src.indicators import momentum
from src.indicators.engine import candles_to_dataframe

FEATURE_VERSION = "features_v1"


def compute_index_regime_features(index_df: pd.DataFrame, config) -> pd.DataFrame:  # noqa: ANN001 — AppConfig
    """Market-regime features for one index's candle DataFrame."""
    if index_df.empty:
        return pd.DataFrame(index=index_df.index)

    close = index_df["close"]
    columns: dict[str, pd.Series] = {}
    columns.update(market_regime.index_returns(close, config.features.return_horizons))
    columns["volatility"] = market_regime.index_volatility(close, config.features.volatility_period)
    columns["trend"] = market_regime.index_trend(
        close, config.features.trend_fast_sma_period, config.features.trend_slow_sma_period
    )
    for period in config.indicators.sma_periods:
        columns[f"position_vs_sma_{period}"] = market_regime.position_vs_sma(close, period)
    return pd.DataFrame(columns, index=index_df.index)


def compute_stock_features(
    stock_df: pd.DataFrame,
    index_dfs: dict[str, pd.DataFrame],
    config,  # noqa: ANN001 — AppConfig
) -> pd.DataFrame:
    """Stock features, including relative performance vs each index in
    `index_dfs` (keyed by index symbol, e.g. {"KSE100": df, "KMI30": df})."""
    if stock_df.empty:
        return pd.DataFrame(index=stock_df.index)

    close, high, low = stock_df["close"], stock_df["high"], stock_df["low"]
    open_, volume = stock_df["open"], stock_df["volume"]
    horizons = config.features.return_horizons
    columns: dict[str, pd.Series] = {}

    stock_returns: dict[int, pd.Series] = {}
    for h in horizons:
        stock_returns[h] = momentum.roc(close, h)
        columns[f"return_{h}bar"] = stock_returns[h]
        columns[f"log_return_{h}bar"] = stock_features.log_return(close, h)

    columns["volume_change"] = stock_features.volume_change(volume)
    columns["volume_acceleration"] = stock_features.volume_acceleration(volume)

    for period in config.indicators.sma_periods:
        columns[f"ma_distance_{period}"] = stock_features.ma_distance(close, period)

    period = config.features.high_low_breakout_period
    columns[f"high_low_breakout_{period}"] = stock_features.high_low_breakout(high, low, close, period)
    columns["intraday_range_pct"] = stock_features.intraday_range(high, low, close)
    columns["gap_pct"] = stock_features.gap(open_, close.shift(1))
    columns["relative_volume"] = stock_features.relative_volume(volume, config.indicators.volume_ma_period)

    for index_symbol, index_df in index_dfs.items():
        if index_df.empty:
            continue
        aligned_index_close = index_df["close"].reindex(stock_df.index)
        for h in horizons:
            index_return = momentum.roc(aligned_index_close, h)
            columns[f"relative_return_{h}bar_vs_{index_symbol}"] = stock_features.relative_performance(
                stock_returns[h], index_return
            )

    return pd.DataFrame(columns, index=stock_df.index)


def _store(session_factory, symbol: str, timeframe: str, features_df: pd.DataFrame, prefix: str = "") -> tuple[int, int]:  # noqa: ANN001
    from src.database.repository import insert_feature

    written = duplicate = 0
    with session_factory() as session:
        for column in features_df.columns:
            for timestamp, value in features_df[column].items():
                if pd.isna(value):
                    continue
                inserted = insert_feature(
                    session,
                    symbol=symbol,
                    timeframe=timeframe,
                    timestamp=timestamp.to_pydatetime(),
                    feature_name=f"{prefix}{column}",
                    feature_value=float(value),
                    feature_version=FEATURE_VERSION,
                )
                if inserted:
                    written += 1
                else:
                    duplicate += 1
    return written, duplicate


def compute_and_store_features(
    config,  # noqa: ANN001 — AppConfig
    session_factory,  # noqa: ANN001
    symbol: str,
    timeframe: str,
    start: dt.datetime,
    end: dt.datetime,
) -> dict[str, int]:
    """Read candles for `symbol` and every configured index, compute stock +
    market-regime features, and store them.

    Index regime features are stored under each INDEX's own symbol (e.g.
    symbol="KSE100", feature_name="trend"); the stock's features — including
    its relative performance vs each index — under the stock's symbol.
    If `symbol` is itself an index, only its regime features are produced.

    Returns {"written": N, "duplicate": M}.
    """
    from src.database.repository import get_candles

    indices = list(config.symbols.indices)
    with session_factory() as session:
        index_dfs = {
            idx: candles_to_dataframe(get_candles(session, idx, timeframe, start, end)) for idx in indices
        }
        stock_df = (
            pd.DataFrame()
            if symbol in indices
            else candles_to_dataframe(get_candles(session, symbol, timeframe, start, end))
        )

    total_written = total_duplicate = 0

    if symbol in indices:
        regime_df = compute_index_regime_features(index_dfs[symbol], config)
        w, d = _store(session_factory, symbol, timeframe, regime_df)
        total_written += w
        total_duplicate += d
    else:
        for idx, idx_df in index_dfs.items():
            regime_df = compute_index_regime_features(idx_df, config)
            w, d = _store(session_factory, idx, timeframe, regime_df)
            total_written += w
            total_duplicate += d
        features_df = compute_stock_features(stock_df, index_dfs, config)
        w, d = _store(session_factory, symbol, timeframe, features_df)
        total_written += w
        total_duplicate += d

    return {"written": total_written, "duplicate": total_duplicate}
