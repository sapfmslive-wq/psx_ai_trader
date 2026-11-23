"""Training dataset assembly and chronological splitting (Phase 15).

Everything here is built from the same pure functions the stored features
and labels use (indicators, stock features, index-regime features, direction
labels), computed in memory from candles — so a model never depends on
which Feature rows happen to have been written to the database, and the
dataset is reproducible from candles + config alone.

**Scale-free features only.** Raw price levels (SMA/EMA values, Bollinger
bands, VWAP, OBV, raw ATR/MACD) are excluded or re-expressed as a percentage
of price. A model fed price levels can learn "where price is" instead of
"what price is doing" — which looks brilliant in-sample and is useless
out-of-sample as the level drifts.

Leakage rules enforced here:
  * X at bar t uses only bars at or before t (inherited from the feature
    functions; tests/test_ml_dataset.py re-checks it on the assembled matrix).
  * y at bar t is the Phase 13 direction label (uses bars after t) and lives
    in a separate Series that never enters X.
  * Segments are chronological and PURGED: a training row whose label
    reaches `horizon` bars ahead into the next segment is dropped, so no
    label ever "sees" prices that belong to a later segment.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from src.features import labels
from src.features.engine import compute_index_regime_features, compute_stock_features
from src.indicators.engine import compute_indicators

DATASET_VERSION = "ml_features_v1"


@dataclass
class Dataset:
    X: pd.DataFrame  # usable rows only, original bar order
    y: pd.Series  # 1.0 UP / 0.0 DOWN, aligned to X
    future_return: pd.Series  # % return over the horizon (for reporting, never a feature)
    positions: np.ndarray  # each row's position in the ORIGINAL bar series
    feature_names: list[str]
    n_input_bars: int
    dropped_all_nan_columns: list[str] = field(default_factory=list)
    n_dropped_feature_nan_rows: int = 0
    n_dropped_unlabeled_rows: int = 0


def build_dataset(
    stock_df: pd.DataFrame,
    index_dfs: dict[str, pd.DataFrame],
    config,  # noqa: ANN001 — AppConfig
    horizon: int,
) -> Dataset:
    """Assemble the scale-free feature matrix and the direction label.

    Raises ValueError on any non-positive close price: every percentage-based
    feature and label divides by a price, so a zero/negative price would
    silently flip signs (found the hard way with a synthetic series that
    drifted below zero). Real candles cannot contain one — the quality engine
    excludes non-positive ticks — so this is a loud stop on impossible input.
    """
    for name, frame_ in (("stock", stock_df), *index_dfs.items()):
        if len(frame_) and (frame_["close"] <= 0).any():
            raise ValueError(
                f"{name} candles contain non-positive close prices; percentage returns would "
                f"invert their sign. Check the data (quality engine should have excluded these)."
            )
    close = stock_df["close"]
    ic = config.indicators
    indicators = compute_indicators(stock_df, ic)
    stock_feats = compute_stock_features(stock_df, index_dfs, config)

    cols: dict[str, pd.Series] = {}
    for name in (
        f"rsi_{ic.rsi_period}", f"roc_{ic.roc_period}", f"adx_{ic.adx_period}",
        f"plus_di_{ic.adx_period}", f"minus_di_{ic.adx_period}",
        f"stoch_k_{ic.stochastic_k_period}", f"stoch_d_{ic.stochastic_d_period}",
        f"rolling_volatility_{ic.bollinger_period}", "vwap_distance_pct",
    ):
        cols[name] = indicators[name]

    cols["atr_pct"] = indicators[f"atr_{ic.atr_period}"] / close * 100
    cols["macd_pct"] = indicators["macd_line"] / close * 100
    cols["macd_signal_pct"] = indicators["macd_signal"] / close * 100
    cols["macd_histogram_pct"] = indicators["macd_histogram"] / close * 100

    bb_lower = indicators[f"bb_lower_{ic.bollinger_period}"]
    bb_upper = indicators[f"bb_upper_{ic.bollinger_period}"]
    bb_middle = indicators[f"bb_middle_{ic.bollinger_period}"]
    cols["bb_width_pct"] = (bb_upper - bb_lower) / bb_middle * 100
    cols["bb_position"] = (close - bb_lower) / (bb_upper - bb_lower)

    for period in ic.ema_periods:
        ema = indicators[f"ema_{period}"]
        cols[f"ema_distance_{period}"] = (close - ema) / ema * 100

    for name in stock_feats.columns:  # already scale-free by construction
        cols[name] = stock_feats[name]

    for symbol, index_df in index_dfs.items():
        if index_df.empty:
            continue
        regime = compute_index_regime_features(index_df, config).reindex(stock_df.index)
        for name in regime.columns:
            cols[f"{symbol}_{name}"] = regime[name]

    X = pd.DataFrame(cols, index=stock_df.index).replace([np.inf, -np.inf], np.nan)

    all_nan = [c for c in X.columns if X[c].isna().all()]
    X = X.drop(columns=all_nan)

    future_return = labels.future_return(close, horizon)
    y = labels.direction_label(future_return)

    feature_ok = X.notna().all(axis=1)
    label_ok = y.notna()
    usable = feature_ok & label_ok

    return Dataset(
        X=X[usable],
        y=y[usable],
        future_return=future_return[usable],
        positions=np.flatnonzero(usable.to_numpy()),
        feature_names=list(X.columns),
        n_input_bars=len(stock_df),
        dropped_all_nan_columns=all_nan,
        n_dropped_feature_nan_rows=int((~feature_ok).sum()),
        n_dropped_unlabeled_rows=int((feature_ok & ~label_ok).sum()),
    )


@dataclass
class Split:
    train: np.ndarray  # row indices into Dataset.X
    validation: np.ndarray
    test: np.ndarray
    purged_from_train: int
    purged_from_validation: int


def split_chronological(
    positions: np.ndarray, train_fraction: float, validation_fraction: float, horizon: int
) -> Split:
    """Chronological train / validation / test split with a purge.

    Boundaries are set by row count, then rows are dropped from the END of
    train (and of validation) whose label horizon reaches into the next
    segment: a row at original bar position `p` has a label built from
    prices up to `p + horizon`, so it is kept only if `p + horizon` is
    strictly before the first bar position of the following segment. This is
    done on original bar positions, not row counts, so rows removed for NaN
    cannot make the purge too small. Raises if train or validation ends up
    empty rather than returning an unusable split.
    """
    n = len(positions)
    a = int(n * train_fraction)
    b = int(n * (train_fraction + validation_fraction))
    if not (0 < a < b < n):
        raise ValueError(f"Cannot split {n} rows with fractions {train_fraction}/{validation_fraction}.")

    rows = np.arange(n)
    train_all, validation_all, test = rows[:a], rows[a:b], rows[b:]

    train = train_all[positions[train_all] + horizon < positions[validation_all[0]]]
    validation = validation_all[positions[validation_all] + horizon < positions[test[0]]]
    if len(train) == 0 or len(validation) == 0:
        raise ValueError(
            f"A segment is empty after purging {horizon} bars at each boundary "
            f"(train={len(train)}, validation={len(validation)}, test={len(test)}). "
            f"Use more data or a shorter horizon."
        )
    return Split(
        train=train,
        validation=validation,
        test=test,
        purged_from_train=len(train_all) - len(train),
        purged_from_validation=len(validation_all) - len(validation),
    )
