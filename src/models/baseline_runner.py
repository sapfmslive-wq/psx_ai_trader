"""Baseline comparison (Phase 14): runs every baseline on one symbol's
candles over the same chronological test segment and reports prediction
quality side by side.

Two layers, same split as the rest of the project:
    compute_baseline_report()          - pure: candle DataFrame + config in,
                                         BaselineReport out. No database.
    evaluate_and_store_baselines()     - reads candles and records one
                                         `model_runs` row per baseline.

How the evaluation is kept honest:
  * One chronological split; the series is never shuffled.
  * The last `horizon` train bars are purged (their labels reach into the
    test period). Only the majority-class baseline looks at train at all.
  * Every baseline is scored on the SAME test bars, against the same
    direction labels (Phase 13's `direction_label`, recomputed in memory
    with the same function so no database read is needed).
  * Abstaining baselines (e.g. RSI) are scored only where they made a call,
    and their coverage is reported next to the score.
  * Each row carries the "null accuracy" — what a constant guess would have
    scored on those same bars — and a conservative verdict.
No transaction costs are applied anywhere here; see src/models/__init__.py.
"""

from __future__ import annotations

import datetime as dt
import json
import math
import platform
from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from src.features import labels
from src.indicators.engine import candles_to_dataframe
from src.models import baseline
from src.models import evaluation as ev

BASELINE_DESCRIPTIONS = {
    "buy_and_hold": "always predicts UP",
    "previous_direction": "predicts the last bar's move continues",
    "moving_average": "UP when fast SMA > slow SMA, else DOWN",
    "rsi": "oversold -> UP, overbought -> DOWN, else abstains",
    "random": "seeded fair-coin guess",
    "majority_class": "always the more common direction in TRAIN labels",
}


@dataclass
class BaselineRow:
    name: str
    description: str
    metrics: ev.ClassificationMetrics
    n_effective: int  # approx. independent outcomes (see evaluation.effective_sample_size)
    p_value: float
    verdict: str


@dataclass
class BaselineReport:
    symbol: str
    timeframe: str
    horizon: int
    n_bars: int
    train_bars: int
    test_bars: int
    test_start: pd.Timestamp
    test_end: pd.Timestamp
    test_up_rate: float
    majority_class_from_train: float | None
    buy_hold_gross_return_pct: float  # over the test segment, BEFORE any costs
    random_seed: int
    rows: list[BaselineRow] = field(default_factory=list)


def compute_baseline_report(
    df: pd.DataFrame, config, horizon: int, symbol: str = "", timeframe: str = ""  # noqa: ANN001 — AppConfig
) -> BaselineReport:
    """Score every baseline on the chronological test segment of `df`."""
    cfg = config.baselines
    minimum = cfg.ma_slow_period + horizon + 10
    if len(df) < minimum:
        raise ValueError(
            f"Need at least {minimum} candles to evaluate baselines at horizon {horizon} "
            f"(got {len(df)}). Build more candles first."
        )

    close = df["close"]
    future_ret = labels.future_return(close, horizon)
    y = labels.direction_label(future_ret)

    train_end, test_start = ev.chronological_split(len(df), cfg.train_fraction, horizon)
    train_labels = y.iloc[:train_end]
    test_index = df.index[test_start:]

    majority_pred, majority_class = baseline.majority_class_predictions(train_labels, df.index)
    predictions = {
        "buy_and_hold": baseline.buy_and_hold_predictions(df.index),
        "previous_direction": baseline.previous_direction_predictions(close),
        "moving_average": baseline.moving_average_predictions(close, cfg.ma_fast_period, cfg.ma_slow_period),
        "rsi": baseline.rsi_predictions(close, config.indicators.rsi_period, cfg.rsi_oversold, cfg.rsi_overbought),
        "random": baseline.random_predictions(df.index, cfg.random_seed),
        "majority_class": majority_pred,
    }

    y_test = y.iloc[test_start:]
    known_test = y_test.dropna()
    test_up_rate = float(known_test.mean()) if len(known_test) else float("nan")

    report = BaselineReport(
        symbol=symbol,
        timeframe=timeframe,
        horizon=horizon,
        n_bars=len(df),
        train_bars=train_end,
        test_bars=len(test_index),
        test_start=test_index[0],
        test_end=test_index[-1],
        test_up_rate=test_up_rate,
        majority_class_from_train=majority_class,
        buy_hold_gross_return_pct=float((close.iloc[-1] / close.iloc[test_start] - 1) * 100),
        random_seed=cfg.random_seed,
    )
    for name, pred in predictions.items():
        metrics = ev.classification_metrics(y_test, pred.iloc[test_start:])
        n_effective = ev.effective_sample_size(metrics.n_predicted, horizon)
        p_value = ev.approx_p_value_vs_chance(metrics.accuracy, n_effective)
        report.rows.append(
            BaselineRow(
                name, BASELINE_DESCRIPTIONS[name], metrics, n_effective, p_value,
                ev.verdict(metrics, p_value, n_effective),
            )
        )
    return report


def _clean(value):
    """JSON has no NaN: store undefined numbers as null."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def evaluate_and_store_baselines(
    config,  # noqa: ANN001 — AppConfig
    session_factory,  # noqa: ANN001
    symbol: str,
    timeframe: str,
    start: dt.datetime,
    end: dt.datetime,
    horizon: int,
) -> BaselineReport:
    """Read candles in [start, end), build the report, and record one
    `model_runs` row per baseline (experiment tracking, Section 100) with
    seed, Python and package versions (Section 59)."""
    from src.database.repository import get_candles, insert_model_run

    with session_factory() as session:
        candles = get_candles(session, symbol, timeframe, start, end)
    df = candles_to_dataframe(candles)
    report = compute_baseline_report(df, config, horizon, symbol=symbol, timeframe=timeframe)

    stamp = dt.datetime.now(dt.timezone.utc)
    versions = {"pandas": pd.__version__, "numpy": np.__version__}
    train_start = df.index[0].to_pydatetime()
    train_last = df.index[max(report.train_bars - 1, 0)].to_pydatetime()
    with session_factory() as session:
        for row in report.rows:
            insert_model_run(
                session,
                model_id=f"{symbol}_{timeframe}_{horizon}BAR_{row.name}_{stamp:%Y%m%d%H%M%S}",
                symbol=symbol,
                timeframe=timeframe,
                model_type=f"baseline:{row.name}",
                training_start=train_start,
                training_end=train_last,
                test_start=report.test_start.to_pydatetime(),
                test_end=report.test_end.to_pydatetime(),
                hyperparameters_json=json.dumps(
                    {"horizon_bars": horizon, "description": row.description,
                     "train_fraction": config.baselines.train_fraction}
                ),
                metrics_json=json.dumps(
                    {**{k: _clean(v) for k, v in asdict(row.metrics).items()},
                     "n_effective": row.n_effective,
                     "p_value_vs_chance": _clean(row.p_value), "verdict": row.verdict}
                ),
                feature_list_json="[]",  # baselines read raw price only, no engineered features
                dataset_version=f"candles:{start.date()}..{end.date()}",
                random_seed=report.random_seed,
                python_version=platform.python_version(),
                package_versions_json=json.dumps(versions),
                status="BASELINE",
            )
    return report
