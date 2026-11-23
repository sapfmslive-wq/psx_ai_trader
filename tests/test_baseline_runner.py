"""Tests for src.models.baseline_runner (Phase 14), pure report logic plus
the model_runs recording via a fake repository (no SQLAlchemy needed)."""

from __future__ import annotations

import datetime as dt
import json
import math
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from src.models.baseline_runner import compute_baseline_report, evaluate_and_store_baselines

HORIZON = 5


def _config(seed=42, train_fraction=0.7):
    return SimpleNamespace(
        baselines=SimpleNamespace(
            ma_fast_period=5, ma_slow_period=20, rsi_oversold=30.0, rsi_overbought=70.0,
            train_fraction=train_fraction, random_seed=seed,
        ),
        indicators=SimpleNamespace(rsi_period=14),
    )


def _df(close):
    close = pd.Series(close, dtype=float)
    idx = pd.date_range("2026-09-21 09:32", periods=len(close), freq="1min", tz="UTC")
    return pd.DataFrame(
        {"open": close.values, "high": close.values + 0.1, "low": close.values - 0.1,
         "close": close.values, "volume": 100.0},
        index=idx,
    )


def _random_walk(n=600, seed=1):
    return 100 + np.cumsum(np.random.default_rng(seed).normal(0, 0.5, n))


def _row(report, name):
    return next(r for r in report.rows if r.name == name)


def test_report_has_all_six_baselines_on_one_shared_test_segment():
    report = compute_baseline_report(_df(_random_walk()), _config(), HORIZON, "THCCL", "1m")
    assert [r.name for r in report.rows] == [
        "buy_and_hold", "previous_direction", "moving_average", "rsi", "random", "majority_class",
    ]
    assert report.train_bars + HORIZON + report.test_bars == report.n_bars  # purge gap accounted for
    # the three never-abstaining baselines are scored on exactly the same bars
    counts = {_row(report, n).metrics.n_predicted for n in ("buy_and_hold", "random", "majority_class")}
    assert len(counts) == 1


def test_buy_and_hold_accuracy_equals_the_test_up_rate():
    report = compute_baseline_report(_df(_random_walk()), _config(), HORIZON)
    assert math.isclose(_row(report, "buy_and_hold").metrics.accuracy, report.test_up_rate)


def test_random_baseline_is_reproducible_and_near_chance():
    df = _df(_random_walk(1500, seed=3))
    a = compute_baseline_report(df, _config(seed=9), HORIZON)
    b = compute_baseline_report(df, _config(seed=9), HORIZON)
    assert _row(a, "random").metrics.accuracy == _row(b, "random").metrics.accuracy
    assert 0.40 < _row(a, "random").metrics.accuracy < 0.60


def test_rsi_baseline_abstains_so_its_coverage_is_below_one():
    report = compute_baseline_report(_df(_random_walk()), _config(), HORIZON)
    assert _row(report, "rsi").metrics.coverage < 1.0


def test_always_up_on_a_pure_uptrend_is_not_credited_as_skill():
    report = compute_baseline_report(_df(np.arange(100.0, 1100.0)), _config(), HORIZON)
    row = _row(report, "buy_and_hold")
    assert row.metrics.accuracy == 1.0  # every bar went up
    # ...but a constant guess scores exactly that too, so the verdict says so
    assert "NO BETTER THAN ALWAYS GUESSING" in row.verdict


def test_majority_class_learns_from_train_only_not_from_the_test_period():
    # first 70% falls, last 30% rises: a predictor that peeked at the test
    # period would call UP; one restricted to train must call DOWN.
    close = list(np.linspace(200, 100, 420)) + list(np.linspace(100, 250, 180))
    report = compute_baseline_report(_df(close), _config(), HORIZON)
    assert report.majority_class_from_train == 0.0
    row = _row(report, "majority_class")
    assert row.metrics.accuracy < 0.2  # confidently wrong on a reversed test period


def test_mutating_test_period_prices_cannot_change_the_majority_class():
    base = _random_walk(600, seed=11)
    report_a = compute_baseline_report(_df(base), _config(), HORIZON)
    test_start = int(600 * 0.7)
    mutated = base.copy()
    mutated[test_start:] += 5000.0  # only bars from the split onward
    report_b = compute_baseline_report(_df(mutated), _config(), HORIZON)
    assert report_a.majority_class_from_train == report_b.majority_class_from_train


def test_rows_carry_the_effective_sample_size():
    report = compute_baseline_report(_df(_random_walk(900)), _config(), HORIZON)
    for row in report.rows:
        assert row.n_effective == row.metrics.n_predicted // HORIZON


def test_known_false_positive_on_a_random_walk_is_no_longer_called_significant():
    # Regression. This exact series (a pure random walk: no skill exists to
    # find) used to get p=0.0000 for the moving-average rule from the naive
    # test, and a verdict that read "ABOVE 50%" for an accuracy of 0.39.
    close = 100 + np.cumsum(np.random.default_rng(21).normal(0, 0.3, 1200))
    cfg = _config()
    cfg.baselines.ma_fast_period, cfg.baselines.ma_slow_period = 10, 30
    report = compute_baseline_report(_df(close), cfg, HORIZON)
    row = _row(report, "moving_average")
    assert "NOT DISTINGUISHABLE" in row.verdict
    assert "ABOVE 50%" not in row.verdict


def test_false_positive_rate_on_pure_noise_stays_near_the_nominal_five_percent():
    # No structure exists in a random walk, so any "significant" verdict is a
    # false alarm. Measured with the naive test this was 12-35%; with
    # effective-sample-size thinning it is <= ~3%. Guard well above that
    # (10%) so seed choice cannot make this flaky, yet far below the old rates.
    significant = {}
    trials = 80
    for t in range(trials):
        close = 100 + np.cumsum(np.random.default_rng(5000 + t).normal(0, 0.3, 1200))
        report = compute_baseline_report(_df(close), _config(), HORIZON)
        for row in report.rows:
            flagged = row.verdict.startswith(("SIGNIFICANTLY", "BEATS", "ABOVE 50%"))
            significant[row.name] = significant.get(row.name, 0) + int(flagged)
    for name, count in significant.items():
        assert count / trials <= 0.10, f"{name} falsely flagged on {count}/{trials} noise series"


def test_too_few_candles_raises_a_clear_error():
    with pytest.raises(ValueError):
        compute_baseline_report(_df(_random_walk(20)), _config(), HORIZON)


def test_buy_hold_gross_return_is_reported_over_the_test_segment():
    close = np.linspace(100, 200, 300)
    report = compute_baseline_report(_df(close), _config(), HORIZON)
    test_start = int(300 * 0.7)
    expected = (close[-1] / close[test_start] - 1) * 100
    assert math.isclose(report.buy_hold_gross_return_pct, expected)


# ---------------- recording to model_runs via a fake repository ----------------

@dataclass
class _Candle:
    open_time: dt.datetime
    open: float
    high: float
    low: float
    close: float
    volume: float


class _Session:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _candles(n=300):
    base = dt.datetime(2026, 9, 21, 9, 32, tzinfo=dt.timezone.utc)
    closes = _random_walk(n, seed=2)
    return [_Candle(base + dt.timedelta(minutes=i), c, c + 0.1, c - 0.1, c, 100.0) for i, c in enumerate(closes)]


def _install(monkeypatch, candles):
    runs = []
    fake = types.ModuleType("src.database.repository")
    fake.get_candles = lambda session, symbol, timeframe, start, end: candles

    def insert_model_run(session, **kw):
        if any(r["model_id"] == kw["model_id"] for r in runs):
            return False
        runs.append(kw)
        return True

    fake.insert_model_run = insert_model_run
    monkeypatch.setitem(sys.modules, "src.database.repository", fake)
    return runs


def _store(config, candles):
    return evaluate_and_store_baselines(
        config, lambda: _Session(), "THCCL", "1m",
        candles[0].open_time, candles[-1].open_time + dt.timedelta(minutes=1), HORIZON,
    )


def test_one_model_run_is_recorded_per_baseline_with_provenance(monkeypatch):
    candles = _candles()
    runs = _install(monkeypatch, candles)
    _store(_config(seed=123), candles)

    assert len(runs) == 6
    assert {r["model_type"] for r in runs} == {
        f"baseline:{n}" for n in
        ("buy_and_hold", "previous_direction", "moving_average", "rsi", "random", "majority_class")
    }
    assert len({r["model_id"] for r in runs}) == 6  # unique ids
    for r in runs:
        assert r["random_seed"] == 123
        assert r["status"] == "BASELINE"
        assert r["feature_list_json"] == "[]"
        assert r["python_version"]
        assert "pandas" in json.loads(r["package_versions_json"])
        assert r["training_end"] < r["test_start"]  # chronological, purged


def test_recorded_metrics_are_valid_json_with_no_nan_literals(monkeypatch):
    candles = _candles()
    runs = _install(monkeypatch, candles)
    _store(_config(), candles)
    for r in runs:
        text = r["metrics_json"]
        assert "NaN" not in text
        parsed = json.loads(text)  # would raise on invalid JSON
        assert "verdict" in parsed and "accuracy" in parsed
