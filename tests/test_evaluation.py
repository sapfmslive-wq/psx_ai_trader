"""Tests for src.models.evaluation (Phase 14)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from src.models import evaluation as ev


def _m(true, pred):
    return ev.classification_metrics(pd.Series(true, dtype=float), pd.Series(pred, dtype=float))


def test_confusion_counts_and_metrics_match_hand_calculation():
    #        truth: U U U D D D
    #         pred: U U D D D U     tp=2 fn=1 tn=2 fp=1
    m = _m([1, 1, 1, 0, 0, 0], [1, 1, 0, 0, 0, 1])
    assert (m.tp, m.fn, m.tn, m.fp) == (2, 1, 2, 1)
    assert math.isclose(m.accuracy, 4 / 6)
    assert math.isclose(m.recall_up, 2 / 3)
    assert math.isclose(m.precision_up, 2 / 3)
    assert math.isclose(m.f1_up, 2 / 3)
    assert math.isclose(m.balanced_accuracy, (2 / 3 + 2 / 3) / 2)
    assert m.coverage == 1.0


def test_abstentions_reduce_coverage_but_not_accuracy():
    m = _m([1, 1, 0, 0], [1, np.nan, 0, np.nan])
    assert m.n_labeled == 4 and m.n_predicted == 2
    assert math.isclose(m.coverage, 0.5)
    assert m.accuracy == 1.0  # scored only where it actually called


def test_unknown_labels_are_dropped():
    m = _m([1, np.nan, 0], [1, 1, 0])
    assert m.n_labeled == 2 and m.accuracy == 1.0


def test_always_up_on_imbalanced_labels_has_high_accuracy_but_balanced_accuracy_of_half():
    truth = [1] * 8 + [0] * 2
    m = _m(truth, [1] * 10)
    assert math.isclose(m.accuracy, 0.8)  # looks good...
    assert math.isclose(m.balanced_accuracy, 0.5)  # ...but it never finds a DOWN
    assert math.isclose(m.null_accuracy, 0.8)  # exactly what a constant guess scores


def test_undefined_ratios_are_nan_not_zero():
    m = _m([1, 1], [1, 1])  # no DOWN bars scored: recall_down undefined
    assert m.accuracy == 1.0
    assert math.isclose(m.balanced_accuracy, 1.0)  # only the defined recall is used
    empty = _m([1, 0], [np.nan, np.nan])
    assert math.isnan(empty.accuracy) and empty.n_predicted == 0 and empty.coverage == 0.0


def test_p_value_is_one_at_chance_and_small_for_a_big_edge():
    assert math.isclose(ev.approx_p_value_vs_chance(0.5, 1000), 1.0)
    assert ev.approx_p_value_vs_chance(0.6, 1000) < 0.001
    assert ev.approx_p_value_vs_chance(0.6, 20) > 0.05  # same accuracy, tiny sample
    assert math.isnan(ev.approx_p_value_vs_chance(float("nan"), 100))


def _metrics(acc, n, null):
    return ev.ClassificationMetrics(n, n, 1.0, acc, acc, acc, acc, acc, null, 0, 0, 0, 0)


def test_verdict_branches():
    assert "INSUFFICIENT" in ev.verdict(_metrics(0.9, 10, 0.5), 0.0001, n_effective=10)
    assert "NOT DISTINGUISHABLE" in ev.verdict(_metrics(0.51, 500, 0.5), 0.6, n_effective=100)
    assert "NO BETTER THAN ALWAYS GUESSING" in ev.verdict(_metrics(0.7, 500, 0.7), 0.0001, n_effective=100)
    assert "BEATS A CONSTANT GUESS" in ev.verdict(_metrics(0.75, 500, 0.6), 0.0001, n_effective=100)


def test_significantly_below_chance_is_called_worse_not_above_50():
    # regression: accuracy 0.39 with a tiny p-value used to read "ABOVE 50%..."
    text = ev.verdict(_metrics(0.39, 500, 0.55), 0.001, n_effective=100)
    assert "WORSE THAN CHANCE" in text
    assert "ABOVE 50%" not in text


def test_verdict_judges_independent_outcomes_not_raw_bars():
    # plenty of raw bars, but too few independent outcomes -> no verdict
    assert "INSUFFICIENT" in ev.verdict(_metrics(0.9, 400, 0.5), 0.0001, n_effective=12)


def test_effective_sample_size_thins_by_the_horizon():
    assert ev.effective_sample_size(300, 5) == 60
    assert ev.effective_sample_size(299, 5) == 59
    assert ev.effective_sample_size(100, 1) == 100
    assert ev.effective_sample_size(0, 5) == 0
    assert ev.effective_sample_size(10, 0) == 10  # defensive: horizon < 1 treated as 1


def test_chronological_split_is_ordered_and_purges_the_horizon():
    train_end, test_start = ev.chronological_split(100, 0.7, horizon=5)
    assert test_start == 70
    assert train_end == 65  # last 5 train bars purged: their labels peek past bar 70
    assert train_end < test_start


def test_chronological_split_never_goes_negative():
    train_end, test_start = ev.chronological_split(10, 0.1, horizon=30)
    assert train_end == 0 and test_start == 1
