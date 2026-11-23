"""Tests for src.models.ml_metrics and src.models.ml_models (Phase 15), run
against real scikit-learn."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.models import ml_metrics as mm
from src.models import ml_models as mdl


# ------------------------------ metrics ------------------------------

def test_brier_and_log_loss_match_hand_calculation():
    y = [1, 0, 1, 0]
    p = [0.9, 0.2, 0.6, 0.4]
    m = mm.probability_metrics(y, p)
    assert math.isclose(m.brier, np.mean([(0.9 - 1) ** 2, 0.2**2, (0.6 - 1) ** 2, 0.4**2]))
    expected_ll = -np.mean([math.log(0.9), math.log(0.8), math.log(0.6), math.log(0.6)])
    assert math.isclose(m.log_loss, expected_ll, rel_tol=1e-6)


def test_threshold_decisions_feed_the_classification_metrics():
    m = mm.probability_metrics([1, 1, 0, 0], [0.9, 0.4, 0.6, 0.1])  # threshold 0.5
    assert (m.classification.tp, m.classification.fn, m.classification.fp, m.classification.tn) == (1, 1, 1, 1)
    assert math.isclose(m.classification.accuracy, 0.5)


def test_roc_auc_is_one_for_perfect_ranking_and_half_for_constant_scores():
    assert mm.probability_metrics([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]).roc_auc == 1.0
    assert mm.probability_metrics([0, 1, 0, 1], [0.5, 0.5, 0.5, 0.5]).roc_auc == 0.5


def test_single_class_truth_makes_auc_nan_not_a_crash():
    m = mm.probability_metrics([1, 1, 1], [0.6, 0.7, 0.8])
    assert math.isnan(m.roc_auc)
    assert not math.isnan(m.brier)


def test_calibration_table_counts_sum_to_n_and_observed_rate_is_correct():
    rng = np.random.default_rng(0)
    p = rng.random(500)
    y = (rng.random(500) < p).astype(float)  # perfectly calibrated by construction
    table = mm.calibration_table(y, p, bins=10)
    assert sum(row["n"] for row in table) == 500
    assert all(0 <= row["observed_up_rate"] <= 1 for row in table)
    assert mm.expected_calibration_error(table) < 0.08  # small for a calibrated source


def test_overconfident_probabilities_have_a_large_expected_calibration_error():
    rng = np.random.default_rng(1)
    y = (rng.random(800) < 0.5).astype(float)  # pure coin flips
    overconfident = np.where(rng.random(800) < 0.5, 0.95, 0.05)  # claims near-certainty
    assert mm.probability_metrics(y, overconfident).ece > 0.35


# ------------------------------ models ------------------------------

def _separable(n=300, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    y = (X[:, 0] + 0.3 * rng.normal(size=n) > 0).astype(float)  # feature 0 drives the label
    return X, y


def test_every_model_fits_and_returns_valid_probabilities():
    X, y = _separable()
    for name in mdl.MODEL_ABBREVIATIONS:
        result = mdl.fit_with_validation_selection(name, X[:200], y[:200], X[200:], y[200:], seed=3)
        p = mdl.predict_up_probability(result.estimator, X[200:])
        assert p.shape == (100,) and ((p >= 0) & (p <= 1)).all(), name
        assert result.params in [g["params"] for g in result.grid], name
        assert result.validation_log_loss == min(g["validation_log_loss"] for g in result.grid), name
        # the planted signal is learnable by every model
        acc = ((p >= 0.5) == (y[200:] == 1)).mean()
        assert acc > 0.75, f"{name} failed to learn an obvious signal (acc={acc:.2f})"


def test_selection_uses_validation_data_only_not_test():
    X, y = _separable(400, seed=5)
    a = mdl.fit_with_validation_selection("logistic", X[:200], y[:200], X[200:300], y[200:300], seed=1)
    # scrambling the held-out TEST block must not change anything chosen
    X2, y2 = X.copy(), y.copy()
    X2[300:], y2[300:] = X2[300:] * -7, 1 - y2[300:]
    b = mdl.fit_with_validation_selection("logistic", X2[:200], y2[:200], X2[200:300], y2[200:300], seed=1)
    assert a.params == b.params and a.validation_log_loss == b.validation_log_loss


def test_same_seed_gives_identical_models_and_different_seed_can_differ():
    X, y = _separable()
    a = mdl.fit_with_validation_selection("random_forest", X[:200], y[:200], X[200:], y[200:], seed=7)
    b = mdl.fit_with_validation_selection("random_forest", X[:200], y[:200], X[200:], y[200:], seed=7)
    np.testing.assert_array_equal(mdl.predict_up_probability(a.estimator, X[200:]),
                                  mdl.predict_up_probability(b.estimator, X[200:]))


def test_single_class_training_labels_raise_instead_of_fitting_garbage():
    X = np.random.default_rng(0).normal(size=(50, 3))
    with pytest.raises(ValueError):
        mdl.fit_with_validation_selection("logistic", X, np.ones(50), X, np.ones(50), seed=0)


def test_unknown_model_name_is_rejected():
    with pytest.raises(ValueError):
        mdl.candidate_params("deep_neural_net")


def test_probability_is_looked_up_by_class_label_not_column_position():
    X, y = _separable()
    flipped = 1.0 - y  # classes_ still [0.0, 1.0]; assert lookup uses the 1.0 column
    est = mdl.make_estimator("logistic", {"C": 1.0}, 0).fit(X, flipped)
    p = mdl.predict_up_probability(est, X)
    assert np.allclose(p, est.predict_proba(X)[:, list(est.classes_).index(1.0)])


def test_platt_calibration_fixes_overconfidence_and_keeps_the_ranking():
    rng = np.random.default_rng(2)
    true_p = rng.uniform(0.35, 0.65, 2000)
    y = (rng.random(2000) < true_p).astype(float)
    raw = np.clip(0.5 + (true_p - 0.5) * 6, 0.01, 0.99)  # overconfident but well-ordered
    cal = mdl.PlattCalibrator().fit(raw[:1000], y[:1000])
    assert not cal.is_identity
    out = cal.transform(raw[1000:])
    before = mm.probability_metrics(y[1000:], raw[1000:])
    after = mm.probability_metrics(y[1000:], out)
    assert after.ece < before.ece and after.brier < before.brier
    assert math.isclose(before.roc_auc, after.roc_auc, abs_tol=1e-9)  # ranking untouched


def test_platt_falls_back_to_identity_on_tiny_or_single_class_validation():
    p = np.array([0.2, 0.8, 0.6])
    assert mdl.PlattCalibrator().fit(p, [1, 0, 1]).is_identity  # too few rows
    cal = mdl.PlattCalibrator().fit(np.linspace(0.1, 0.9, 60), np.ones(60))
    assert cal.is_identity
    np.testing.assert_array_equal(cal.transform(p), p)
