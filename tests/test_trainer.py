"""Tests for src.models.trainer (Phase 15) against real scikit-learn.

The two series that matter: a pure random walk (no signal exists, so any
"skill" found is a false positive) and an AR(1)-returns series (a genuine,
learnable signal the pipeline must be able to detect).
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

from src.models import ml_metrics as mm
from src.models import ml_models as mdl
from src.models.dataset import build_dataset, split_chronological
from src.models.trainer import clears_baselines, train_and_evaluate
from tests._ml_helpers import ar_walk, frame, make_config, random_walk, regime_walk

warnings.filterwarnings("ignore")
H = 5
N = 1500


def _indices(n, seed):
    return {
        "KSE100": frame(random_walk(n, seed + 10, 45000.0, 30.0), volume_seed=3),
        "KMI30": frame(random_walk(n, seed + 20, 70000.0, 40.0), volume_seed=4),
    }


def _run(series, models, seed=7, n=N, **kw):
    return train_and_evaluate(frame(series), _indices(n, seed), make_config(), H, models, "THCCL", "1m", **kw)


def test_segments_are_chronological_purged_and_the_report_is_well_formed():
    report, fitted = _run(random_walk(N, 7), ["logistic"])
    assert report.train_period[1] < report.validation_period[0]
    assert report.validation_period[1] < report.test_period[0]
    assert report.n_train + report.n_validation + report.n_test + report.purged_from_train \
        + report.purged_from_validation == report.n_rows
    assert report.purged_from_train == H and report.purged_from_validation == H
    assert report.models_compared == 1 and report.alpha == 0.05
    assert set(fitted) == {"logistic"}
    assert {b.name for b in report.baselines} == {
        "buy_and_hold", "previous_direction", "moving_average", "rsi", "random", "majority_class"}


def test_bonferroni_alpha_tightens_with_the_number_of_models_compared():
    report, _ = _run(random_walk(N, 7), ["logistic", "hist_gradient_boosting"])
    assert report.models_compared == 2 and report.alpha == 0.025


def test_pure_noise_is_not_reported_as_skill_by_any_model():
    report, _ = _run(random_walk(N, 7), ["logistic", "random_forest", "gradient_boosting", "hist_gradient_boosting"])
    for r in report.results:
        assert not r.clears_baselines, r.name
        assert not r.verdict.startswith("BEATS"), (r.name, r.verdict)
        assert 0.40 < r.test.roc_auc < 0.60, (r.name, r.test.roc_auc)  # ranking is a coin flip


def test_false_positive_rate_on_noise_stays_low_across_many_series():
    # Measured: 0/40 for logistic and HGB at alpha=0.025. Guard far above that
    # (<=2 of 15) so seed choice can't make it flaky, but far below a broken test.
    clears = 0
    for t in range(15):
        stock = frame(random_walk(900, 3000 + t))
        report, _ = train_and_evaluate(stock, _indices(900, 3000 + t), make_config(), H, ["logistic"])
        clears += int(report.results[0].clears_baselines)
    assert clears <= 2, f"{clears}/15 noise series falsely 'cleared the baselines'"


def test_a_genuine_signal_is_detected_and_the_report_says_the_simple_rule_is_just_as_good():
    """The planted signal is 'the last move continues' (autocorrelated returns).
    Models must detect it (beat chance, rank well, lean on the 1-bar return
    features) — but the `previous_direction` baseline encodes that exact
    mechanism, so the honest verdict is that models do NOT clear the baselines
    by much. The framework reporting that, instead of crowing about a win over
    chance, is the behavior being tested."""
    report, _ = _run(ar_walk(N, 7), ["logistic", "hist_gradient_boosting"])
    assert report.best_baseline_name == "previous_direction"
    assert report.best_baseline_accuracy > 0.68
    for r in report.results:
        assert r.test.roc_auc > 0.65, (r.name, r.test.roc_auc)
        assert r.test.classification.accuracy > 0.60, (r.name, r.test.classification.accuracy)
        assert r.verdict.startswith("BEATS"), (r.name, r.verdict)  # real skill versus a constant guess
        # ...and the flag is exactly the documented rule, not a looser one
        assert r.clears_baselines == (r.test.classification.accuracy > report.best_baseline_accuracy)
    chosen = next(r for r in report.results if r.name == report.selected_by_validation)
    top = [i["feature"] for i in chosen.importances[:3]]
    assert any(f.startswith(("return_", "log_return_", "roc_", "relative_return_")) for f in top), top


def test_the_clears_baselines_rule_needs_both_a_real_beat_and_outscoring_the_best_baseline():
    beats = "BEATS A CONSTANT GUESS (approximate test, adjusted for overlapping labels; still not proof)"
    assert clears_baselines(beats, 0.70, 0.65) is True
    assert clears_baselines(beats, 0.65, 0.70) is False  # beats chance, but a simple rule does better
    assert clears_baselines(beats, 0.70, 0.70) is False  # a tie is not a win
    assert clears_baselines("NOT DISTINGUISHABLE FROM CHANCE", 0.90, 0.50) is False
    assert clears_baselines("ABOVE 50% BUT NO BETTER THAN ALWAYS GUESSING THE MAJORITY CLASS", 0.9, 0.5) is False
    assert clears_baselines("SIGNIFICANTLY WORSE THAN CHANCE (anti-predictive on this sample; not a usable signal)", 0.3, 0.2) is False


def test_trees_find_structure_that_no_simple_baseline_can_see():
    """Momentum in high-volatility blocks, mean-reversion in low-volatility
    blocks: previous_direction averages ~50%, but a tree model combining the
    last return with the volatility regime can do better. Asserted on an
    average over a FIXED set of seeds chosen in advance — single-seed results
    are noisy (the test sets hold only ~100 independent outcomes) and picking
    a lucky seed would prove nothing."""
    aucs, prev_dir = [], []
    for seed in (1, 2, 3, 4):
        stock = frame(regime_walk(3000, seed))
        report, _ = train_and_evaluate(stock, _indices(3000, seed), make_config(), H, ["random_forest"])
        aucs.append(report.results[0].test.roc_auc)
        prev_dir.append(next(b.accuracy for b in report.baselines if b.name == "previous_direction"))
    assert np.mean(prev_dir) < 0.60, prev_dir  # the simple rule is blind to this structure
    assert np.mean(aucs) > 0.58, aucs  # noise-only series measured 0.49-0.53


def test_model_selection_never_looks_at_the_test_segment():
    """Section 32: scramble ONLY the test period's prices. Everything chosen
    or fitted from train/validation must be bit-for-bit unchanged; only the
    test results may move."""
    series = ar_walk(N, 11)
    names = ["logistic", "random_forest"]
    base, _ = _run(series, names, seed=11)

    test_start = base.test_period[0]

    def scramble(df):
        out = df.copy()
        cols = out.columns.get_indexer(["open", "high", "low", "close"])
        out.iloc[(out.index >= test_start).argmax():, cols] *= 3.0
        return out

    stock = scramble(frame(series))
    idx = {k: scramble(v) for k, v in _indices(N, 11).items()}
    after, _ = train_and_evaluate(stock, idx, make_config(), H, names, "THCCL", "1m")

    assert base.n_rows == after.n_rows
    for a, b in zip(base.results, after.results):
        assert a.params == b.params and a.grid == b.grid
        assert a.validation_log_loss == b.validation_log_loss
        assert a.importances == b.importances  # permutation importance is validation-only
        assert (a.validation.brier, a.validation.log_loss, a.validation.roc_auc) == \
               (b.validation.brier, b.validation.log_loss, b.validation.roc_auc)
        assert not np.allclose(a.test_predictions["p_up"], b.test_predictions["p_up"])  # the test DID change
    assert base.selected_by_validation == after.selected_by_validation


def test_label_shuffle_control_shuffled_labels_average_to_chance_and_the_real_model_stands_apart():
    """Leakage control. Train on SHUFFLED labels many times; if test skill came
    from leakage, the average shuffled-label AUC would sit well above 0.5.

    Why an average and not a single draw: a model fit to shuffled labels is a
    RANDOM function of the features. When the features genuinely relate to the
    test labels (as in this planted-signal series), that random function's AUC
    lands far from 0.5 in either direction depending on its random sign
    (measured sd ~0.085, with tails both above 0.6 and below 0.4). Only the MEAN over
    shuffles is expected to be 0.5. (An earlier single-draw version of this
    test failed for exactly that reason; investigation showed means of
    0.490 / 0.504 over 30 shuffles vs real AUCs of 0.78, i.e. no leakage, and the test was fixed.)
    """
    cfg = make_config()
    ds = build_dataset(frame(ar_walk(N, 7)), _indices(N, 7), cfg, H)
    s = split_chronological(ds.positions, 0.6, 0.2, H)
    X_tr, y_tr = ds.X.iloc[s.train], ds.y.iloc[s.train]
    X_va, y_va = ds.X.iloc[s.validation], ds.y.iloc[s.validation]
    X_te, y_te = ds.X.iloc[s.test], ds.y.iloc[s.test]

    def test_auc(fit):
        return mm.probability_metrics(y_te, mdl.predict_up_probability(fit.estimator, X_te)).roc_auc

    for name, shuffles, tolerance in (("logistic", 20, 0.08), ("random_forest", 10, 0.10)):
        real_auc = test_auc(mdl.fit_with_validation_selection(name, X_tr, y_tr, X_va, y_va, 42))
        shuffled = []
        for k in range(shuffles):
            labels = np.random.default_rng(100 + k).permutation(y_tr.to_numpy())
            shuffled.append(test_auc(mdl.fit_with_validation_selection(name, X_tr, labels, X_va, y_va, 42)))
        shuffled = np.array(shuffled)
        assert abs(shuffled.mean() - 0.5) < tolerance, (name, shuffled.mean())
        assert real_auc > 0.65, (name, real_auc)  # the real model has genuine skill here...
        assert real_auc > shuffled.mean() + 1.5 * shuffled.std(), (name, real_auc, shuffled.mean(), shuffled.std())


def test_same_seed_reproduces_identical_predictions():
    a, _ = _run(ar_walk(N, 3), ["random_forest"], seed=3)
    b, _ = _run(ar_walk(N, 3), ["random_forest"], seed=3)
    np.testing.assert_array_equal(a.results[0].test_predictions["p_up"], b.results[0].test_predictions["p_up"])


def test_unknown_model_and_too_little_data_fail_loudly_before_training():
    with pytest.raises(ValueError):
        _run(random_walk(N, 7), ["deep_neural_net"])
    with pytest.raises(ValueError) as exc_info:
        train_and_evaluate(frame(random_walk(120, 1)), _indices(120, 1), make_config(), H, ["logistic"])
    assert "min_rows" in str(exc_info.value) or "Cannot split" in str(exc_info.value) or "empty" in str(exc_info.value)


def test_calibration_table_and_importances_are_populated():
    report, _ = _run(ar_walk(N, 7), ["logistic"])
    r = report.results[0]
    assert sum(row["n"] for row in r.calibration_table) == report.n_test
    assert len(r.importances) == len(report.feature_names)
    assert [i["rank"] for i in r.importances] == list(range(1, len(r.importances) + 1))
