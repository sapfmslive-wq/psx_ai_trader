"""The four ML classifiers and how they are fit (Phase 15, Sections 24/33/82).

CPU-friendly by design (the target is ~2 cores / ~6 GB): small fixed
hyperparameter grids, shallow trees, one thread (`n_jobs=1` plus a
threadpool limit), no deep learning.

Hyperparameters are chosen ONLY on the validation segment, by validation log
loss (a proper scoring rule that rewards honest probabilities, not just
correct guesses). The chosen model is fit on the training segment alone — it
is not refit on train+validation, because the validation segment is then
used a second time to fit the probability calibrator. That reuse is disclosed
in the report; the TEST segment is never touched by any of it (Section 32).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits

MODEL_ABBREVIATIONS = {
    "logistic": "LR",
    "random_forest": "RF",
    "gradient_boosting": "GB",
    "hist_gradient_boosting": "HGB",
}

_EPS = 1e-6


def candidate_params(name: str) -> list[dict]:
    """The (deliberately small) hyperparameter grid for one model."""
    if name == "logistic":
        return [{"C": c} for c in (0.01, 0.1, 1.0, 10.0)]
    if name == "random_forest":
        return [{"max_depth": d, "min_samples_leaf": m} for d in (3, 6) for m in (10, 30)]
    if name == "gradient_boosting":
        return [{"learning_rate": lr, "max_depth": d} for lr in (0.05, 0.1) for d in (2, 3)]
    if name == "hist_gradient_boosting":
        return [{"learning_rate": lr, "max_depth": d} for lr in (0.05, 0.1) for d in (3, 5)]
    raise ValueError(f"Unknown model name: {name!r}")


def make_estimator(name: str, params: dict, seed: int):
    if name == "logistic":
        return Pipeline([
            ("scale", StandardScaler()),
            ("model", LogisticRegression(C=params["C"], max_iter=2000)),
        ])
    if name == "random_forest":
        return RandomForestClassifier(
            n_estimators=100, max_depth=params["max_depth"], min_samples_leaf=params["min_samples_leaf"],
            random_state=seed, n_jobs=1,
        )
    if name == "gradient_boosting":
        return GradientBoostingClassifier(
            n_estimators=100, learning_rate=params["learning_rate"], max_depth=params["max_depth"],
            subsample=0.8, random_state=seed,
        )
    if name == "hist_gradient_boosting":
        return HistGradientBoostingClassifier(
            max_iter=100, learning_rate=params["learning_rate"], max_depth=params["max_depth"],
            l2_regularization=1.0, early_stopping=False, random_state=seed,
        )
    raise ValueError(f"Unknown model name: {name!r}")


def predict_up_probability(estimator, X) -> np.ndarray:
    """P(UP) from any fitted estimator, looked up by class label (never by
    column position)."""
    classes = list(estimator.classes_)
    return estimator.predict_proba(X)[:, classes.index(1.0)]


def _clip(p) -> np.ndarray:
    return np.clip(np.asarray(p, dtype=float), _EPS, 1 - _EPS)


@dataclass
class FitResult:
    estimator: object
    params: dict
    validation_log_loss: float
    grid: list[dict] = field(default_factory=list)  # every candidate and its validation log loss


def fit_with_validation_selection(name, X_train, y_train, X_val, y_val, seed: int) -> FitResult:
    """Fit every grid candidate on train, score on validation, keep the best."""
    if len(np.unique(y_train)) < 2:
        raise ValueError("Training labels contain a single class; cannot fit a classifier.")

    best: FitResult | None = None
    grid: list[dict] = []
    with threadpool_limits(limits=1):
        for params in candidate_params(name):
            estimator = make_estimator(name, params, seed)
            estimator.fit(X_train, y_train)
            p_val = _clip(predict_up_probability(estimator, X_val))
            loss = float(log_loss(y_val, p_val, labels=[0.0, 1.0]))
            grid.append({"params": params, "validation_log_loss": loss})
            if best is None or loss < best.validation_log_loss:
                best = FitResult(estimator, params, loss)
    best.grid = grid
    return best


class PlattCalibrator:
    """Platt scaling: a 1-D logistic regression mapping a model's raw
    probability (on the logit scale) to a calibrated one, fit on validation
    data. Monotonic, so it never changes how bars are RANKED (ROC-AUC is
    unaffected) — only how trustworthy the stated probabilities are. Falls
    back to the identity (and says so) when validation is too small or has
    a single class."""

    MIN_ROWS = 30

    def __init__(self) -> None:
        self._model: LogisticRegression | None = None
        self.is_identity = True

    @staticmethod
    def _logit(p) -> np.ndarray:
        p = _clip(p)
        return np.log(p / (1 - p)).reshape(-1, 1)

    def fit(self, p_raw, y) -> "PlattCalibrator":
        y = np.asarray(y, dtype=float)
        if len(y) < self.MIN_ROWS or len(np.unique(y)) < 2:
            self._model, self.is_identity = None, True
            return self
        self._model = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(p_raw), y)
        self.is_identity = False
        return self

    def transform(self, p_raw) -> np.ndarray:
        p_raw = np.asarray(p_raw, dtype=float)
        if self._model is None:
            return p_raw
        return self._model.predict_proba(self._logit(p_raw))[:, 1]
