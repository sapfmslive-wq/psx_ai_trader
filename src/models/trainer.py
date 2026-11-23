"""Model training and evaluation (Phase 15).

    train_and_evaluate()  - pure: candle DataFrames + config in, a
                            TrainingReport and the fitted models out.
    train_and_store()     - reads candles, calls the above, saves each model
                            (joblib + metadata JSON), records a `model_runs`
                            row, and stores the TEST-segment predictions.

How the evaluation is kept honest (Sections 31, 32, 61, 101):
  * One chronological train / validation / test split, never shuffled, with
    a `horizon`-bar purge at each boundary.
  * Hyperparameters are chosen on VALIDATION log loss only. The probability
    calibrator is fit on validation too (so validation is used twice — once
    to choose, once to calibrate — and that is disclosed). Permutation
    importance is also computed on validation. The TEST segment influences
    nothing; tests assert that scrambling it leaves every choice unchanged.
  * "Which model is selected" is decided by validation log loss, not by test
    results. Test results are reported for ALL models.
  * Every model is scored on the same test rows as the Phase 14 baselines.
    A model only "clears the baselines" if its verdict is a genuine beat of a
    constant guess AND it out-scores the best baseline's accuracy.
  * Several models are tested on the same data, so the significance bar is
    tightened by the number of models compared (Bonferroni: 0.05 / n).
  * Significance uses the effective sample size (outcomes ~ rows / horizon),
    because overlapping labels are not independent (see evaluation.py).
No transaction costs are applied here — see src/models/__init__.py. A model
that predicts direction well can still lose money after fees and slippage.
"""

from __future__ import annotations

import datetime as dt
import json
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance
from threadpoolctl import threadpool_limits

from src.indicators.engine import candles_to_dataframe
from src.models import baseline
from src.models import evaluation as ev
from src.models import ml_metrics as mm
from src.models import ml_models as mdl
from src.models import model_manager as mgr
from src.models.dataset import DATASET_VERSION, build_dataset, split_chronological


@dataclass
class FittedModel:
    name: str
    estimator: object
    calibrator: object
    params: dict
    feature_names: list[str]


@dataclass
class BaselineOnTest:
    name: str
    accuracy: float
    balanced_accuracy: float
    coverage: float
    n_predicted: int
    n_effective: int
    p_value: float
    verdict: str


@dataclass
class ModelResult:
    name: str
    params: dict
    grid: list[dict]
    validation_log_loss: float  # raw model, the selection criterion
    validation: mm.ProbabilityMetrics  # calibrated
    test_raw: mm.ProbabilityMetrics
    test: mm.ProbabilityMetrics  # calibrated — the model's actual output
    calibrator_is_identity: bool
    n_effective: int
    p_value: float
    verdict: str
    clears_baselines: bool
    importances: list[dict]
    calibration_table: list[dict]
    test_predictions: pd.DataFrame  # index = test timestamps; p_up_raw, p_up


@dataclass
class TrainingReport:
    symbol: str
    timeframe: str
    horizon: int
    random_seed: int
    n_input_bars: int
    n_rows: int
    feature_names: list[str]
    dropped_columns: list[str]
    n_dropped_warmup_rows: int
    n_dropped_unlabeled_rows: int
    n_train: int
    n_validation: int
    n_test: int
    purged_from_train: int
    purged_from_validation: int
    train_period: tuple[pd.Timestamp, pd.Timestamp]
    validation_period: tuple[pd.Timestamp, pd.Timestamp]
    test_period: tuple[pd.Timestamp, pd.Timestamp]
    test_up_rate: float
    models_compared: int
    alpha: float
    baselines: list[BaselineOnTest]
    best_baseline_name: str
    best_baseline_accuracy: float
    selected_by_validation: str
    results: list[ModelResult] = field(default_factory=list)
    model_ids: dict[str, str] = field(default_factory=dict)


def clears_baselines(verdict: str, accuracy: float, best_baseline_accuracy: float) -> bool:
    """A model "clears the baselines" only if BOTH hold: its verdict is a
    genuine beat of a constant guess at the (tightened) significance bar, AND
    it out-scores the best baseline on the same rows. Beating chance is not
    enough — a simple rule that already does as well makes the model
    unnecessary, and saying so plainly is the point of having baselines."""
    return verdict.startswith("BEATS") and accuracy > best_baseline_accuracy


def _baseline_predictions(stock_df: pd.DataFrame, y_train: pd.Series, config) -> dict[str, pd.Series]:  # noqa: ANN001
    cfg, close = config.baselines, stock_df["close"]
    majority, _ = baseline.majority_class_predictions(y_train, stock_df.index)
    return {
        "buy_and_hold": baseline.buy_and_hold_predictions(stock_df.index),
        "previous_direction": baseline.previous_direction_predictions(close),
        "moving_average": baseline.moving_average_predictions(close, cfg.ma_fast_period, cfg.ma_slow_period),
        "rsi": baseline.rsi_predictions(close, config.indicators.rsi_period, cfg.rsi_oversold, cfg.rsi_overbought),
        "random": baseline.random_predictions(stock_df.index, cfg.random_seed),
        "majority_class": majority,
    }


def _importances(estimator, X_val: pd.DataFrame, y_val: pd.Series, repeats: int, seed: int) -> list[dict]:
    """Permutation importance on VALIDATION (never test): how much worse the
    log loss gets when one feature's values are shuffled. A ranking of what
    the model leans on — not evidence that a feature causes anything."""
    if y_val.nunique() < 2:
        return []
    with threadpool_limits(limits=1):
        result = permutation_importance(
            estimator, X_val, y_val, scoring="neg_log_loss", n_repeats=repeats, random_state=seed, n_jobs=1
        )
    order = np.argsort(-result.importances_mean)
    return [
        {"rank": rank + 1, "feature": X_val.columns[i], "importance": float(result.importances_mean[i]),
         "std": float(result.importances_std[i])}
        for rank, i in enumerate(order)
    ]


def train_and_evaluate(
    stock_df: pd.DataFrame,
    index_dfs: dict[str, pd.DataFrame],
    config,  # noqa: ANN001 — AppConfig (or a stand-in carrying the same sections)
    horizon: int,
    model_names: list[str] | None = None,
    symbol: str = "",
    timeframe: str = "",
) -> tuple[TrainingReport, dict[str, FittedModel]]:
    mc = config.models
    names = list(model_names or mc.enabled)
    for name in names:
        mdl.candidate_params(name)  # raises ValueError on an unknown model, before any work

    ds = build_dataset(stock_df, index_dfs, config, horizon)
    if len(ds.X) < mc.min_rows:
        raise ValueError(
            f"Only {len(ds.X)} usable rows after warm-up/label drops; models.min_rows is {mc.min_rows}. "
            f"Build more candles first."
        )
    split = split_chronological(ds.positions, mc.train_fraction, mc.validation_fraction, horizon)

    X_tr, y_tr = ds.X.iloc[split.train], ds.y.iloc[split.train]
    X_va, y_va = ds.X.iloc[split.validation], ds.y.iloc[split.validation]
    X_te, y_te = ds.X.iloc[split.test], ds.y.iloc[split.test]

    # ---- baselines, scored on exactly the rows the models will be scored on ----
    test_index = X_te.index
    baseline_rows: list[BaselineOnTest] = []
    for bname, pred in _baseline_predictions(stock_df, y_tr, config).items():
        m = ev.classification_metrics(y_te, pred.reindex(test_index))
        n_eff = ev.effective_sample_size(m.n_predicted, horizon)
        p = ev.approx_p_value_vs_chance(m.accuracy, n_eff)
        baseline_rows.append(BaselineOnTest(
            bname, m.accuracy, m.balanced_accuracy, m.coverage, m.n_predicted, n_eff, p, ev.verdict(m, p, n_eff)
        ))
    scored = [b for b in baseline_rows if b.n_effective >= ev.MIN_INDEPENDENT_OUTCOMES and not math.isnan(b.accuracy)]
    if scored:
        top = max(scored, key=lambda b: b.accuracy)
        best_name, best_acc = top.name, top.accuracy
    else:
        up = float(y_te.mean())
        best_name, best_acc = "constant guess", max(up, 1 - up)

    alpha = 0.05 / max(1, len(names))
    n_eff_models = ev.effective_sample_size(len(y_te), horizon)

    results: list[ModelResult] = []
    fitted: dict[str, FittedModel] = {}
    for name in names:
        fit = mdl.fit_with_validation_selection(name, X_tr, y_tr, X_va, y_va, mc.random_seed)
        with threadpool_limits(limits=1):
            p_va_raw = mdl.predict_up_probability(fit.estimator, X_va)
            p_te_raw = mdl.predict_up_probability(fit.estimator, X_te)
        calibrator = mdl.PlattCalibrator().fit(p_va_raw, y_va)
        p_va, p_te = calibrator.transform(p_va_raw), calibrator.transform(p_te_raw)

        test_cal = mm.probability_metrics(y_te, p_te)
        acc = test_cal.classification.accuracy
        p_value = ev.approx_p_value_vs_chance(acc, n_eff_models)
        verdict = ev.verdict(test_cal.classification, p_value, n_eff_models, alpha)
        clears = clears_baselines(verdict, acc, best_acc)

        results.append(ModelResult(
            name=name, params=fit.params, grid=fit.grid, validation_log_loss=fit.validation_log_loss,
            validation=mm.probability_metrics(y_va, p_va),
            test_raw=mm.probability_metrics(y_te, p_te_raw), test=test_cal,
            calibrator_is_identity=calibrator.is_identity,
            n_effective=n_eff_models, p_value=p_value, verdict=verdict, clears_baselines=clears,
            importances=_importances(fit.estimator, X_va, y_va, mc.permutation_repeats, mc.random_seed),
            calibration_table=mm.calibration_table(y_te, p_te),
            test_predictions=pd.DataFrame({"p_up_raw": p_te_raw, "p_up": p_te}, index=test_index),
        ))
        fitted[name] = FittedModel(name, fit.estimator, calibrator, fit.params, list(ds.feature_names))

    selected = min(results, key=lambda r: r.validation_log_loss).name  # chosen on VALIDATION only

    def period(rows: np.ndarray) -> tuple[pd.Timestamp, pd.Timestamp]:
        return ds.X.index[rows[0]], ds.X.index[rows[-1]]

    report = TrainingReport(
        symbol=symbol, timeframe=timeframe, horizon=horizon, random_seed=mc.random_seed,
        n_input_bars=ds.n_input_bars, n_rows=len(ds.X), feature_names=list(ds.feature_names),
        dropped_columns=ds.dropped_all_nan_columns, n_dropped_warmup_rows=ds.n_dropped_feature_nan_rows,
        n_dropped_unlabeled_rows=ds.n_dropped_unlabeled_rows,
        n_train=len(split.train), n_validation=len(split.validation), n_test=len(split.test),
        purged_from_train=split.purged_from_train, purged_from_validation=split.purged_from_validation,
        train_period=period(split.train), validation_period=period(split.validation), test_period=period(split.test),
        test_up_rate=float(y_te.mean()), models_compared=len(names), alpha=alpha,
        baselines=baseline_rows, best_baseline_name=best_name, best_baseline_accuracy=best_acc,
        selected_by_validation=selected, results=results,
    )
    return report, fitted


# ------------------------------ persistence ------------------------------

def _json_safe(value):
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating,)):
        return _json_safe(float(value))
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _metrics_dict(pm: mm.ProbabilityMetrics) -> dict:
    c = pm.classification
    return _json_safe({
        "n": pm.n, "up_rate": pm.up_rate, "accuracy": c.accuracy, "balanced_accuracy": c.balanced_accuracy,
        "precision_up": c.precision_up, "recall_up": c.recall_up, "f1_up": c.f1_up,
        "null_accuracy": c.null_accuracy, "roc_auc": pm.roc_auc, "pr_auc": pm.pr_auc,
        "brier": pm.brier, "log_loss": pm.log_loss, "ece": pm.ece,
        "confusion": {"tp": c.tp, "fp": c.fp, "tn": c.tn, "fn": c.fn},
    })


def train_and_store(
    config,  # noqa: ANN001 — AppConfig
    session_factory,  # noqa: ANN001
    symbol: str,
    timeframe: str,
    start: dt.datetime,
    end: dt.datetime,
    horizon: int,
    model_names: list[str] | None = None,
    models_dir=None,
) -> TrainingReport:
    """Train, save model files, record `model_runs`, store test predictions."""
    from src.database.repository import get_candles, insert_model_run, insert_prediction

    with session_factory() as session:
        stock_df = candles_to_dataframe(get_candles(session, symbol, timeframe, start, end))
        index_dfs = {
            idx: candles_to_dataframe(get_candles(session, idx, timeframe, start, end))
            for idx in config.symbols.indices if idx != symbol
        }

    report, fitted = train_and_evaluate(stock_df, index_dfs, config, horizon, model_names, symbol, timeframe)

    models_dir = models_dir or mgr.default_models_dir()
    today = dt.datetime.now(dt.timezone.utc).date()
    existing = mgr.existing_model_ids(models_dir)
    versions = mgr.current_versions()

    for result in report.results:
        model_id = mgr.make_model_id(symbol, timeframe, horizon, result.name, today, existing)
        existing.append(model_id)
        report.model_ids[result.name] = model_id
        model = fitted[result.name]

        summary = {
            "validation": _metrics_dict(result.validation),
            "test": _metrics_dict(result.test),
            "test_uncalibrated": _metrics_dict(result.test_raw),
            "calibrator_is_identity": result.calibrator_is_identity,
            "n_effective": result.n_effective, "p_value_vs_chance": result.p_value,
            "alpha": report.alpha, "models_compared": report.models_compared,
            "verdict": result.verdict, "clears_baselines": result.clears_baselines,
            "best_baseline": {"name": report.best_baseline_name, "accuracy": report.best_baseline_accuracy},
            "selected_by_validation": report.selected_by_validation == result.name,
        }
        mgr.save_model(models_dir, model_id, model.estimator, model.calibrator, model.feature_names, {
            "symbol": symbol, "timeframe": timeframe, "horizon_bars": horizon, "model_type": result.name,
            "hyperparameters": result.params, "hyperparameter_grid": result.grid,
            "dataset_version": DATASET_VERSION, "random_seed": report.random_seed,
            "train_period": [str(t) for t in report.train_period],
            "validation_period": [str(t) for t in report.validation_period],
            "test_period": [str(t) for t in report.test_period],
            "metrics": _json_safe(summary),
        })

        with session_factory() as session:
            insert_model_run(
                session,
                model_id=model_id, symbol=symbol, timeframe=timeframe, model_type=f"ml:{result.name}",
                training_start=report.train_period[0].to_pydatetime(),
                training_end=report.train_period[1].to_pydatetime(),
                validation_start=report.validation_period[0].to_pydatetime(),
                validation_end=report.validation_period[1].to_pydatetime(),
                test_start=report.test_period[0].to_pydatetime(),
                test_end=report.test_period[1].to_pydatetime(),
                hyperparameters_json=json.dumps(_json_safe({"params": result.params, "grid": result.grid,
                                                            "horizon_bars": horizon})),
                metrics_json=json.dumps(_json_safe(summary)),
                feature_list_json=json.dumps(model.feature_names),
                dataset_version=DATASET_VERSION, random_seed=report.random_seed,
                python_version=versions["python"], package_versions_json=json.dumps(versions),
                status="TRAINED",
            )
            for ts, row in result.test_predictions.iterrows():
                p_up = float(row["p_up"])
                insert_prediction(
                    session,
                    symbol=symbol, timeframe=timeframe, timestamp=ts.to_pydatetime(),
                    model_version=model_id, feature_version=DATASET_VERSION,
                    prediction="UP" if p_up >= 0.5 else "DOWN",
                    probability_up=p_up, probability_down=1.0 - p_up,
                    probability_neutral=None,  # binary model: no NEUTRAL class is modeled
                    expected_return=None,  # no return-magnitude regressor in this phase
                )
    return report
