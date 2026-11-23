"""Tests for src.models.model_manager and trainer.train_and_store (Phase 15)."""

from __future__ import annotations

import datetime as dt
import json
import re
import sys
import types
import warnings
from dataclasses import dataclass

import numpy as np
import pytest

from src.models import ml_models as mdl
from src.models import model_manager as mgr
from src.models.dataset import build_dataset
from src.models.trainer import train_and_store
from tests._ml_helpers import ar_walk, frame, make_config, random_walk

warnings.filterwarnings("ignore")
H = 5
N = 900
ID_PATTERN = re.compile(r"^THCCL_1M_5B_(LR|RF|GB|HGB)_\d{8}_\d{3}$")


def test_model_id_format_and_sequence_numbering():
    day = dt.date(2026, 9, 14)
    first = mgr.make_model_id("THCCL", "1m", 15, "random_forest", day, [])
    assert first == "THCCL_1M_15B_RF_20260914_001"
    second = mgr.make_model_id("THCCL", "1m", 15, "random_forest", day, [first])
    assert second == "THCCL_1M_15B_RF_20260914_002"
    # a different model, symbol or day starts its own sequence
    assert mgr.make_model_id("THCCL", "1m", 15, "logistic", day, [first]).endswith("LR_20260914_001")
    assert mgr.make_model_id("OGDC", "1m", 15, "random_forest", day, [first]).endswith("_001")
    assert mgr.make_model_id("THCCL", "1m", 15, "random_forest", dt.date(2026, 9, 15), [first]).endswith("_001")


def _tiny_model():
    X = np.random.default_rng(0).normal(size=(120, 4))
    y = (X[:, 0] > 0).astype(float)
    est = mdl.make_estimator("logistic", {"C": 1.0}, 0).fit(X, y)
    cal = mdl.PlattCalibrator().fit(mdl.predict_up_probability(est, X), y)
    return est, cal, X


def test_save_and_load_round_trip_gives_identical_predictions(tmp_path):
    est, cal, X = _tiny_model()
    mgr.save_model(tmp_path, "M_001", est, cal, ["a", "b", "c", "d"], {"symbol": "THCCL", "random_seed": 9})
    loaded = mgr.load_model(tmp_path, "M_001")
    before = cal.transform(mdl.predict_up_probability(est, X))
    after = loaded.calibrator.transform(mdl.predict_up_probability(loaded.estimator, X))
    np.testing.assert_array_equal(before, after)
    assert loaded.feature_names == ["a", "b", "c", "d"]
    assert loaded.version_warnings == []


def test_metadata_records_everything_needed_to_reproduce(tmp_path):
    est, cal, _ = _tiny_model()
    _, meta_path = mgr.save_model(tmp_path, "M_002", est, cal, ["a", "b", "c", "d"], {"random_seed": 9, "symbol": "X"})
    meta = json.loads(meta_path.read_text())
    assert meta["random_seed"] == 9 and meta["feature_names"] == ["a", "b", "c", "d"]
    assert {"python", "scikit-learn", "pandas", "numpy", "joblib"} <= set(meta["versions"])
    assert "created_at_utc" in meta and "calibrator_is_identity" in meta


def test_loading_with_different_library_versions_warns_instead_of_failing_silently(tmp_path):
    est, cal, _ = _tiny_model()
    _, meta_path = mgr.save_model(tmp_path, "M_003", est, cal, ["a", "b", "c", "d"], {})
    meta = json.loads(meta_path.read_text())
    meta["versions"]["scikit-learn"] = "0.0.1"
    meta_path.write_text(json.dumps(meta))
    warnings_out = mgr.load_model(tmp_path, "M_003").version_warnings
    assert any("scikit-learn" in w for w in warnings_out)


def test_loading_a_missing_model_raises_a_clear_error(tmp_path):
    with pytest.raises(FileNotFoundError):
        mgr.load_model(tmp_path, "NOPE_001")


# ---------------- train_and_store with a fake repository ----------------

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


def _candles(df):
    return [_Candle(ts.to_pydatetime(), r.open, r.high, r.low, r.close, r.volume) for ts, r in df.iterrows()]


def _install(monkeypatch, frames):
    store = {"runs": [], "predictions": []}
    fake = types.ModuleType("src.database.repository")
    fake.get_candles = lambda session, symbol, timeframe, start, end: _candles(frames[symbol])
    fake.insert_model_run = lambda session, **kw: (store["runs"].append(kw), True)[1]
    fake.insert_prediction = lambda session, **kw: (store["predictions"].append(kw), True)[1]
    monkeypatch.setitem(sys.modules, "src.database.repository", fake)
    return store


def _frames():
    return {
        "THCCL": frame(ar_walk(N, 5)),
        "KSE100": frame(random_walk(N, 15, 45000.0, 30.0), volume_seed=3),
        "KMI30": frame(random_walk(N, 25, 70000.0, 40.0), volume_seed=4),
    }


def _store(frames, models_dir, names=("logistic", "random_forest")):
    t0 = frames["THCCL"].index[0].to_pydatetime()
    return train_and_store(
        make_config(), lambda: _Session(), "THCCL", "1m", t0, t0 + dt.timedelta(days=2), H,
        list(names), models_dir=models_dir,
    )


def test_train_and_store_saves_files_records_runs_and_stores_test_predictions(monkeypatch, tmp_path):
    frames = _frames()
    store = _install(monkeypatch, frames)
    report = _store(frames, tmp_path)

    assert len(store["runs"]) == 2
    for run in store["runs"]:
        assert ID_PATTERN.match(run["model_id"])
        assert run["status"] == "TRAINED" and run["model_type"].startswith("ml:")
        assert run["random_seed"] == 42 and run["dataset_version"] == "ml_features_v1"
        assert run["training_end"] < run["validation_start"] and run["validation_end"] < run["test_start"]
        assert json.loads(run["feature_list_json"]) == report.feature_names
        assert "NaN" not in run["metrics_json"]
        assert json.loads(run["metrics_json"])["test"]["n"] == report.n_test
        assert (tmp_path / "trained" / f"{run['model_id']}.joblib").is_file()
        assert (tmp_path / "metadata" / f"{run['model_id']}.json").is_file()

    assert len(store["predictions"]) == 2 * report.n_test
    for p in store["predictions"]:
        assert p["probability_neutral"] is None and p["expected_return"] is None  # not modeled: stored as NULL
        assert abs(p["probability_up"] + p["probability_down"] - 1.0) < 1e-9
        assert p["prediction"] == ("UP" if p["probability_up"] >= 0.5 else "DOWN")
        assert p["feature_version"] == "ml_features_v1"
    assert {p["model_version"] for p in store["predictions"]} == set(report.model_ids.values())


def test_a_saved_model_reproduces_its_own_stored_test_predictions(monkeypatch, tmp_path):
    frames = _frames()
    _install(monkeypatch, frames)
    report = _store(frames, tmp_path, names=("logistic",))
    result = report.results[0]

    loaded = mgr.load_model(tmp_path, report.model_ids["logistic"])
    ds = build_dataset(frames["THCCL"], {k: v for k, v in frames.items() if k != "THCCL"}, make_config(), H)
    X_test = ds.X.loc[result.test_predictions.index][loaded.feature_names]
    p = loaded.calibrator.transform(mdl.predict_up_probability(loaded.estimator, X_test))
    np.testing.assert_allclose(p, result.test_predictions["p_up"].to_numpy(), rtol=1e-9, atol=1e-12)


def test_running_twice_creates_a_second_sequence_number_not_an_overwrite(monkeypatch, tmp_path):
    frames = _frames()
    _install(monkeypatch, frames)
    first = _store(frames, tmp_path, names=("logistic",))
    second = _store(frames, tmp_path, names=("logistic",))
    a, b = first.model_ids["logistic"], second.model_ids["logistic"]
    assert a != b and a.endswith("_001") and b.endswith("_002")
    assert len(list((tmp_path / "trained").glob("*.joblib"))) == 2
