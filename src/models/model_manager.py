"""Model identity, saving and loading (Phase 15, Sections 34/59/100).

Every trained model gets an ID like `THCCL_1M_5B_RF_20261023_001`
(symbol, timeframe, horizon in bars, model abbreviation, date, sequence) and
is saved as two files:

    models/trained/<id>.joblib   the fitted estimator + calibrator + feature list
    models/metadata/<id>.json    everything needed to reproduce and audit it:
                                 hyperparameters, feature list, training /
                                 validation / test periods, seed, metrics,
                                 and the Python / scikit-learn / pandas /
                                 numpy versions it was trained with

SECURITY: .joblib files are pickles. Loading one can execute arbitrary code,
so only load model files this project wrote itself, never ones from an
untrusted source.
"""

from __future__ import annotations

import datetime as dt
import json
import platform
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

import joblib
import numpy as np
import pandas as pd
import sklearn

from src.models.ml_models import MODEL_ABBREVIATIONS
from src.utils.file_utils import resolve_path


def current_versions() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "scikit-learn": sklearn.__version__,
        "pandas": pd.__version__,
        "numpy": np.__version__,
        "joblib": joblib.__version__,
    }


def default_models_dir() -> Path:
    return resolve_path("models")


def make_model_id(
    symbol: str, timeframe: str, horizon: int, model_name: str, on_date: dt.date, existing_ids: Iterable[str]
) -> str:
    """Build the next unused ID for this symbol/timeframe/horizon/model/date."""
    prefix = f"{symbol}_{timeframe.upper()}_{horizon}B_{MODEL_ABBREVIATIONS[model_name]}_{on_date:%Y%m%d}_"
    taken = [
        int(i[len(prefix):]) for i in existing_ids if i.startswith(prefix) and i[len(prefix):].isdigit()
    ]
    return f"{prefix}{max(taken, default=0) + 1:03d}"


def existing_model_ids(models_dir: Path) -> list[str]:
    meta = models_dir / "metadata"
    return sorted(p.stem for p in meta.glob("*.json")) if meta.is_dir() else []


def save_model(
    models_dir: Path, model_id: str, estimator, calibrator, feature_names: list[str], metadata: dict
) -> tuple[Path, Path]:
    trained = models_dir / "trained"
    meta = models_dir / "metadata"
    trained.mkdir(parents=True, exist_ok=True)
    meta.mkdir(parents=True, exist_ok=True)

    model_path = trained / f"{model_id}.joblib"
    meta_path = meta / f"{model_id}.json"
    full_meta = {
        **metadata,
        "model_id": model_id,
        "feature_names": list(feature_names),
        "calibrator_is_identity": bool(calibrator.is_identity),
        "created_at_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "versions": current_versions(),
    }
    joblib.dump({"estimator": estimator, "calibrator": calibrator, "feature_names": list(feature_names)}, model_path)
    meta_path.write_text(json.dumps(full_meta, indent=2, default=str), encoding="utf-8")
    return model_path, meta_path


@dataclass
class LoadedModel:
    estimator: object
    calibrator: object
    feature_names: list[str]
    metadata: dict
    version_warnings: list[str] = field(default_factory=list)


def load_model(models_dir: Path, model_id: str) -> LoadedModel:
    """Load a model this project saved (see the SECURITY note above).

    A mismatch between the versions it was trained with and the ones
    installed now is returned as `version_warnings`, not raised — pickled
    scikit-learn models can misbehave across versions, and the caller should
    decide whether to trust the result.
    """
    model_path = models_dir / "trained" / f"{model_id}.joblib"
    meta_path = models_dir / "metadata" / f"{model_id}.json"
    if not model_path.is_file() or not meta_path.is_file():
        raise FileNotFoundError(f"Model {model_id!r} not found under {models_dir}.")

    bundle = joblib.load(model_path)
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    now = current_versions()
    warnings = [
        f"{pkg}: trained with {then}, installed {now.get(pkg)}"
        for pkg, then in metadata.get("versions", {}).items()
        if now.get(pkg) != then
    ]
    return LoadedModel(bundle["estimator"], bundle["calibrator"], bundle["feature_names"], metadata, warnings)
