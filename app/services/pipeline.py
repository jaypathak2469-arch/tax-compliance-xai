"""Live access to the trained pipeline.

Every function here loads an existing artifact or replays the exact encode /
decode recipe used by code/run_phase5.py. Nothing is trained, fitted or
refit. The MLP and preprocessor are loaded once per session via
st.cache_resource.
"""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from . import paths

CLASSES = ["Low", "Medium", "High"]


class PipelineError(RuntimeError):
    """Raised when an artifact or dependency required for inference is missing."""


def _require_torch():
    try:
        import torch  # noqa: F401
        return torch
    except ImportError as exc:  # pragma: no cover - depends on the host env
        raise PipelineError(
            "PyTorch is not installed. Run: python -m pip install -r requirements-app.txt"
        ) from exc


def require_torch():
    """Public entry point for other services that need the torch module."""
    return _require_torch()


@st.cache_resource(show_spinner=False)
def get_preprocessor():
    path = paths.resolve_preprocessor()
    if path is None:
        raise PipelineError(
            f"Preprocessor not found at {paths.PREPROCESSOR_PRIMARY} or "
            f"{paths.PREPROCESSOR_FALLBACK}."
        )
    return joblib.load(path)


@st.cache_resource(show_spinner=False)
def get_scaler_stats() -> pd.DataFrame:
    if not paths.SCALER_STATS.exists():
        raise PipelineError(f"Scaler statistics not found at {paths.SCALER_STATS}.")
    return pd.read_csv(paths.SCALER_STATS).set_index("feature")


@st.cache_resource(show_spinner=False)
def get_feature_index() -> tuple[list[str], dict[str, int]]:
    names = list(get_preprocessor().get_feature_names_out())
    return names, {f: i for i, f in enumerate(names)}


@st.cache_resource(show_spinner=False)
def get_model():
    """Load the existing Phase 4 checkpoint. Never trains or refits."""
    torch = _require_torch()
    if not paths.CODE_DIR.is_dir():
        raise PipelineError(f"code/ directory not found at {paths.CODE_DIR}.")
    if not paths.MODEL_CHECKPOINT.exists():
        raise PipelineError(f"Checkpoint not found at {paths.MODEL_CHECKPOINT}.")
    paths.ensure_code_on_path()
    from counterfactual import load_model  # existing Phase 5 module

    names, _ = get_feature_index()
    return load_model(paths.MODEL_CHECKPOINT, input_dim=len(names))


def numeric_and_categorical_columns() -> tuple[list[str], list[str]]:
    """The raw-input column split expected by the preprocessor, straight from
    scaler_stats (numeric) plus the fixed categorical column name."""
    scaler_stats = get_scaler_stats()
    return list(scaler_stats.index), ["income_sources"]


def encode(raw: dict):
    """Raw taxpayer dict -> standardized model input tensor.

    Mirrors code/run_phase5.py::encode() exactly: build a one-row DataFrame in
    [numeric..., categorical] order and pass it through the fitted preprocessor.
    """
    torch = _require_torch()
    pre = get_preprocessor()
    num, cat = numeric_and_categorical_columns()
    row = pd.DataFrame([raw])[num + cat]
    z = pre.transform(row).astype(np.float32)[0]
    return torch.tensor(z)


def decode_raw(z) -> dict:
    """Standardized tensor -> raw-unit dict. Mirrors run_phase5.py::decode_raw()."""
    paths.ensure_code_on_path()
    from constraints import decode_income_source  # existing Phase 5 module

    scaler_stats = get_scaler_stats()
    _, feature_to_idx = get_feature_index()
    out = {}
    for f in scaler_stats.index:
        j = feature_to_idx[f]
        out[f] = float(
            z[j].detach().cpu().item() * scaler_stats.loc[f, "scale"]
            + scaler_stats.loc[f, "mean"]
        )
    out["income_sources"] = decode_income_source(z, feature_to_idx)
    return out


def predict_probs(x0) -> np.ndarray:
    """Softmax class probabilities for a single standardized input."""
    torch = _require_torch()
    model = get_model()
    with torch.no_grad():
        return torch.softmax(model(x0.unsqueeze(0)), dim=1)[0].numpy()


def status() -> tuple[bool, str]:
    """Whether the live pipeline can currently run, and why not if it can't."""
    try:
        get_model()
        get_preprocessor()
        get_scaler_stats()
        return True, ""
    except PipelineError as exc:
        return False, str(exc)
