"""Individual-client model comparison page for TAX-XAI.

Shows live predictions from any Phase 3 classical model artifacts that are
available locally, plus the existing Phase 4 MLP. Missing model artifacts are
reported explicitly rather than reconstructed or silently substituted.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

from services import datasets, pipeline, predictor, records

ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"
CLASS_NAMES = list(pipeline.CLASSES)
MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "XGBoost": MODEL_DIR / "xgboost.joblib",
    "MLP": MODEL_DIR / "mlp_final_selected.pt",
}


def _load_profiles() -> pd.DataFrame:
    return datasets.load_modelled_profiles().copy()


def _raw_row(df: pd.DataFrame, profile_id: str) -> dict:
    row = df.loc[df["profile_id"].astype(str) == str(profile_id)]
    if row.empty:
        raise ValueError(f"Profile {profile_id} was not found.")
    return row.iloc[0].to_dict()


def _class_result(probs: np.ndarray) -> dict:
    probs = np.asarray(probs, dtype=float).reshape(-1)
    pred_idx = int(np.argmax(probs))
    return {"probs": probs, "pred_idx": pred_idx, "pred_class": CLASS_NAMES[pred_idx]}


def _predict_classical(model, raw: dict) -> dict:
    x = pipeline.encode(raw)
    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()
    x = np.asarray(x, dtype=float)
    if x.ndim == 1:
        x = x.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(x), dtype=float)[0]
        # Align columns if a model exposes a subset/order of classes.
        if hasattr(model, "classes_"):
            aligned = np.zeros(len(CLASS_NAMES), dtype=float)
            for col, cls in enumerate(model.classes_):
                try:
                    aligned[int(cls)] = probs[col]
                except (ValueError, TypeError, IndexError):
                    if str(cls) in CLASS_NAMES:
                        aligned[CLASS_NAMES.index(str(cls))] = probs[col]
            if aligned.sum() > 0:
                probs = aligned / aligned.sum()
        return _class_result(probs)

    pred = model.predict(x)[0]
    probs = np.zeros(len(CLASS_NAMES), dtype=float)
    try:
        idx = int(pred)
    except (TypeError, ValueError):
        idx = CLASS_NAMES.index(str(pred))
    probs[idx] = 1.0
    return _class_result(probs)


def _predict_one(name: str, raw: dict) -> tuple[dict | None, str | None]:
    path = MODEL_FILES[name]
    if not path.exists():
        return None, f"Artifact not found: {path.name}"
    try:
        if name == "MLP":
            return predictor.predict(raw), None
        import joblib
        model = joblib.load(path)
        return _predict_classical(model, raw), None
    except Exception as exc:  # keep one bad artifact from breaking the page
        return None, f"Could not evaluate {name}: {type(exc).__name__}: {exc}"


def _bar_chart(results: dict[str, dict]) -> None:
    rows = []
    for name, result in results.items():
        for idx, cls in enumerate(CLASS_NAMES):
            rows.append({"Model": name, "Risk": cls, "Probability": float(result["probs"][idx])})
    chart = pd.DataFrame(rows)
    pivot = chart.pivot(index="Model", columns="Risk", values="Probability")
    st.bar_chart(pivot, y=CLASS_NAMES, x_label="Model", y_label="Probability")


def render() -> None:
    st.title("Model Comparison")
    st.caption("Individual-client comparison using the model artifacts available in this project.")

    profiles = _load_profiles()
    ids = profiles["profile_id"].astype(str).tolist()
    default_id = "TAXP04291" if "TAXP04291" in ids else ids[0]
    profile_id = st.selectbox("Select taxpayer", ids, index=ids.index(default_id))
    raw = _raw_row(profiles, profile_id)

    st.info(
        "This view compares model outputs for the same taxpayer. It is a model-level "
        "comparison, not a statement that one model is legally or operationally preferable."
    )

    results: dict[str, dict] = {}
    errors: dict[str, str] = {}
    for name in MODEL_FILES:
        result, error = _predict_one(name, raw)
        if result is not None:
            results[name] = result
        else:
            errors[name] = error or "Unavailable"

    if not results:
        st.error("No model artifact could be evaluated for this taxpayer.")
        return

    st.subheader(f"Predictions for {profile_id}")
    cols = st.columns(len(results))
    for col, (name, result) in zip(cols, results.items()):
        with col:
            st.metric(name, result["pred_class"])
            st.caption(f"Low {result['probs'][0]:.1%} · Medium {result['probs'][1]:.1%} · High {result['probs'][2]:.1%}")

    st.subheader("Probability comparison")
    _bar_chart(results)

    table_rows = []
    for name, result in results.items():
        table_rows.append({
            "Model": name,
            "Prediction": result["pred_class"],
            "Low": result["probs"][0],
            "Medium": result["probs"][1],
            "High": result["probs"][2],
        })
    table = pd.DataFrame(table_rows).set_index("Model")
    st.dataframe(table.style.format({"Low": "{:.2%}", "Medium": "{:.2%}", "High": "{:.2%}"}), use_container_width=True)

    predictions = [r["pred_class"] for r in results.values()]
    if len(set(predictions)) == 1:
        st.success(f"All available models predict {predictions[0]} for this taxpayer.")
    else:
        st.warning("The available models disagree on the predicted risk class. Review the probability distribution and model-level metrics before interpreting the result.")

    if errors:
        with st.expander("Model artifacts not available", expanded=False):
            for name, error in errors.items():
                st.caption(f"**{name}:** {error}")

    st.subheader("Reference inputs")
    shown = [c for c in datasets.NUMERIC_FEATURES + [datasets.CATEGORICAL_FEATURE] if c in raw]
    st.dataframe(pd.DataFrame([{c: raw.get(c) for c in shown}]), use_container_width=True)
