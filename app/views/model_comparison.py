
"""Individual-client model comparison page for TAX-XAI.

Compares predictions from available Phase 3 classical models and the
existing Phase 4 MLP. Uses existing trained artifacts only.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import streamlit as st

from services import datasets, pipeline, predictor


ROOT = Path(__file__).resolve().parents[2]
MODEL_DIR = ROOT / "models"

CLASS_NAMES = list(pipeline.CLASSES)

MODEL_FILES = {
    "Logistic Regression": MODEL_DIR / "logistic_regression.joblib",
    "Random Forest": MODEL_DIR / "random_forest.joblib",
    "XGBoost": MODEL_DIR / "xgboost.joblib",
    "MLP": MODEL_DIR / "mlp_final_selected.pt",
}


@st.cache_data(show_spinner=False)
def _load_profiles() -> pd.DataFrame:
    return datasets.load_modelled_profiles().copy()


@st.cache_resource(show_spinner=False)
def _load_classical_model(name: str, path_string: str):
    """Load an existing classical model once per Streamlit session."""
    path = Path(path_string)

    if not path.exists():
        raise FileNotFoundError(f"Model artifact not found: {path.name}")

    model = joblib.load(path)

    # Avoid native multithreading instability during XGBoost inference.
    if hasattr(model, "get_params"):
        params = model.get_params()
        if "n_jobs" in params:
            model.set_params(n_jobs=1)

    return model


def _raw_row(df: pd.DataFrame, profile_id: str) -> dict:
    row = df.loc[df["profile_id"].astype(str) == str(profile_id)]

    if row.empty:
        raise ValueError(f"Profile {profile_id} was not found.")

    return row.iloc[0].to_dict()


def _class_result(probs: np.ndarray) -> dict:
    probs = np.asarray(probs, dtype=float).reshape(-1)

    if len(probs) != len(CLASS_NAMES):
        raise ValueError(
            f"Expected {len(CLASS_NAMES)} class probabilities, "
            f"received {len(probs)}."
        )

    if not np.isfinite(probs).all() or probs.sum() <= 0:
        raise ValueError("Model returned invalid class probabilities.")

    probs = probs / probs.sum()
    pred_idx = int(np.argmax(probs))

    return {
        "probs": probs,
        "pred_idx": pred_idx,
        "pred_class": CLASS_NAMES[pred_idx],
    }


def _predict_classical(model, raw: dict) -> dict:
    """Generate class probabilities using the existing fitted pipeline."""
    x = pipeline.encode(raw)

    if hasattr(x, "detach"):
        x = x.detach().cpu().numpy()

    x = np.asarray(x, dtype=np.float32)

    if x.ndim == 1:
        x = x.reshape(1, -1)

    if hasattr(model, "predict_proba"):
        raw_probs = np.asarray(model.predict_proba(x), dtype=float)[0]

        if hasattr(model, "classes_"):
            aligned = np.zeros(len(CLASS_NAMES), dtype=float)

            for col, cls in enumerate(model.classes_):
                if col >= len(raw_probs):
                    continue

                try:
                    idx = int(cls)
                    if 0 <= idx < len(CLASS_NAMES):
                        aligned[idx] = raw_probs[col]
                        continue
                except (ValueError, TypeError):
                    pass

                if str(cls) in CLASS_NAMES:
                    aligned[CLASS_NAMES.index(str(cls))] = raw_probs[col]

            raw_probs = aligned

        return _class_result(raw_probs)

    pred = model.predict(x)[0]

    try:
        idx = int(pred)
    except (TypeError, ValueError):
        idx = CLASS_NAMES.index(str(pred))

    probs = np.zeros(len(CLASS_NAMES), dtype=float)
    probs[idx] = 1.0

    return _class_result(probs)


def _predict_one(name: str, raw: dict) -> tuple[dict | None, str | None]:
    path = MODEL_FILES[name]

    if not path.exists():
        return None, f"Artifact not found: {path.name}"

    try:
        if name == "MLP":
            return predictor.predict(raw), None

        model = _load_classical_model(name, str(path))
        result = _predict_classical(model, raw)

        return result, None

    except Exception as exc:
        return None, (
            f"Could not evaluate {name}: "
            f"{type(exc).__name__}: {exc}"
        )


def _bar_chart(results: dict[str, dict]) -> None:
    rows = []

    for name, result in results.items():
        for idx, cls in enumerate(CLASS_NAMES):
            rows.append({
                "Model": name,
                "Risk": cls,
                "Probability": float(result["probs"][idx]),
            })

    chart = pd.DataFrame(rows)

    pivot = chart.pivot(
        index="Model",
        columns="Risk",
        values="Probability",
    )

    st.bar_chart(
        pivot,
        y=CLASS_NAMES,
        x_label="Model",
        y_label="Probability",
    )


def render() -> None:
    st.title("Model Comparison")

    st.caption(
        "Individual-client comparison using the trained model "
        "artifacts available in this project."
    )

    try:
        profiles = _load_profiles()

        if profiles.empty:
            st.error("No taxpayer profiles are available.")
            return

        ids = profiles["profile_id"].astype(str).tolist()

        default_id = (
            "TAXP04291" if "TAXP04291" in ids else ids[0]
        )

        profile_id = st.selectbox(
            "Select taxpayer",
            ids,
            index=ids.index(default_id),
        )

        raw = _raw_row(profiles, profile_id)

    except Exception as exc:
        st.error(f"Could not load taxpayer profiles: {exc}")
        return

    st.info(
        "This view compares model outputs for the same taxpayer. "
        "It is a model-level comparison, not a statement that one "
        "model is legally or operationally preferable."
    )

    results: dict[str, dict] = {}
    errors: dict[str, str] = {}

    with st.spinner("Evaluating available models..."):
        for name in MODEL_FILES:
            result, error = _predict_one(name, raw)

            if result is not None:
                results[name] = result
            else:
                errors[name] = error or "Model unavailable."

    if not results:
        st.error("No model artifact could be evaluated.")
        for name, error in errors.items():
            st.warning(f"{name}: {error}")
        return

    st.subheader(f"Predictions for {profile_id}")

    cols = st.columns(len(results))

    for col, (name, result) in zip(cols, results.items()):
        with col:
            st.metric(name, result["pred_class"])

            st.caption(
                f"Low {result['probs'][0]:.1%} · "
                f"Medium {result['probs'][1]:.1%} · "
                f"High {result['probs'][2]:.1%}"
            )

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

    st.dataframe(
        table.style.format({
            "Low": "{:.2%}",
            "Medium": "{:.2%}",
            "High": "{:.2%}",
        }),
        use_container_width=True,
    )

    predictions = [
        result["pred_class"] for result in results.values()
    ]

    if len(set(predictions)) == 1:
        st.success(
            f"All available models predict {predictions[0]} "
            "for this taxpayer."
        )
    else:
        st.warning(
            "The available models disagree on the predicted risk "
            "class. Review their probability distributions before "
            "interpreting the result."
        )

    if errors:
        with st.expander(
            "Model artifacts unavailable or evaluation errors",
            expanded=False,
        ):
            for name, error in errors.items():
                st.caption(f"**{name}:** {error}")

    st.subheader("Reference inputs")

    shown = [
        col
        for col in (
            datasets.NUMERIC_FEATURES
            + [datasets.CATEGORICAL_FEATURE]
        )
        if col in raw
    ]

    st.dataframe(
        pd.DataFrame([
            {col: raw.get(col) for col in shown}
        ]),
        use_container_width=True,
    )
