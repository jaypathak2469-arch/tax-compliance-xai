"""Batch CSV assessment page for TAX-XAI.

Supports:
1. Model-ready CSVs containing the 20 raw model inputs.
2. Existing taxpayer-profile CSVs containing profile_id and tax fields. For
   those files, the persisted Phase 2 modelled profiles are used to supply
   the transaction-behaviour aggregates already used by the live model.

No model is trained or refitted and no project artifact is modified.
"""

from __future__ import annotations

import pandas as pd
import streamlit as st

from services import datasets, predictor, validation


REQUIRED_COLUMNS = datasets.NUMERIC_FEATURES + [datasets.CATEGORICAL_FEATURE]
PROFILE_ID = "profile_id"


def _template_csv() -> bytes:
    defaults = {
        "age": 35,
        "annual_income": 800000,
        "tax_return_filed": 1,
        "late_filing_count": 0,
        "outstanding_dues": 0,
        "previous_default_count": 0,
        "income_growth_rate": 0.08,
        "deduction_claim_ratio": 0.20,
        "transaction_count": 20,
        "total_transaction_value": 250000,
        "average_transaction_value": 12500,
        "transaction_anomaly_count": 1,
        "transaction_anomaly_ratio": 0.05,
        "avg_anomaly_score": 0.10,
        "avg_location_risk_score": 0.10,
        "avg_device_risk_score": 0.10,
        "avg_transaction_frequency": 20,
        "std_transaction_amount": 5000,
        "max_transaction_amount": 40000,
        "income_sources": "Salary",
    }
    return (
        pd.DataFrame([defaults], columns=REQUIRED_COLUMNS)
        .to_csv(index=False)
        .encode("utf-8")
    )


def _normalise_value(value):
    if pd.isna(value):
        return None
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    return value


def _validate_columns(df: pd.DataFrame) -> tuple[list[str], list[str]]:
    required = set(REQUIRED_COLUMNS)
    actual = set(df.columns)
    missing = [c for c in REQUIRED_COLUMNS if c not in actual]
    extra = [c for c in df.columns if c not in required and c != PROFILE_ID]
    return missing, extra


def _is_profile_csv(df: pd.DataFrame) -> bool:
    """Detect the existing taxpayer-profile format."""
    return PROFILE_ID in df.columns and datasets.CATEGORICAL_FEATURE in df.columns


def _prepare_profile_csv(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Enrich an existing taxpayer CSV with persisted modelled aggregates.

    Returns:
      prepared: rows that can be validated/predicted
      excluded: rows whose profile_id was not part of the persisted modelled
                population (for example the 54 zero-transaction exclusions).
    """
    modelled = datasets.load_modelled_profiles()
    if modelled.empty or PROFILE_ID not in modelled.columns:
        excluded = df.copy()
        excluded["_batch_issue"] = "Persisted modelled profile data is unavailable."
        return pd.DataFrame(), excluded

    aggregate_columns = [
        c for c in REQUIRED_COLUMNS
        if c in modelled.columns and c not in df.columns
    ]

    lookup_cols = [PROFILE_ID] + aggregate_columns
    lookup = modelled[lookup_cols].drop_duplicates(PROFILE_ID)

    prepared = df.merge(
        lookup,
        on=PROFILE_ID,
        how="left",
        suffixes=("", "__persisted"),
        indicator="_model_match",
    )

    # Only use persisted values for features missing from the upload.
    # Uploaded values are preserved and therefore remain explicit user input.
    missing_model_features = [
        c for c in REQUIRED_COLUMNS if c not in df.columns
    ]
    excluded = prepared[prepared["_model_match"] != "both"].copy()

    if missing_model_features:
        excluded_missing = prepared[
            prepared["_model_match"].eq("both")
            & prepared[missing_model_features].isna().any(axis=1)
        ].copy()
        excluded = pd.concat([excluded, excluded_missing], ignore_index=True)

    valid_match = prepared["_model_match"].eq("both")
    if missing_model_features:
        valid_match &= ~prepared[missing_model_features].isna().any(axis=1)

    prepared = prepared[valid_match].copy()

    excluded["_batch_issue"] = excluded.apply(
        lambda row: (
            "profile_id was not found in the persisted modelled population."
            if row.get("_model_match") != "both"
            else "One or more model features could not be supplied from the persisted profile."
        ),
        axis=1,
    )

    return prepared, excluded


def _assess_dataframe(
    df: pd.DataFrame,
    source_rows: list[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    valid_rows = []
    invalid_rows = []

    if source_rows is None:
        source_rows = [int(i) + 2 for i in range(len(df))]

    for position, (_, row) in enumerate(df.iterrows()):
        raw = {
            feature: _normalise_value(row[feature])
            for feature in REQUIRED_COLUMNS
        }
        issues = validation.validate_input(raw)

        base = {
            "source_row": source_rows[position],
            "profile_id": row.get(PROFILE_ID, ""),
        }

        if issues:
            invalid_rows.append({
                **base,
                "status": "Invalid",
                "issues": " | ".join(issues),
            })
            continue

        try:
            result = predictor.predict(raw)
            probs = result["probs"]
            valid_rows.append({
                **base,
                "status": "Valid",
                "predicted_risk": result["pred_class"],
                "low_probability": float(probs[0]),
                "medium_probability": float(probs[1]),
                "high_probability": float(probs[2]),
            })
        except Exception as exc:
            invalid_rows.append({
                **base,
                "status": "Prediction failed",
                "issues": str(exc),
            })

    return pd.DataFrame(valid_rows), pd.DataFrame(invalid_rows)


def _risk_counts(results: pd.DataFrame) -> None:
    counts = results["predicted_risk"].value_counts().reindex(
        datasets.CLASSES, fill_value=0
    )
    cols = st.columns(3)
    for col, risk in zip(cols, datasets.CLASSES):
        with col:
            st.metric(risk, int(counts[risk]))


def render() -> None:
    st.markdown("## Batch assessment")
    st.caption(
        "Assess multiple taxpayers with the existing fitted preprocessor and "
        "trained MLP. Upload either a model-ready CSV or an existing taxpayer "
        "profile CSV."
    )

    st.download_button(
        "Download model-ready CSV template",
        data=_template_csv(),
        file_name="tax_xai_batch_template.csv",
        mime="text/csv",
        help="Contains the exact 20 raw input columns expected by the live model.",
    )

    uploaded = st.file_uploader(
        "Upload taxpayer CSV",
        type=["csv"],
        help="You can upload a model-ready CSV or a taxpayer profile CSV containing profile_id.",
    )

    if uploaded is None:
        st.info("Upload a CSV to begin batch assessment.")
        return

    # Clear results when a different uploaded file is selected.
    file_token = f"{uploaded.name}:{getattr(uploaded, 'size', 0)}"
    if st.session_state.get("batch_file_token") != file_token:
        st.session_state.pop("batch_results", None)
        st.session_state.pop("batch_invalid", None)
        st.session_state["batch_file_token"] = file_token

    try:
        df = pd.read_csv(uploaded)
    except Exception as exc:
        st.error(f"Could not read the CSV: {exc}")
        return

    if df.empty:
        st.warning("The uploaded CSV contains no data rows.")
        return

    if len(df) > 5000:
        st.warning(
            "This demo limits one upload to 5,000 rows. "
            "Please split a larger file into smaller batches."
        )
        return

    profile_mode = _is_profile_csv(df)
    missing, extra = _validate_columns(df)

    st.markdown("### 1. File validation")

    if not missing:
        st.success(
            "Model-ready CSV detected. All 20 required model input columns are present."
        )
        prepared = df.copy()
        excluded = pd.DataFrame()
        source_rows = [int(i) + 2 for i in df.index]
        mode_label = "Model-ready CSV"
    elif profile_mode:
        st.info(
            "Taxpayer profile CSV detected. The app will use the persisted "
            "modelled profiles to supply the transaction-behaviour features "
            "required by the live model."
        )
        prepared, excluded = _prepare_profile_csv(df)
        source_rows = [
            int(i) + 2 for i in prepared.index
        ]
        mode_label = "Existing taxpayer profile CSV"
    else:
        st.error(
            "This CSV is neither model-ready nor a recognised taxpayer-profile "
            "CSV. Download the template or provide profile_id plus income_sources "
            "and the taxpayer fields."
        )
        st.markdown("#### Missing model columns")
        st.write(", ".join(missing))
        if extra:
            st.info(f"Extra columns will be ignored: {', '.join(extra)}")
        return

    if extra:
        st.info(f"Extra columns will be ignored: {', '.join(extra)}")

    st.caption(f"Input type: **{mode_label}** · {len(df):,} uploaded rows")

    if profile_mode and not excluded.empty:
        st.warning(
            f"{len(excluded):,} uploaded row(s) could not be matched to the "
            "persisted modelled population and will not be predicted."
        )

    if prepared.empty:
        st.error("No rows are available for prediction after file preparation.")
        if not excluded.empty:
            st.dataframe(
                excluded[[PROFILE_ID, "_batch_issue"]]
                if PROFILE_ID in excluded.columns
                else excluded[["_batch_issue"]],
                use_container_width=True,
                hide_index=True,
            )
        return

    preview_columns = [
        c for c in REQUIRED_COLUMNS if c in prepared.columns
    ]
    st.markdown("#### Prepared input preview")
    st.dataframe(
        prepared[preview_columns].head(10),
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("### 2. Row validation and prediction")

    if st.button(
        "Run batch assessment",
        type="primary",
        use_container_width=True,
    ):
        with st.spinner(
            f"Validating and assessing {len(prepared):,} rows with the live model..."
        ):
            results, invalid = _assess_dataframe(prepared, source_rows)

        # Add unmatched/excluded rows to the invalid report.
        if not excluded.empty:
            excluded_report = pd.DataFrame({
                "source_row": [
                    int(i) + 2 for i in excluded.index
                ],
                "profile_id": (
                    excluded[PROFILE_ID].tolist()
                    if PROFILE_ID in excluded.columns
                    else [""] * len(excluded)
                ),
                "status": "Excluded",
                "issues": excluded["_batch_issue"].tolist(),
            })
            invalid = pd.concat([invalid, excluded_report], ignore_index=True)

        st.session_state["batch_results"] = results
        st.session_state["batch_invalid"] = invalid
        st.session_state["batch_source_name"] = uploaded.name

    results = st.session_state.get("batch_results")
    invalid = st.session_state.get("batch_invalid")

    if results is None:
        st.caption("No batch assessment has been run for this upload yet.")
        return

    st.markdown("### 3. Assessment summary")

    summary_cols = st.columns(4)
    with summary_cols[0]:
        st.metric("Uploaded rows", len(df))
    with summary_cols[1]:
        st.metric("Predicted", len(results))
    with summary_cols[2]:
        st.metric("Invalid / excluded", len(invalid))
    with summary_cols[3]:
        coverage = len(results) / len(df) if len(df) else 0.0
        st.metric("Prediction coverage", f"{coverage:.1%}")

    if not results.empty:
        _risk_counts(results)

        st.markdown("### 4. Batch results")
        st.dataframe(
            results[
                [
                    "source_row",
                    "profile_id",
                    "predicted_risk",
                    "low_probability",
                    "medium_probability",
                    "high_probability",
                ]
            ],
            use_container_width=True,
            hide_index=True,
        )

        csv_bytes = results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download prediction results",
            data=csv_bytes,
            file_name="tax_xai_batch_results.csv",
            mime="text/csv",
            use_container_width=True,
        )

    if not invalid.empty:
        st.markdown("### 5. Validation / exclusion issues")
        st.dataframe(
            invalid[
                ["source_row", "profile_id", "status", "issues"]
            ],
            use_container_width=True,
            hide_index=True,
        )

        invalid_bytes = invalid.to_csv(index=False).encode("utf-8")
        st.download_button(
            "Download validation report",
            data=invalid_bytes,
            file_name="tax_xai_batch_validation_report.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.caption(
        "Interpretation: predictions are model outputs for the uploaded records. "
        "They are not tax, legal, or regulatory determinations."
    )
