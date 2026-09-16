"""Read-only access to the project's data and result files.

Every function here reads an existing file. Nothing writes, refits, retrains or
recomputes a model artifact. Results are cached for the session so the same CSV
is not parsed on every rerun.
"""
from __future__ import annotations

import json

import pandas as pd
import streamlit as st

from . import paths

CLASSES = ["Low", "Medium", "High"]
RISK_ORDER = {c: i for i, c in enumerate(CLASSES)}

# The 19 numeric model inputs, in preprocessor output order.
NUMERIC_FEATURES = [
    "age", "annual_income", "tax_return_filed", "late_filing_count",
    "outstanding_dues", "previous_default_count", "income_growth_rate",
    "deduction_claim_ratio", "transaction_count", "total_transaction_value",
    "average_transaction_value", "transaction_anomaly_count",
    "transaction_anomaly_ratio", "avg_anomaly_score", "avg_location_risk_score",
    "avg_device_risk_score", "avg_transaction_frequency",
    "std_transaction_amount", "max_transaction_amount",
]
CATEGORICAL_FEATURE = "income_sources"

FEATURE_GROUPS = {
    "Taxpayer information": [
        "age", "annual_income", "income_sources", "income_growth_rate",
    ],
    "Compliance information": [
        "tax_return_filed", "late_filing_count", "outstanding_dues",
        "previous_default_count", "deduction_claim_ratio",
    ],
    "Transaction behaviour": [
        "transaction_count", "total_transaction_value", "average_transaction_value",
        "max_transaction_amount", "std_transaction_amount",
        "avg_transaction_frequency", "transaction_anomaly_count",
        "transaction_anomaly_ratio", "avg_anomaly_score",
        "avg_location_risk_score", "avg_device_risk_score",
    ],
}

FEATURE_LABELS = {
    "age": "Age",
    "annual_income": "Annual income",
    "income_sources": "Income sources",
    "income_growth_rate": "Income growth rate",
    "tax_return_filed": "Tax return filed",
    "late_filing_count": "Late filings",
    "outstanding_dues": "Outstanding dues",
    "previous_default_count": "Previous defaults",
    "deduction_claim_ratio": "Deduction claim ratio",
    "transaction_count": "Transaction count",
    "total_transaction_value": "Total transaction value",
    "average_transaction_value": "Average transaction value",
    "max_transaction_amount": "Largest transaction",
    "std_transaction_amount": "Transaction amount spread",
    "avg_transaction_frequency": "Average transaction frequency",
    "transaction_anomaly_count": "Anomalous transactions",
    "transaction_anomaly_ratio": "Anomaly ratio",
    "avg_anomaly_score": "Average anomaly score",
    "avg_location_risk_score": "Average location risk",
    "avg_device_risk_score": "Average device risk",
}


def label(feature: str) -> str:
    return FEATURE_LABELS.get(feature, feature.replace("_", " ").capitalize())


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def load_modelled_profiles() -> pd.DataFrame:
    """The 4,946 profiles that entered Phase 2, tagged with their split.

    Reassembled from the three persisted split files rather than rebuilt, so the
    rows are exactly the ones the pipeline used.
    """
    frames = []
    for split, path in (
        ("train", paths.TRAIN_RAW),
        ("validation", paths.VAL_RAW),
        ("test", paths.TEST_RAW),
    ):
        if not path.exists():
            continue
        df = pd.read_csv(path)
        df["split"] = split
        frames.append(df)
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


@st.cache_data(show_spinner=False)
def population_counts() -> dict:
    """Source population vs modelled population, with the exclusion made explicit."""
    modelled = load_modelled_profiles()
    out = {
        "modelled": int(len(modelled)),
        "source": None,
        "excluded": None,
        "transactions": None,
    }
    if paths.INTEGRATED_RAW.exists():
        source = pd.read_csv(paths.INTEGRATED_RAW, usecols=["profile_id", "transaction_count"])
        out["source"] = int(len(source))
        out["excluded"] = int(source["transaction_count"].isna().sum())
    if paths.TRANSACTIONS_RAW.exists():
        # Counting rows without materialising the whole frame.
        with open(paths.TRANSACTIONS_RAW, "rb") as fh:
            out["transactions"] = max(sum(1 for _ in fh) - 1, 0)
    return out


@st.cache_data(show_spinner=False)
def risk_distribution(scope: str = "modelled") -> pd.DataFrame:
    """Class counts with an explicit denominator.

    scope="modelled" uses the 4,946 profiles that entered Phase 2.
    scope="test" uses the 742-record held-out split.
    """
    df = load_modelled_profiles()
    if df.empty:
        return pd.DataFrame(columns=["risk", "count", "share"])
    if scope == "test":
        df = df[df["split"] == "test"]
    counts = df["overall_risk"].value_counts().reindex(CLASSES, fill_value=0)
    total = int(counts.sum())
    return pd.DataFrame({
        "risk": CLASSES,
        "count": counts.to_numpy(),
        "share": counts.to_numpy() / total if total else 0.0,
        "denominator": total,
    })


@st.cache_data(show_spinner=False)
def counterfactual_results() -> pd.DataFrame:
    if not paths.CF_RESULTS.exists():
        return pd.DataFrame()
    return pd.read_csv(paths.CF_RESULTS)


@st.cache_data(show_spinner=False)
def counterfactual_summary() -> pd.DataFrame:
    if not paths.CF_SUMMARY.exists():
        return pd.DataFrame()
    return pd.read_csv(paths.CF_SUMMARY)


@st.cache_data(show_spinner=False)
def negative_tests() -> pd.DataFrame:
    if not paths.NEGATIVE_TESTS.exists():
        return pd.DataFrame()
    return pd.read_csv(paths.NEGATIVE_TESTS)


@st.cache_data(show_spinner=False)
def phase5_metrics() -> dict:
    if not paths.PHASE5_METRICS.exists():
        return {}
    return json.loads(paths.PHASE5_METRICS.read_text())


@st.cache_data(show_spinner=False)
def reported_metrics() -> dict:
    if not paths.REPORTED_METRICS.exists():
        return {}
    return json.loads(paths.REPORTED_METRICS.read_text())


@st.cache_data(show_spinner=False)
def changed_feature_frequency(mode: str = "constrained") -> pd.DataFrame:
    """How often the optimizer moved each feature. Not feature importance."""
    df = counterfactual_results()
    if df.empty:
        return pd.DataFrame(columns=["feature", "cases"])
    subset = df[df["mode"] == mode]
    counts: dict[str, int] = {}
    for cell in subset["changed_features"].fillna(""):
        for feature in (f for f in str(cell).split("|") if f):
            counts[feature] = counts.get(feature, 0) + 1
    out = pd.DataFrame(
        sorted(counts.items(), key=lambda kv: kv[1], reverse=True),
        columns=["feature", "cases"],
    )
    out["total_cases"] = int(len(subset))
    return out


@st.cache_data(show_spinner=False)
def feature_ranges() -> pd.DataFrame:
    """Observed min/median/max per numeric feature across the modelled set.

    Used to pre-fill and sanity-check the assessment form. These are observed
    dataset ranges, not constraint bounds.
    """
    df = load_modelled_profiles()
    if df.empty:
        return pd.DataFrame()
    present = [f for f in NUMERIC_FEATURES if f in df.columns]
    stats = df[present].agg(["min", "median", "max"]).T
    stats.index.name = "feature"
    return stats.reset_index()


@st.cache_data(show_spinner=False)
def income_source_categories() -> list[str]:
    df = load_modelled_profiles()
    if df.empty or CATEGORICAL_FEATURE not in df.columns:
        return []
    return sorted(df[CATEGORICAL_FEATURE].dropna().unique().tolist())
