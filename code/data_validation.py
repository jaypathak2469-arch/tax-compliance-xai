"""Re-runs the Phase 1 integrity checks as executable assertions.

This exists so the audit conclusions are reproducible rather than narrative.
Any failure here means an upstream assumption of the pipeline has broken.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Phase 1: overall_risk_score = round(0.55*(1-tax_compliance_score)
#                                   + 0.45*financial_risk_score, 3)
LEAK_W_TAX = 0.55
LEAK_W_FIN = 0.45


def verify_leakage_formula(integrated: pd.DataFrame) -> dict:
    d = integrated.dropna(subset=["overall_risk_score"])
    recon = np.round(
        LEAK_W_TAX * (1 - d["tax_compliance_score"]) + LEAK_W_FIN * d["financial_risk_score"], 3
    )
    resid = (d["overall_risk_score"] - recon).abs()
    return {
        "n_checked": int(len(d)),
        "exact_match_rate": float(np.isclose(d["overall_risk_score"], recon, atol=1e-9).mean()),
        "max_residual": float(resid.max()),
    }


def verify_aggregates(transactions: pd.DataFrame, integrated: pd.DataFrame) -> dict:
    g = transactions.groupby("profile_id").agg(
        transaction_count=("transaction_id", "count"),
        total_transaction_value=("transaction_amount", "sum"),
        average_transaction_value=("transaction_amount", "mean"),
        transaction_anomaly_count=("is_anomaly", "sum"),
    )
    g["transaction_anomaly_ratio"] = g.transaction_anomaly_count / g.transaction_count
    m = integrated.set_index("profile_id").join(g, rsuffix="_calc")
    m = m.dropna(subset=["transaction_count"])
    out = {}
    for c in g.columns:
        out[c] = float((m[c] - m[f"{c}_calc"]).abs().max())
    return out


def verify_join(tax: pd.DataFrame, transactions: pd.DataFrame, integrated: pd.DataFrame) -> dict:
    tax_ids, int_ids = set(tax.profile_id), set(integrated.profile_id)
    txn_ids = set(transactions.profile_id)
    return {
        "tax_ids_unique": bool(tax.profile_id.is_unique),
        "integrated_ids_unique": bool(integrated.profile_id.is_unique),
        "tax_equals_integrated_ids": tax_ids == int_ids,
        "txn_ids_subset_of_tax": txn_ids.issubset(tax_ids),
        "n_profiles_without_transactions": len(tax_ids - txn_ids),
    }


def run_all(data: dict) -> dict:
    """Run every check; raise on the ones that would invalidate the pipeline."""
    report = {
        "join": verify_join(data["tax"], data["transactions"], data["integrated"]),
        "aggregates": verify_aggregates(data["transactions"], data["integrated"]),
        "leakage_formula": verify_leakage_formula(data["integrated"]),
    }

    j = report["join"]
    if not j["tax_equals_integrated_ids"]:
        raise ValueError("tax and integrated profile_id sets diverged")
    if not j["txn_ids_subset_of_tax"]:
        raise ValueError("transactions reference unknown profile_id values")

    # average_transaction_value is stored rounded to 2dp; allow that tolerance only.
    tolerances = {
        "transaction_count": 1e-9,
        "total_transaction_value": 1e-6,
        "average_transaction_value": 0.005 + 1e-9,
        "transaction_anomaly_count": 1e-9,
        "transaction_anomaly_ratio": 1e-9,
    }
    for col, tol in tolerances.items():
        got = report["aggregates"][col]
        if got > tol:
            raise ValueError(f"aggregate mismatch for {col}: max diff {got} > {tol}")

    if report["leakage_formula"]["exact_match_rate"] < 1.0:
        raise ValueError("leakage formula no longer reproduces overall_risk_score exactly")

    logger.info("all integrity checks passed")
    return report
