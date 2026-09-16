"""Taxpayer-level feature construction.

Only behavioural observables are used. The four target-derived score columns
identified in Phase 1 are dropped here and never re-enter the pipeline.

Note on transaction_frequency: Phase 1 established it varies *within* a profile
(4,821 of 4,946 profiles have more than one distinct value), so it is a
transaction-level attribute and is aggregated by mean rather than taken as a
per-taxpayer constant.
"""
from __future__ import annotations

import logging

import pandas as pd

logger = logging.getLogger(__name__)


def engineer_transaction_features(transactions: pd.DataFrame) -> pd.DataFrame:
    """Six additional taxpayer-level aggregates not present in the integrated file."""
    g = transactions.groupby("profile_id").agg(
        avg_anomaly_score=("anomaly_score", "mean"),
        avg_location_risk_score=("location_risk_score", "mean"),
        avg_device_risk_score=("device_risk_score", "mean"),
        avg_transaction_frequency=("transaction_frequency", "mean"),
        std_transaction_amount=("transaction_amount", "std"),
        max_transaction_amount=("transaction_amount", "max"),
    )
    # std is undefined for single-transaction profiles -> 0 dispersion, not missing.
    n_single = int(g.std_transaction_amount.isna().sum())
    g["std_transaction_amount"] = g["std_transaction_amount"].fillna(0.0)
    logger.info("engineered 6 features; std filled with 0.0 for %d single-txn profiles", n_single)
    return g.reset_index()


def build_modelling_frame(cfg: dict, data: dict) -> tuple[pd.DataFrame, dict]:
    """Assemble the leakage-free modelling table.

    Returns the frame plus a provenance dict describing what was dropped and why.
    """
    integrated = data["integrated"].copy()
    prov: dict = {"n_start": int(len(integrated))}

    # 1. Drop the 54 zero-transaction profiles (Phase 1 §8b).
    if cfg["drop_zero_transaction_profiles"]:
        mask = integrated["transaction_count"].isna()
        dropped = integrated.loc[mask, "profile_id"].tolist()
        prov["n_dropped_zero_transaction"] = len(dropped)
        prov["dropped_profile_ids"] = dropped
        integrated = integrated.loc[~mask].copy()

    # 2. Attach engineered features.
    eng = engineer_transaction_features(data["transactions"])
    before = len(integrated)
    integrated = integrated.merge(eng, on="profile_id", how="left", validate="one_to_one")
    if len(integrated) != before:
        raise ValueError("merge changed row count; join key is not unique")
    if integrated[eng.columns.drop("profile_id")].isna().any().any():
        raise ValueError("engineered features contain NaN after merge")

    # 3. Remove leakage columns.
    leak = [c for c in cfg["leakage_exclusions"] if c in integrated.columns]
    integrated = integrated.drop(columns=leak)
    prov["leakage_columns_dropped"] = leak

    # 4. Select final columns.
    f = cfg["features"]
    feature_cols = (
        f["numeric_tax"] + f["numeric_txn_existing"] + f["numeric_txn_engineered"] + f["categorical"]
    )
    target = cfg["target"]["name"]
    missing = [c for c in feature_cols + [target] if c not in integrated.columns]
    if missing:
        raise ValueError(f"expected columns absent from frame: {missing}")

    frame = integrated[["profile_id"] + feature_cols + [target]].copy()
    prov["n_final"] = int(len(frame))
    prov["n_features_pre_encoding"] = len(feature_cols)
    prov["feature_columns"] = feature_cols
    return frame, prov
