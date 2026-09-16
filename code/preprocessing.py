"""Splitting and preprocessing.

All transformations are fitted on the training split only. The scaler statistics
are additionally exported in raw units because Phase 5 needs them to translate
constraints between raw and standardised feature space.
"""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logger = logging.getLogger(__name__)


def encode_target(y: pd.Series, classes: list[str]) -> tuple[np.ndarray, dict]:
    mapping = {c: i for i, c in enumerate(classes)}
    unknown = set(y.unique()) - set(mapping)
    if unknown:
        raise ValueError(f"unexpected target labels: {unknown}")
    return y.map(mapping).to_numpy(), mapping


def stratified_split(frame: pd.DataFrame, cfg: dict):
    """70/15/15 stratified split. Performed in two stages on the label."""
    seed = cfg["seed"]
    s = cfg["split"]
    target = cfg["target"]["name"]
    y = frame[target]

    train, hold = train_test_split(
        frame, train_size=s["train"], stratify=y, random_state=seed, shuffle=True
    )
    # Split the remaining 30% evenly into val and test.
    rel_val = s["val"] / (s["val"] + s["test"])
    val, test = train_test_split(
        hold, train_size=rel_val, stratify=hold[target], random_state=seed, shuffle=True
    )
    logger.info("split sizes train=%d val=%d test=%d", len(train), len(val), len(test))
    return train.reset_index(drop=True), val.reset_index(drop=True), test.reset_index(drop=True)


def build_preprocessor(cfg: dict) -> ColumnTransformer:
    f = cfg["features"]
    numeric = f["numeric_tax"] + f["numeric_txn_existing"] + f["numeric_txn_engineered"]
    categorical = f["categorical"]

    # No imputer: Phase 1 confirmed zero missing values in the modelled rows and
    # feature_engineering asserts the engineered block is complete. Adding one
    # would silently mask a future data fault.
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("scale", StandardScaler())]), numeric),
            (
                "cat",
                Pipeline(
                    [("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))]
                ),
                categorical,
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def output_feature_names(preprocessor: ColumnTransformer) -> list[str]:
    return list(preprocessor.get_feature_names_out())


def scaler_stats(preprocessor: ColumnTransformer, cfg: dict) -> pd.DataFrame:
    """Per-feature mean/scale in raw units — required by the Phase 5 constraint
    projection to move bounds between raw and standardised space."""
    f = cfg["features"]
    numeric = f["numeric_tax"] + f["numeric_txn_existing"] + f["numeric_txn_engineered"]
    scaler: StandardScaler = preprocessor.named_transformers_["num"].named_steps["scale"]
    return pd.DataFrame(
        {"feature": numeric, "mean": scaler.mean_, "scale": scaler.scale_}
    )


def class_weights(y: np.ndarray, classes: list[str]) -> dict:
    """Balanced class weights computed on the TRAINING split only.

    Used by Phase 3 (class_weight) and Phase 4 (weighted cross-entropy).
    """
    counts = np.bincount(y, minlength=len(classes))
    w = len(y) / (len(classes) * counts)
    return {
        "counts": {c: int(counts[i]) for i, c in enumerate(classes)},
        "weights": {c: float(w[i]) for i, c in enumerate(classes)},
        "weight_vector": [float(x) for x in w],
    }
