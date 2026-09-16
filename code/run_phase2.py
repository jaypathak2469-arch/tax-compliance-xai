"""PHASE 2 — data preparation.

Verifies integrity, builds the leakage-free modelling frame, splits it, fits the
preprocessor on training data only, and persists every artifact Phase 3+ needs.
"""
from __future__ import annotations

import json
import logging
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

import data_loader as dl
import data_validation as dv
import feature_engineering as fe
import preprocessing as pp

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
)
log = logging.getLogger("phase2")


def main() -> dict:
    cfg = dl.load_config()
    np.random.seed(cfg["seed"])
    root = dl.project_root()
    proc_dir = root / cfg["paths"]["processed_dir"]
    models_dir = root / cfg["paths"]["models_dir"]
    metrics_dir = root / cfg["paths"]["metrics_dir"]
    for d in (proc_dir, models_dir, metrics_dir):
        d.mkdir(parents=True, exist_ok=True)

    log.info("loading raw datasets")
    data = dl.load_raw(cfg)

    log.info("running integrity checks")
    integrity = dv.run_all(data)
    log.info(
        "leakage formula reproduces target score exactly on %d rows",
        integrity["leakage_formula"]["n_checked"],
    )

    log.info("building leakage-free modelling frame")
    frame, prov = fe.build_modelling_frame(cfg, data)
    log.info("frame %s | dropped %d zero-txn profiles", frame.shape,
             prov.get("n_dropped_zero_transaction", 0))

    # Guard: no excluded column may survive into the frame.
    surviving = [c for c in cfg["leakage_exclusions"] if c in frame.columns]
    if surviving:
        raise AssertionError(f"leakage columns present in modelling frame: {surviving}")

    frame.to_csv(proc_dir / "modelling_frame.csv", index=False)

    log.info("stratified 70/15/15 split")
    train, val, test = pp.stratified_split(frame, cfg)

    # Verify no profile_id appears in more than one split.
    ids = [set(s.profile_id) for s in (train, val, test)]
    if ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2]:
        raise AssertionError("profile_id overlap detected between splits")

    target = cfg["target"]["name"]
    classes = cfg["target"]["classes"]

    log.info("fitting preprocessor on TRAIN ONLY")
    preprocessor = pp.build_preprocessor(cfg)
    X_train = preprocessor.fit_transform(train)
    X_val = preprocessor.transform(val)
    X_test = preprocessor.transform(test)
    feat_names = pp.output_feature_names(preprocessor)

    y_train, mapping = pp.encode_target(train[target], classes)
    y_val, _ = pp.encode_target(val[target], classes)
    y_test, _ = pp.encode_target(test[target], classes)

    # Persist arrays and splits.
    np.savez_compressed(
        proc_dir / "splits.npz",
        X_train=X_train, X_val=X_val, X_test=X_test,
        y_train=y_train, y_val=y_val, y_test=y_test,
    )
    for name, df in (("train", train), ("val", val), ("test", test)):
        df.to_csv(proc_dir / f"{name}_raw.csv", index=False)

    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")
    stats = pp.scaler_stats(preprocessor, cfg)
    stats.to_csv(proc_dir / "scaler_stats.csv", index=False)

    meta = {
        "seed": cfg["seed"],
        "target": target,
        "class_mapping": mapping,
        "n_rows_source": prov["n_start"],
        "n_rows_dropped_zero_transaction": prov.get("n_dropped_zero_transaction", 0),
        "n_rows_modelled": prov["n_final"],
        "leakage_columns_dropped": prov["leakage_columns_dropped"],
        "features_pre_encoding": prov["feature_columns"],
        "n_features_pre_encoding": prov["n_features_pre_encoding"],
        "features_post_encoding": feat_names,
        "n_features_post_encoding": len(feat_names),
        "collinear_drop_for_linear": cfg["collinear_drop_for_linear"],
        "split_sizes": {"train": len(train), "val": len(val), "test": len(test)},
        "class_counts": {
            s: {c: int((df[target] == c).sum()) for c in classes}
            for s, df in (("train", train), ("val", val), ("test", test))
        },
        "integrity": integrity,
    }
    with open(metrics_dir / "phase2_metadata.json", "w") as fh:
        json.dump(meta, fh, indent=2)

    log.info("phase 2 artifacts written to %s", proc_dir)
    return meta


if __name__ == "__main__":
    m = main()
    print("\n" + json.dumps(
        {k: v for k, v in m.items() if k != "features_post_encoding"}, indent=2))
