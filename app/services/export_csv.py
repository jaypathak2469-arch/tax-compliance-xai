"""Per-assessment CSV export.

Flattens one structured record from ``services/records.py`` into a single CSV
with clearly labelled sections. Nothing here computes anything new — every
value is already present on the record (itself a JSON-safe copy of what
``predictor.predict()`` and ``explainer.generate()`` returned live).
"""
from __future__ import annotations

import csv
import io

from . import datasets, validation


def _write_section(writer: csv.writer, title: str) -> None:
    writer.writerow([])
    writer.writerow([title])


def record_to_csv(record: dict) -> str:
    """Return the CSV text for one assessment record."""
    buf = io.StringIO()
    writer = csv.writer(buf)

    writer.writerow(["TAX-XAI — per-assessment export"])
    writer.writerow(["Record ID", record["record_id"]])
    writer.writerow(["Profile ID", record["profile_id"]])
    writer.writerow(["Source", record["source"]])
    if record.get("dataset_label"):
        writer.writerow(["Dataset label (overall_risk)", record["dataset_label"]])
    if record.get("split"):
        writer.writerow(["Dataset split", record["split"]])
    writer.writerow(["Assessed at", record["created_at"]])
    writer.writerow(["Last updated", record["updated_at"]])

    _write_section(writer, "Prediction (live model output)")
    writer.writerow(["Predicted class", record["pred_class"]])
    writer.writerow(["P(Low)", f"{record['probs'][0]:.6f}"])
    writer.writerow(["P(Medium)", f"{record['probs'][1]:.6f}"])
    writer.writerow(["P(High)", f"{record['probs'][2]:.6f}"])

    _write_section(writer, "Input values")
    writer.writerow(["Feature", "Label", "Value"])
    for feature, value in record["raw_input"].items():
        writer.writerow([feature, datasets.label(feature), value])

    cf = record.get("cf")
    if cf:
        _write_section(writer, "Counterfactual recourse (live model output)")
        writer.writerow(["Target class", cf["target_class"]])
        writer.writerow(["Result class", cf["predicted_class"]])
        writer.writerow(["Status", cf["status"]])
        writer.writerow(["Success (reached target class)", cf["success"]])
        writer.writerow(["Constrained search", cf["constrained"]])
        writer.writerow(["L0 (features changed)", cf["l0"]])
        writer.writerow(["L1 distance", f"{cf['l1']:.6f}"])
        writer.writerow(["L2 distance", f"{cf['l2']:.6f}"])
        writer.writerow(["Optimisation steps", cf["optimization_steps"]])
        writer.writerow(["Runtime (seconds)", f"{cf['runtime_seconds']:.6f}"])
        writer.writerow(["Final objective", f"{cf['final_objective']:.6f}"])
        writer.writerow(["P(Low) before", f"{cf['p0'][0]:.6f}"])
        writer.writerow(["P(Medium) before", f"{cf['p0'][1]:.6f}"])
        writer.writerow(["P(High) before", f"{cf['p0'][2]:.6f}"])
        writer.writerow(["P(Low) after", f"{cf['probs'][0]:.6f}"])
        writer.writerow(["P(Medium) after", f"{cf['probs'][1]:.6f}"])
        writer.writerow(["P(High) after", f"{cf['probs'][2]:.6f}"])

        _write_section(writer, "Counterfactual feature changes")
        writer.writerow(["Feature", "Label", "Original", "Counterfactual", "Change"])
        for feature in cf["changed_features"]:
            before = cf["raw_before"][feature]
            after = cf["raw_after"][feature]
            writer.writerow([feature, datasets.label(feature), before, after, after - before])

        _write_section(writer, "Constraint validation")
        checks = validation.constraint_report(cf["feasible"], cf["issues"], cf["finite"])
        writer.writerow(["Check", "State", "Detail"])
        for check in checks:
            writer.writerow([check.name, check.state, check.detail])
        writer.writerow(["Overall", "VALID" if validation.overall_valid(checks) else "INVALID", ""])

    return buf.getvalue()
