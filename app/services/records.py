"""Canonical, session-scoped assessment records.

This module is the single source of truth for anything Reports, CSV export,
PDF export, and cross-page drill-down need. It never calls the model, the
preprocessor or the counterfactual generator itself — it only takes the
already-computed output of ``services/predictor.py`` and
``services/explainer.py`` and stores a JSON-safe copy of it (tensors and numpy
arrays are converted to plain floats/lists once, here, so nothing downstream
has to know about torch).

Two stores live in ``st.session_state``:

``assessment_history`` (this module's responsibility)
    A list of structured records, newest last. Each record holds the raw
    input, the prediction, and — once generated — the attached counterfactual
    result. This is the canonical store for Reports, exports and any
    drill-down view.

``history`` (owned by the existing views, kept for backward compatibility)
    The small flat list of display strings the Reports page already rendered
    before this module existed. ``record_assessment()`` appends the minimal
    compatibility row so nothing that already reads ``history`` breaks, but
    it derives that row from the same prediction result rather than
    recomputing anything.
"""
from __future__ import annotations

import datetime as dt
import uuid

import streamlit as st

STORE_KEY = "assessment_history"
LEGACY_KEY = "history"


def _new_id() -> str:
    return uuid.uuid4().hex[:10]


def store() -> list[dict]:
    """The canonical record list, created on first use."""
    return st.session_state.setdefault(STORE_KEY, [])


def _legacy_row(profile_id: str, pred_class: str, probs: list[float], when: dt.datetime) -> dict:
    """The exact shape app/views/reports.py already expected from `history`."""
    return {
        "Profile": profile_id,
        "Predicted risk": pred_class,
        "Low %": f"{probs[0] * 100:.2f}",
        "Medium %": f"{probs[1] * 100:.2f}",
        "High %": f"{probs[2] * 100:.2f}",
        "Assessed at": when.strftime("%H:%M:%S"),
    }


def record_assessment(
    *,
    profile_id: str,
    source: str,
    raw_input: dict,
    probs,
    pred_class: str,
    dataset_label: str | None = None,
    split: str | None = None,
) -> dict:
    """Create a new structured record from a live prediction result.

    Appends to both the canonical ``assessment_history`` and the legacy
    ``history`` list, and returns the new record (with its ``record_id``) so
    the caller can later attach a counterfactual result to it.
    """
    now = dt.datetime.now()
    probs_list = [float(p) for p in probs]
    record = {
        "record_id": _new_id(),
        "created_at": now.isoformat(timespec="seconds"),
        "updated_at": now.isoformat(timespec="seconds"),
        "profile_id": profile_id,
        "source": source,  # "manual" | "profile" | "session"
        "dataset_label": dataset_label,
        "split": split,
        "raw_input": dict(raw_input),
        "probs": probs_list,
        "pred_class": pred_class,
        "cf": None,
    }
    store().append(record)
    st.session_state.setdefault(LEGACY_KEY, []).append(
        _legacy_row(profile_id, pred_class, probs_list, now)
    )
    return record


def _serialize_cf(result: dict) -> dict:
    """Plain-Python copy of a services.explainer.generate() result.

    Converts numpy/torch scalars to floats/ints/lists once, here, so every
    consumer (CSV, PDF, the Reports table) works with JSON-safe data.
    """
    return {
        "status": str(result["status"]),
        "predicted_class": str(result["predicted_class"]),
        "success": bool(result["success"]),
        "target_class": str(result["target_class"]),
        "feasible": bool(result["feasible"]),
        "issues": list(result["issues"]),
        "finite": bool(result["finite"]),
        "p0": [float(x) for x in result["p0"]],
        "probs": [float(x) for x in result["probs"]],
        "l0": int(result["l0"]),
        "l1": float(result["l1"]),
        "l2": float(result["l2"]),
        "optimization_steps": int(result["optimization_steps"]),
        "runtime_seconds": float(result["runtime_seconds"]),
        "final_objective": float(result["final_objective"]),
        "changed_features": list(result["changed_features"]),
        "raw_before": {k: float(v) if not isinstance(v, str) else v
                       for k, v in result["raw_before"].items()},
        "raw_after": {k: float(v) if not isinstance(v, str) else v
                      for k, v in result["raw_after"].items()},
        "constrained": bool(result["constrained"]),
    }


def attach_counterfactual(record_id: str, cf_result: dict) -> dict | None:
    """Attach a live counterfactual result to an existing record, in place.

    Returns the updated record, or None if no record with that id exists in
    this session (e.g. the session was reset between the assessment and the
    counterfactual call).
    """
    for rec in store():
        if rec["record_id"] == record_id:
            rec["cf"] = _serialize_cf(cf_result)
            rec["updated_at"] = dt.datetime.now().isoformat(timespec="seconds")
            return rec
    return None


def get(record_id: str) -> dict | None:
    for rec in store():
        if rec["record_id"] == record_id:
            return rec
    return None


def latest() -> dict | None:
    records = store()
    return records[-1] if records else None


def all_records_newest_first() -> list[dict]:
    return list(reversed(store()))


def summary_rows() -> list[dict]:
    """One compact row per record, newest first, for the Reports table."""
    rows = []
    for rec in all_records_newest_first():
        cf = rec.get("cf")
        rows.append({
            "Record": rec["record_id"],
            "Profile": rec["profile_id"],
            "Source": rec["source"],
            "Predicted risk": rec["pred_class"],
            "Recourse generated": "Yes" if cf else "No",
            "Recourse outcome": (cf["status"] if cf else "—"),
            "Assessed at": rec["created_at"].replace("T", " "),
        })
    return rows
