"""Local audit logging for TAX-XAI.

The audit log is local to the Streamlit application. It records observable
application events without collecting passwords, secrets, or unnecessary
personal information.

Events are written to results/audit/audit_log.csv and mirrored in the current
Streamlit session for immediate display.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import streamlit as st

from services import paths


COLUMNS = [
    "timestamp_utc",
    "event",
    "profile_id",
    "source",
    "model",
    "prediction",
    "details",
]


def _audit_path() -> Path:
    root = Path(paths.PROJECT_ROOT)
    folder = root / "results" / "audit"
    folder.mkdir(parents=True, exist_ok=True)
    return folder / "audit_log.csv"


def _session_rows() -> list[dict[str, str]]:
    if "audit_log" not in st.session_state:
        st.session_state["audit_log"] = []
    return st.session_state["audit_log"]


def _safe_text(value: Any, limit: int = 500) -> str:
    if value is None:
        return ""
    text = str(value).replace("\n", " ").replace("\r", " ")
    return text[:limit]


def _write_row(row: dict[str, str]) -> None:
    path = _audit_path()
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=COLUMNS)
        if not exists:
            writer.writeheader()
        writer.writerow({key: row.get(key, "") for key in COLUMNS})


def log_event(
    event: str,
    *,
    profile_id: str | None = None,
    source: str | None = None,
    model: str | None = None,
    prediction: str | None = None,
    details: str | dict | None = None,
) -> None:
    """Record one application event."""
    if isinstance(details, dict):
        details = json.dumps(details, ensure_ascii=False, sort_keys=True)
    row = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "event": _safe_text(event, 100),
        "profile_id": _safe_text(profile_id, 100),
        "source": _safe_text(source, 100),
        "model": _safe_text(model, 100),
        "prediction": _safe_text(prediction, 100),
        "details": _safe_text(details, 1000),
    }
    _session_rows().append(row)
    _write_row(row)


def _fingerprint(value: Any) -> str:
    try:
        raw = json.dumps(value, sort_keys=True, default=str).encode("utf-8")
    except Exception:
        raw = repr(value).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()[:16]


def capture_observable_events() -> None:
    """Capture newly-created assessment/counterfactual state once per session.

    This keeps integration lightweight: app.py calls this on every rerun.
    """
    latest = st.session_state.get("latest_assessment")
    if latest:
        fp = _fingerprint(
            {
                "profile_id": latest.get("profile_id"),
                "pred_class": latest.get("pred_class"),
                "timestamp": latest.get("timestamp"),
            }
        )
        seen = st.session_state.setdefault("_audit_seen_assessments", set())
        if fp not in seen:
            seen.add(fp)
            log_event(
                "Risk assessment",
                profile_id=latest.get("profile_id"),
                source=latest.get("source", "assessment"),
                model=latest.get("model", "MLP"),
                prediction=latest.get("pred_class"),
                details={"record_id": latest.get("record_id", "")},
            )

    latest_cf = st.session_state.get("latest_cf")
    if latest_cf:
        fp = _fingerprint(
            {
                "profile_id": latest_cf.get("profile_id"),
                "status": latest_cf.get("status"),
                "pred_class": latest_cf.get("pred_class"),
                "source_risk": latest_cf.get("source_risk"),
                "raw_after": latest_cf.get("raw_after"),
            }
        )
        seen = st.session_state.setdefault("_audit_seen_counterfactuals", set())
        if fp not in seen:
            seen.add(fp)
            log_event(
                "Counterfactual generated",
                profile_id=latest_cf.get("profile_id"),
                source="recourse",
                model=latest_cf.get("model", "MLP"),
                prediction=latest_cf.get("pred_class"),
                details={
                    "status": latest_cf.get("status"),
                    "valid": latest_cf.get("valid"),
                    "constraint_feasible": latest_cf.get("constraint_feasible"),
                },
            )


def read_log() -> list[dict[str, str]]:
    path = _audit_path()
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def clear_current_session() -> None:
    st.session_state["audit_log"] = []
    st.session_state["_audit_seen_assessments"] = set()
    st.session_state["_audit_seen_counterfactuals"] = set()
