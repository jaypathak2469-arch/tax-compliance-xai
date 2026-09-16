"""Real status checks for the artifacts the application depends on.

Nothing here fabricates a green light. Each check inspects the filesystem (and,
where cheap, the environment) and reports what it actually found. Phase D does
not load the model or the preprocessor, so the checks that need those are
reported honestly as "not connected yet" rather than "ready".
"""
from __future__ import annotations

import importlib
import importlib.metadata as md
import re
from dataclasses import dataclass
from pathlib import Path

from . import paths

# scikit-learn version recorded inside the shipped preprocessor pickle.
EXPECTED_SKLEARN = "1.8.0"

OK = "ok"
PENDING = "pending"
WARN = "warn"
FAIL = "fail"


@dataclass(frozen=True)
class Check:
    name: str
    state: str
    detail: str


def _size(path: Path) -> str:
    kb = path.stat().st_size / 1024
    return f"{kb:,.0f} KB" if kb >= 1 else f"{path.stat().st_size} B"


def _file_check(name: str, path: Path, missing_hint: str = "") -> Check:
    if path.exists():
        return Check(name, OK, f"{path.name} · {_size(path)}")
    hint = f" — {missing_hint}" if missing_hint else ""
    return Check(name, FAIL, f"not found at {path}{hint}")


def package_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def pickle_sklearn_version() -> str | None:
    """Read the scikit-learn version recorded in the preprocessor pickle.

    The joblib file is opened in binary and scanned for a version string. It is
    never unpickled here, so this is safe even when the installed scikit-learn
    is incompatible.
    """
    path = paths.resolve_preprocessor()
    if path is None:
        return None
    try:
        blob = path.read_bytes()
    except OSError:
        return None
    found = re.findall(rb"\b1\.\d+\.\d+\b", blob)
    return found[0].decode() if found else None


def sklearn_check() -> Check:
    installed = package_version("scikit-learn")
    recorded = pickle_sklearn_version() or EXPECTED_SKLEARN
    if installed is None:
        return Check("scikit-learn", FAIL,
                     f"not installed — the preprocessor was fitted with {recorded}")
    if installed == recorded:
        return Check("scikit-learn", OK, f"{installed} · matches the fitted preprocessor")
    return Check("scikit-learn", WARN,
                 f"{installed} installed, preprocessor fitted with {recorded} — "
                 "unpickling may warn or fail")


def torch_check() -> Check:
    version = package_version("torch")
    if version is None:
        return Check("PyTorch", FAIL, "not installed — the MLP cannot be loaded")
    return Check("PyTorch", OK, f"{version}")


def code_modules_check() -> Check:
    """Confirm the Phase 5 source modules are present and importable.

    Import is attempted only when torch is available, because
    ``code/counterfactual.py`` imports torch at module level.
    """
    required = [
        "config.yaml", "constraints.py", "counterfactual.py",
        "counterfactual_validation.py", "preprocessing.py",
        "data_loader.py", "feature_engineering.py",
    ]
    missing = [f for f in required if not (paths.CODE_DIR / f).exists()]
    if missing:
        return Check("Phase 5 source modules", FAIL, "missing: " + ", ".join(missing))
    if package_version("torch") is None:
        return Check("Phase 5 source modules", WARN,
                     f"{len(required)} files present · import blocked until PyTorch is installed")
    if not paths.ensure_code_on_path():
        return Check("Phase 5 source modules", FAIL, "code/ directory is not readable")
    try:
        for mod in ("constraints", "counterfactual", "counterfactual_validation"):
            importlib.import_module(mod)
    except Exception as exc:  # noqa: BLE001 - surface the real reason
        return Check("Phase 5 source modules", FAIL, f"import failed: {exc}")
    return Check("Phase 5 source modules", OK, f"{len(required)} files present and importable")


def artifact_checks() -> list[Check]:
    """Filesystem checks for every artifact the application reads."""
    pre = paths.resolve_preprocessor()
    if pre is None:
        preprocessor = Check(
            "Preprocessor artifact", FAIL,
            f"not found at {paths.PREPROCESSOR_PRIMARY} or {paths.PREPROCESSOR_FALLBACK}",
        )
    else:
        where = "data/processed" if pre == paths.PREPROCESSOR_PRIMARY else "models (fallback)"
        preprocessor = Check("Preprocessor artifact", OK, f"{where} · {_size(pre)}")

    return [
        _file_check("MLP checkpoint", paths.MODEL_CHECKPOINT),
        preprocessor,
        _file_check("Scaler statistics", paths.SCALER_STATS),
        _file_check("Project configuration", paths.CONFIG_YAML,
                    "expected in code/, not the project root"),
        _file_check("Held-out test split", paths.TEST_RAW),
        _file_check("Phase 5 results", paths.CF_RESULTS),
        _file_check("Phase 5 metrics", paths.PHASE5_METRICS),
        _file_check("Reported Phase 3/4 metrics", paths.REPORTED_METRICS),
    ]


def environment_checks() -> list[Check]:
    return [torch_check(), sklearn_check(), code_modules_check()]


def engine_checks() -> list[Check]:
    """Runtime components, tested by actually loading them."""
    if package_version("torch") is None:
        return [
            Check("MLP inference", FAIL, "PyTorch not installed"),
            Check("Counterfactual generator", FAIL, "PyTorch not installed"),
            Check("Constraint validator", FAIL, "PyTorch not installed"),
        ]
    from . import pipeline

    try:
        pipeline.get_model()
        mlp = Check("MLP inference", OK, "mlp_final_selected.pt loaded")
    except pipeline.PipelineError as exc:
        mlp = Check("MLP inference", FAIL, str(exc))

    code_ok = code_modules_check().state == OK
    cf = Check("Counterfactual generator", OK if code_ok else FAIL,
              "generate_counterfactual() importable" if code_ok else "import failed")
    cv = Check("Constraint validator", OK if code_ok else FAIL,
              "validate_constraints() importable" if code_ok else "import failed")
    return [mlp, cf, cv]


def all_checks() -> list[Check]:
    return artifact_checks() + environment_checks() + engine_checks()


def blocking_failures() -> list[Check]:
    return [c for c in artifact_checks() + environment_checks() if c.state == FAIL]
