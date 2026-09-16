"""Live counterfactual generation.

Calls the existing code/counterfactual.py::generate_counterfactual() and
code/counterfactual_validation.py::validate_counterfactual() directly. No
second counterfactual algorithm is implemented here — this module only wires
the existing functions to a single profile and formats their output the same
way code/run_phase5.py does (L0/L1/L2, changed-feature list, before/after raw
values), so a live result and the precomputed Phase 5 rows are comparable.
"""
from __future__ import annotations

import time

from . import pipeline


def generate(x0, target: int = 0, constrained: bool = True) -> dict:
    """Run one live constrained counterfactual search from x0 toward `target`.

    Returns everything the Recourse page needs: the optimisation result, an
    independent validation pass, L0/L1/L2, changed raw features, and the
    before/after raw dicts.
    """
    torch = pipeline.require_torch()
    paths_ok = pipeline.paths.ensure_code_on_path()
    if not paths_ok:
        raise pipeline.PipelineError(f"code/ directory not found at {pipeline.paths.CODE_DIR}.")

    from counterfactual import generate_counterfactual, CLASSES
    from counterfactual_validation import validate_counterfactual

    model = pipeline.get_model()
    _, feature_to_idx = pipeline.get_feature_index()
    scaler_stats = pipeline.get_scaler_stats()
    num, _ = pipeline.numeric_and_categorical_columns()

    with torch.no_grad():
        p0 = torch.softmax(model(x0.unsqueeze(0)), dim=1)[0].numpy()

    start = time.perf_counter()
    result = generate_counterfactual(
        model, x0, feature_to_idx, scaler_stats, target=target, constrained=constrained
    )
    wall_time = time.perf_counter() - start

    cf = result["x"]
    check = validate_counterfactual(model, cf, x0, feature_to_idx, scaler_stats, target=target)

    r0 = pipeline.decode_raw(x0)
    rc = pipeline.decode_raw(cf)
    changed = [
        f for f in num
        if abs(rc[f] - r0[f]) > 1e-5 * max(1.0, abs(r0[f]))
    ]
    delta = cf - x0

    return {
        "status": result["status"],
        "predicted_class": CLASSES[result["pred"]],
        "success": result["pred"] == target,
        "target_class": CLASSES[target],
        "feasible": check["feasible"],
        "issues": check["issues"],
        "finite": check["finite"],
        "p0": p0,
        "probs": result["probs"],
        "l0": len(changed),
        "l1": float(torch.abs(delta).sum()),
        "l2": float(torch.linalg.vector_norm(delta)),
        "optimization_steps": result["steps"],
        "runtime_seconds": result["runtime"],
        "wall_time_seconds": wall_time,
        "final_objective": result["objective"],
        "changed_features": changed,
        "raw_before": r0,
        "raw_after": rc,
        "constrained": constrained,
    }
