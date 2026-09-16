from __future__ import annotations
import json, random, time
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import joblib
from counterfactual import load_model, CLASSES, generate_counterfactual
from constraints import decode_income_source
from counterfactual_validation import validate_counterfactual, run_negative_tests

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
MODELS = ROOT / "models"
RESULTS = ROOT / "results"
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)

pre = joblib.load(DATA / "processed/preprocessor.joblib")
scaler_stats = pd.read_csv(DATA / "processed/scaler_stats.csv").set_index("feature")
test = pd.read_csv(DATA / "processed/test_raw.csv")
feature_names = list(pre.get_feature_names_out())
feature_to_idx = {f:i for i,f in enumerate(feature_names)}
model = load_model(MODELS / "mlp_final_selected.pt", len(feature_names))

num = list(scaler_stats.index)
cat = ["income_sources"]

def encode(row):
    return torch.tensor(pre.transform(pd.DataFrame([row])[num+cat]).astype(np.float32)[0])

def decode_raw(z):
    out = {}
    for f in num:
        j = feature_to_idx[f]
        out[f] = float(z[j].detach().cpu().item() * scaler_stats.loc[f,"scale"] + scaler_stats.loc[f,"mean"])
    out["income_sources"] = decode_income_source(z, feature_to_idx)
    return out

subset = test[test["overall_risk"].isin(["Medium", "High"])].reset_index(drop=True)
rows=[]
for _, row in subset.iterrows():
    x0 = encode(row)
    with torch.no_grad():
        p0 = torch.softmax(model(x0.unsqueeze(0)), dim=1)[0].numpy()
    r0 = decode_raw(x0)
    for mode, constrained in [("unconstrained", False), ("constrained", True)]:
        result = generate_counterfactual(model, x0, feature_to_idx, scaler_stats,
                                         target=0, constrained=constrained)
        cf = result["x"]
        check = validate_counterfactual(model, cf, x0, feature_to_idx, scaler_stats, target=0)
        valid = check["feasible"]
        issues = check["issues"]
        rc = decode_raw(cf)
        changed = []
        for f in num:
            if abs(rc[f] - r0[f]) > 1e-5 * max(1.0, abs(r0[f])):
                changed.append(f)
        delta = cf - x0
        rows.append({
            "profile_id": row["profile_id"], "source_risk": row["overall_risk"],
            "mode": mode, "target": "Low", "status": result["status"],
            "predicted_class": CLASSES[result["pred"]], "success": result["pred"] == 0,
            "valid": valid, "constraint_feasible": check["feasible"], "violations": "|".join(issues),
            "p0_low": float(p0[0]), "p0_medium": float(p0[1]), "p0_high": float(p0[2]),
            "p_low": float(result["probs"][0]), "p_medium": float(result["probs"][1]), "p_high": float(result["probs"][2]),
            "l0": len(changed), "l1": float(torch.abs(delta).sum()),
            "l2": float(torch.linalg.vector_norm(delta)), "optimization_steps": result["steps"],
            "runtime_seconds": result["runtime"], "final_objective": result["objective"],
            "changed_features": "|".join(changed),
            **{f"cf_{f}": rc[f] for f in num}
        })

res = pd.DataFrame(rows)
metrics_dir = RESULTS / "metrics"; figures_dir = RESULTS / "figures"
metrics_dir.mkdir(parents=True, exist_ok=True); figures_dir.mkdir(parents=True, exist_ok=True)
res.to_csv(metrics_dir / "counterfactual_results.csv", index=False)

summary=[]
for (mode, source), g in res.groupby(["mode","source_risk"], dropna=False):
    summary.append({"mode":mode,"source_risk":source,"n":len(g),
                    "success_rate":float(g.success.mean()),"valid_rate":float(g.valid.mean()),
                    "mean_l0":float(g.l0.mean()),"mean_l1":float(g.l1.mean()),"mean_l2":float(g.l2.mean()),
                    "mean_steps":float(g.optimization_steps.mean()),"mean_runtime_seconds":float(g.runtime_seconds.mean()),
                    "mean_final_objective":float(g.final_objective.mean())})
summary.append({"mode":"overall","source_risk":"Medium+High","n":len(res),
                "success_rate":float(res.success.mean()),"valid_rate":float(res.valid.mean()),
                "mean_l0":float(res.l0.mean()),"mean_l1":float(res.l1.mean()),"mean_l2":float(res.l2.mean()),
                "mean_steps":float(res.optimization_steps.mean()),"mean_runtime_seconds":float(res.runtime_seconds.mean()),
                "mean_final_objective":float(res.final_objective.mean())})
sumdf=pd.DataFrame(summary)
sumdf.to_csv(metrics_dir / "counterfactual_summary.csv", index=False)

neg=[]
for _, row in subset.head(10).iterrows():
    x0=encode(row)
    for name,rejected,issues in run_negative_tests(x0,feature_to_idx,scaler_stats):
        neg.append({"profile_id":row["profile_id"],"test":name,"rejected":bool(rejected),"issues":"|".join(issues)})
negdf=pd.DataFrame(neg); negdf.to_csv(metrics_dir / "negative_validation_tests.csv", index=False)

# Representative successful constrained counterfactuals.
rep = res[(res["mode"]=="constrained") & (res.success) & (res.valid)].copy().head(10)
rep.to_csv(metrics_dir / "representative_counterfactuals.csv", index=False)

phase5 = {
    "phase": 5,
    "model": "mlp_final_selected.pt",
    "model_source": "existing Phase 4 checkpoint; no retraining in Phase 5",
    "target": "Low",
    "n_test_medium_high": int(len(subset)),
    "source_counts": subset["overall_risk"].value_counts().to_dict(),
    "success_rates": summary,
    "constraint_violation_rate_constrained": float(1.0 - res.loc[res["mode"]=="constrained","valid"].mean()),
    "negative_tests_all_rejected": bool(negdf.rejected.all()),
    "objective": "target cross-entropy + 0.01 L1 + 0.01 L2",
    "optimization_space": "standardized feature space",
    "constraints": {
        "immutable": ["age","previous_default_count","transaction_count"],
        "monotonic_decrease_only": ["outstanding_dues","late_filing_count","transaction_anomaly_ratio"],
        "bounded": {"deduction_claim_ratio":[0.0,1.0],"transaction_anomaly_ratio":[0.0,1.0]},
        "categorical": "income_sources unchanged",
        "derived": "transaction_anomaly_count = transaction_anomaly_ratio * transaction_count"
    },
    "limitation": "Technical recourse results on synthetic data and a model trained on a synthetic generating process; not evidence of real-world regulatory or compliance effectiveness."
}
(metrics_dir/"phase5_metrics.json").write_text(json.dumps(phase5, indent=2))
print(sumdf.to_string(index=False))
print("Negative tests all rejected:", bool(negdf.rejected.all()))
print("Output:", metrics_dir)
