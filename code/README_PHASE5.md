# Phase 5 — Constrained Counterfactual XAI

Phase 5 uses the existing Phase 4 selected checkpoint `models/mlp_final_selected.pt`. It does **not** retrain Phase 4.

Run from the project root:

```bash
python code/run_phase5.py
```

The implementation compares unconstrained gradient counterfactuals with constrained counterfactuals for held-out Medium/High cases targeted toward Low risk.

Objective: target cross-entropy + 0.01 L1 + 0.01 L2.

Constraints:
- Immutable: `age`, `previous_default_count`, `transaction_count`
- Monotonic decrease-only: `outstanding_dues`, `late_filing_count`, `transaction_anomaly_ratio`
- Bounds: `deduction_claim_ratio` and `transaction_anomaly_ratio` in [0, 1]
- `income_sources` remains unchanged
- `transaction_anomaly_count = transaction_anomaly_ratio × transaction_count`

Outputs are written under `results/metrics/` and `results/figures/`.

Interpretation: these are technical counterfactual recourse results on synthetic data and do not establish real-world regulatory or compliance effectiveness.
