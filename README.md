# Constrained Counterfactual XAI Framework for Tax Compliance and Financial Risk Assessment

## Completed phases
- Phase 1: data understanding and validation
- Phase 2: feature engineering and preprocessing
- Phase 3: classical ML models
- Phase 4: MLP deep-learning model
- Phase 5: constrained counterfactual XAI

## Run Phase 5
From the project root:

```bash
python code/run_phase5.py
python code/make_phase5_figures.py
```

Phase 5 loads the existing Phase 4 checkpoint:

```text
models/mlp_final_selected.pt
```

It does not retrain Phase 4.

## Outputs
Results are written to `results/metrics/` and figures to `results/figures/`.

## Environment
Install dependencies from the project's requirements file if available, or use the existing project environment. The Phase 5 code requires PyTorch, pandas, numpy, scikit-learn, joblib, and PyYAML.

## Interpretation
Results are technical counterfactual recourse experiments on synthetic data. They do not establish real-world regulatory, tax, or compliance effectiveness.
