# TAX-XAI application

A user-facing application built around the existing Phase 1–5 pipeline.

## Run

From the project root:

```bash
cd ~/Desktop/"xai_project 3"
python -m pip install -r requirements-app.txt
streamlit run app/app.py
```

## Guarantees

The application is strictly read-only with respect to the research project.
It does not write to `code/`, `data/`, `models/` or `results/`, does not train
or refit anything, and never imports `code/run_phase5_reconstruction.py`.

All model, preprocessing and counterfactual logic is reused from `code/` rather
than reimplemented. Paths are derived from `app/services/paths.py` relative to
the project root, so the project can be moved or renamed.

`app/services/pipeline.py::encode()` selects raw-input columns by name in the
exact order recorded in `data/processed/scaler_stats.csv`, so the 25-feature
order can never silently drift from the fitted preprocessor.

## Layout

```
app/
├── app.py                     entry point, navigation, routing
├── config/reported_metrics.json   Phase 3/4 numbers transcribed from the PPT
├── services/
│   ├── paths.py                project-relative paths, code/ import shim
│   ├── artifacts.py            real filesystem, environment and engine checks
│   ├── datasets.py             cached read-only access to data and results
│   ├── pipeline.py             loads preprocessor + MLP; encode()/decode_raw()
│   ├── predictor.py            live prediction via the real pipeline
│   ├── explainer.py            live counterfactual generation (wraps code/counterfactual.py)
│   └── validation.py           input validation + constraint-report grouping
├── ui/
│   ├── theme.py                palette, Plotly dark theme
│   └── components.py           glass cards, KPI tiles, risk/probability rows, notices
├── views/                      one module per page
└── styles/main.css             glassmorphism, typography, motion
```

## Build phases

| Phase | Scope | State |
|---|---|---|
| D | UI foundation, real artifact checks, real dataset reads | done |
| E | Visual review | approved |
| F | Live MLP inference, live counterfactual generation, live constraint validation | done |
| G / H | Folded into Phase F at the user's request | done |
| I | Analytics completion, CSV/PDF export | pending |
| J | End-to-end testing | in progress — see PHASE_F_TEST_REPORT.md |
