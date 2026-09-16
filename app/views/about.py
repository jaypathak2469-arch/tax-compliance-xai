"""About and methodology — written for a reader reviewing the project."""
from __future__ import annotations

import streamlit as st

from services import datasets
from ui import components as c

SECTIONS = [
    ("The problem",
     "A risk model that outputs Low, Medium or High tells a taxpayer where they "
     "stand but not what would move them. This project adds a recourse layer: for "
     "any Medium or High assessment it searches for the nearest profile the same "
     "model would classify as Low, under explicit rules about what may change."),
    ("Dataset",
     "Three synthetic sources: 5,000 taxpayer compliance profiles, 100,000 "
     "financial transactions, and a 5,000-row integrated file joining the two at "
     "taxpayer level. The integrated aggregates were reconciled against the "
     "transaction-level data before use."),
    ("Data integration and leakage control",
     "Four columns — overall_risk_score, financial_risk_score, tax_compliance_score "
     "and compliance_risk — stand in an exact arithmetic relationship to the target "
     "and were removed from the feature set. 54 profiles with no transaction records "
     "were excluded, since they carry no transaction features and their Low labels "
     "were defaults rather than earned. That leaves 4,946 modelled profiles."),
    ("Feature engineering",
     "Six taxpayer-level aggregates were computed from the transaction table: mean "
     "anomaly, location-risk and device-risk scores, mean transaction frequency, and "
     "the standard deviation and maximum of transaction amount. Single-transaction "
     "profiles take zero dispersion rather than a missing value."),
    ("Preprocessing",
     "Nineteen numeric features are standardised with StandardScaler and "
     "income_sources is one-hot encoded with handle_unknown='ignore', giving 25 "
     "model inputs. Both were fitted on the training split only, in a stratified "
     "70/15/15 split with seed 42. The application loads the fitted artifact and "
     "never refits it."),
    ("Models",
     "Logistic Regression, Random Forest and XGBoost were evaluated as classical "
     "baselines. XGBoost gave the strongest classical held-out metrics. A weighted "
     "MLP was trained separately and reported the highest held-out Macro F1 among "
     "the evaluated models."),
    ("MLP architecture",
     "25 inputs into a 64-unit ReLU layer with dropout 0.3, then a 32-unit ReLU "
     "layer with dropout 0.3, then a 3-way output. Trained with Adam at learning "
     "rate 1e-3, weight decay 1e-4, batch size 64, early stopping on validation "
     "Macro F1 with patience 30, and gradient clipping at norm 5. Weighted "
     "cross-entropy was selected."),
    ("Why the MLP carries the explanation stage",
     "The counterfactual search is gradient-based, so it needs a differentiable "
     "model. Tree ensembles are not differentiable end to end. This is a structural "
     "reason for the choice, not a claim that the MLP is the better classifier."),
    ("Counterfactual method",
     "Starting from the standardised profile, Adam minimises cross-entropy toward "
     "the Low class plus small L1 and L2 penalties on the displacement, stopping "
     "early once the model assigns Low at least 0.60 probability. The constrained "
     "variant projects the candidate back onto the feasibility set after every "
     "optimisation step."),
    ("Constraints",
     "Age, previous defaults and transaction count cannot change. Outstanding dues, "
     "late filings and the anomaly ratio may only decrease. The deduction claim "
     "ratio and the anomaly ratio are bounded to [0, 1]. income_sources keeps its "
     "original category as a clean one-hot. The anomalous-transaction count is "
     "recomputed as the anomaly ratio times the immutable transaction count."),
    ("Evaluation",
     "All 84 held-out Medium and High taxpayers were processed in both modes. Both "
     "reached the Low class in every case; the constrained mode satisfied every "
     "constraint check while the unconstrained mode satisfied none. Four deliberate "
     "corruptions were rejected by the validator on every profile tested."),
]

LIMITATIONS = [
    "The data is synthetic. Results demonstrate technical feasibility, not "
    "real-world regulatory or compliance effectiveness.",
    "Only 37 High-risk records exist across the dataset and 6 in the held-out test "
    "split, so High-class metrics carry wide uncertainty.",
    "Counterfactuals are model-level recourse under mathematical constraints, not "
    "causal claims about what would happen if a taxpayer acted on them.",
    "The constraints encode selected feasibility rules, not realistic domain ranges "
    "for every mutable feature, so some generated values can fall outside observed "
    "ranges.",
    "Changed-feature frequency records how often the optimiser moved a feature. It "
    "is not a measure of feature importance.",
    "L0, L1 and L2 are distances in standardised feature space and do not translate "
    "directly into monetary amounts.",
]


def render() -> None:
    c.page_header(
        "Methodology",
        "How the pipeline behind this application was built, and what its results "
        "can and cannot support.",
    )

    pop = datasets.population_counts()
    c.panel(
        "At a glance",
        c.stat_grid_html([
            ("Source profiles", f"{pop.get('source') or 0:,}"),
            ("Modelled profiles", f"{pop.get('modelled') or 0:,}"),
            ("Transactions", f"{pop.get('transactions') or 0:,}"),
            ("Model inputs", "25"),
        ], columns=4),
        delay=2,
    )
    c.spacer(1.2)

    for i, (title, body) in enumerate(SECTIONS):
        c.panel(title,
                f'<p style="color:var(--muted);font-size:.88rem;line-height:1.72;'
                f'margin:0;max-width:78ch">{body}</p>',
                glass=False)
        c.spacer(0.7)

    c.spacer(0.5)
    bullets = "".join(
        f'<li style="margin-bottom:.55rem;color:var(--muted);line-height:1.68">{l}</li>'
        for l in LIMITATIONS
    )
    c.panel("Limitations",
            f'<ul style="margin:0;padding-left:1.1rem;font-size:.87rem;max-width:78ch">'
            f'{bullets}</ul>',
            note="Stated up front rather than buried, because they bound every claim "
                 "this application makes.",
            delay=2)
