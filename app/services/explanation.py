"""Local model explanation for the live MLP.

This module computes gradient × input attribution for one live assessment.
It uses the exact fitted preprocessor and existing trained MLP already used
by the application.

The output describes local model behavior for the selected prediction. It
must not be interpreted as a causal explanation or as evidence that changing
a feature will produce the same change in the real world.
"""

from __future__ import annotations

import numpy as np

from . import pipeline


def explain_prediction(x0, predicted_index: int | None = None) -> dict:
    """Explain one prediction using gradient × input attribution.

    Parameters
    ----------
    x0:
        Standardized model input tensor produced by pipeline.encode().

    predicted_index:
        Optional class index. If omitted, the model's predicted class is used.

    Returns
    -------
    dict
        Contains class probabilities, predicted class, and feature-level
        attribution values.
    """

    torch = pipeline.require_torch()

    model = pipeline.get_model()
    feature_names, feature_to_idx = pipeline.get_feature_index()

    x = x0.detach().clone().requires_grad_(True)

    model.zero_grad(set_to_none=True)

    logits = model(x.unsqueeze(0))

    if predicted_index is None:
        predicted_index = int(logits.argmax(dim=1).item())

    target_logit = logits[0, predicted_index]

    target_logit.backward()

    gradients = x.grad.detach()

    # Local gradient × input attribution.
    attribution = gradients * x.detach()

    attribution_values = attribution.cpu().numpy().astype(float)

    # Convert logits to probabilities for consistency with the live predictor.
    with torch.no_grad():
        probabilities = torch.softmax(logits, dim=1)[0].cpu().numpy()

    # ------------------------------------------------------------------
    # Aggregate one-hot categorical features back to the original feature.
    # ------------------------------------------------------------------

    aggregated: dict[str, float] = {}

    for feature_name, value in zip(
        feature_names,
        attribution_values,
    ):
        if feature_name.startswith("income_sources_"):
            key = "income_sources"
        else:
            key = feature_name

        aggregated[key] = (
            aggregated.get(key, 0.0)
            + float(value)
        )

    # Keep only features that have meaningful attribution.
    rows = []

    for feature, value in aggregated.items():
        rows.append(
            {
                "feature": feature,
                "attribution": float(value),
                "absolute_attribution": abs(float(value)),
                "direction": (
                    "Higher model output"
                    if value > 0
                    else "Lower model output"
                    if value < 0
                    else "Neutral"
                ),
            }
        )

    rows.sort(
        key=lambda row: row["absolute_attribution"],
        reverse=True,
    )

    predicted_class = pipeline.CLASSES[predicted_index]

    return {
        "predicted_index": predicted_index,
        "predicted_class": predicted_class,
        "probabilities": probabilities,
        "features": rows,
        "method": "Gradient × Input",
        "interpretation": (
            "Local model-level attribution for the selected prediction. "
            "Positive values indicate contribution toward the selected "
            "class logit; negative values indicate contribution away from "
            "that class logit. These values are not causal effects."
        ),
    }


def top_features(
    explanation: dict,
    limit: int = 6,
) -> list[dict]:
    """Return the highest-magnitude feature contributions."""

    return explanation.get("features", [])[:limit]
