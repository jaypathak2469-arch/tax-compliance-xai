"""Live risk prediction.

predict() runs a raw taxpayer profile through the existing fitted preprocessor
and the existing trained MLP checkpoint. It never trains, refits, or falls back
to a fabricated result: a pipeline failure raises pipeline.PipelineError, which
the calling view surfaces to the person rather than showing a guessed class.
"""
from __future__ import annotations

from . import pipeline


def predict(raw: dict) -> dict:
    """Run one profile through the live pipeline.

    Returns {"x0": tensor, "probs": np.ndarray[3], "pred_idx": int,
    "pred_class": str, "raw_input": dict}.
    """
    x0 = pipeline.encode(raw)
    probs = pipeline.predict_probs(x0)
    pred_idx = int(probs.argmax())
    return {
        "x0": x0,
        "probs": probs,
        "pred_idx": pred_idx,
        "pred_class": pipeline.CLASSES[pred_idx],
        "raw_input": dict(raw),
    }
