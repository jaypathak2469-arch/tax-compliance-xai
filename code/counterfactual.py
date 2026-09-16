from __future__ import annotations
import time
import torch
from torch import nn
from constraints import project_constraints, validate_constraints

CLASSES = ["Low", "Medium", "High"]

class MLP(nn.Module):
    def __init__(self, d=25):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d, 64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(32, 3)
        )
    def forward(self, x):
        return self.net(x)


def load_model(checkpoint_path, input_dim=25):
    model = MLP(input_dim)
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    model.load_state_dict(state)
    model.eval()
    return model


def generate_counterfactual(model, x0, feature_to_idx, scaler_stats, target=0,
                            constrained=True, steps=600, lr=0.03, l1=0.01, l2=0.01,
                            success_probability=0.60):
    x = x0.detach().clone().requires_grad_(True)
    optimizer = torch.optim.Adam([x], lr=lr)
    target_t = torch.tensor([target], dtype=torch.long)
    best = x0.detach().clone()
    best_obj = float("inf")
    best_step = 0
    start = time.perf_counter()

    for step in range(1, steps + 1):
        optimizer.zero_grad()
        logits = model(x.unsqueeze(0))
        ce = nn.functional.cross_entropy(logits, target_t)
        delta = x - x0
        objective = ce + l1 * torch.abs(delta).sum() + l2 * torch.sum(delta ** 2)
        objective.backward()
        torch.nn.utils.clip_grad_norm_([x], 5.0)
        optimizer.step()
        with torch.no_grad():
            if constrained:
                x.copy_(project_constraints(x, x0, feature_to_idx, scaler_stats))
            probs = torch.softmax(model(x.unsqueeze(0)), dim=1)[0]
            pred = int(probs.argmax())
            obj_value = float(objective.detach())
            if obj_value < best_obj:
                best_obj = obj_value
                best = x.detach().clone()
                best_step = step
            if pred == target and float(probs[target]) >= success_probability:
                break

    elapsed = time.perf_counter() - start
    with torch.no_grad():
        probs = torch.softmax(model(best.unsqueeze(0)), dim=1)[0].numpy()
    pred = int(probs.argmax())
    valid, issues = validate_constraints(best, x0, feature_to_idx, scaler_stats)
    if not constrained:
        # Baseline is intentionally not expected to satisfy feasibility constraints.
        valid = True
        issues = []
    status = "SUCCESS" if pred == target else "OPTIMIZATION_FAILURE"
    if constrained and not valid:
        status = "INFEASIBLE"
    return {
        "x": best, "probs": probs, "pred": pred, "steps": best_step,
        "runtime": elapsed, "objective": best_obj, "valid": valid,
        "issues": issues, "status": status
    }
