"""Dataset loading for the Constrained Counterfactual XAI project.

The integrated file was verified in Phase 1 to be internally consistent with the
raw sources (all five aggregates reproduce exactly, bar 2dp rounding on
average_transaction_value), so it is used as-is rather than rebuilt.
"""
from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def project_root() -> Path:
    return Path(__file__).resolve().parents[1]


def load_config(path: str | Path | None = None) -> dict:
    cfg_path = Path(path) if path else project_root() / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"config not found: {cfg_path}")
    with open(cfg_path) as fh:
        return yaml.safe_load(fh)


def _read(raw_dir: Path, name: str) -> pd.DataFrame:
    fp = raw_dir / name
    if not fp.exists():
        raise FileNotFoundError(f"expected dataset missing: {fp}")
    df = pd.read_csv(fp)
    logger.info("loaded %s shape=%s", name, df.shape)
    return df


def load_raw(cfg: dict) -> dict[str, pd.DataFrame]:
    """Load all four source files keyed by role."""
    raw_dir = project_root() / cfg["paths"]["raw_dir"]
    files = cfg["files"]
    return {
        "tax": _read(raw_dir, files["tax"]),
        "transactions": _read(raw_dir, files["transactions"]),
        "integrated": _read(raw_dir, files["integrated"]),
        "dictionary": _read(raw_dir, files["dictionary"]),
    }
