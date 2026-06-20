from __future__ import annotations

from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Tuple, Any, Dict

import yaml


@dataclass
class Config:
    # Data
    excel_path: str = ""
    sheet_name: int | str = 0
    time_col: Optional[str] = None
    first_col_is_time: bool = True
    output_dir: str = "outputs/tidal_run"
    seed: int = 42

    # Task
    seq_len: int = 96
    pred_len: int = 12
    train_ratio: float = 0.60
    val_ratio: float = 0.20

    # Artificial timestamp anomalies
    anomaly_rate: float = 0.05
    missing_fraction: float = 0.60
    drift_fraction: float = 0.40
    drift_max_lag: int = 4
    block_count: int = 12
    block_min_len: int = 4
    block_max_len: int = 24
    add_noise_std: float = 0.01

    # Graph
    top_k_graph: int = 6
    graph_threshold: float = 0.05
    k_max: int = 6

    # Model
    hidden_dim: int = 64
    node_input_dim: int = 3
    dropout: float = 0.10
    ode_steps: int = 4
    tau_min: float = 0.10
    lambda_abs: float = 0.20
    missing_loss_weight: float = 0.30
    smooth_loss_weight: float = 0.03

    # Training
    use_gpu: bool = True
    batch_size: int = 32
    epochs: int = 500
    lr: float = 5e-4
    weight_decay: float = 1e-4
    patience: int = 50
    min_delta: float = 1e-6
    grad_clip: float = 1.0
    num_workers: int = 0
    amp: bool = True

    # Running
    models: Optional[List[str]] = None
    save_predictions: bool = True
    save_model: bool = True
    plot_channels: int = 4
    plot_len: int = 500


def _coerce_key_value(value: str) -> Any:
    """Parse a command-line override value into a Python object when possible."""
    lower = value.lower()
    if lower in {"true", "false"}:
        return lower == "true"
    if lower in {"none", "null"}:
        return None
    if "," in value:
        return [v.strip() for v in value.split(",") if v.strip()]
    try:
        if any(c in value for c in [".", "e", "E"]):
            return float(value)
        return int(value)
    except ValueError:
        return value


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Config:
    data: Dict[str, Any] = {}
    if path:
        p = Path(path)
        if p.exists():
            with p.open("r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f) or {}
                if not isinstance(loaded, dict):
                    raise ValueError(f"Config file must contain a mapping, got {type(loaded)!r}")
                data.update(loaded)
    if overrides:
        for k, v in overrides.items():
            if v is not None:
                data[k] = v
    valid = set(Config.__dataclass_fields__.keys())
    unknown = sorted(set(data) - valid)
    if unknown:
        raise KeyError(f"Unknown config keys: {unknown}")
    return Config(**data)


def save_config(cfg: Config, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        yaml.safe_dump(asdict(cfg), f, sort_keys=False, allow_unicode=True)
