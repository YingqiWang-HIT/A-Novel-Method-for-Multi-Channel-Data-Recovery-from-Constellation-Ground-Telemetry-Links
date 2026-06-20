from __future__ import annotations

from typing import Dict, Iterable

from .tidal_net import build_tidal_variant

DEFAULT_TIDAL_VARIANTS = [
    "TiDAL-w/o-NODE",
    "TiDAL-w/o-BiNODE",
    "TiDAL-w/o-GRU",
    "TiDAL-w/o-Li-GRU",
    "TiDAL-w/o-GAT",
    "TiDAL-w/o-Li-GAT",
    "TiDAL-Full",
]


def build_models(model_names, data, cfg):
    if model_names is None:
        model_names = ["TiDAL-Full"]
    if isinstance(model_names, str):
        model_names = [m.strip() for m in model_names.split(",") if m.strip()]
    models = {}
    for name in model_names:
        if name.lower().startswith("tidal"):
            models[name] = build_tidal_variant(
                name,
                data["A"],
                input_dim=data["input_dim"],
                seq_len=data["seq_len"],
                pred_len=data["pred_len"],
                n_channels=data["N"],
                hidden_dim=cfg.hidden_dim,
                node_input_dim=cfg.node_input_dim,
                dropout=cfg.dropout,
                ode_steps=cfg.ode_steps,
                tau_min=cfg.tau_min,
                k_max=cfg.k_max,
            )
        else:
            raise KeyError(
                f"Model {name!r} is not bundled in this public release. "
                "Third-party baselines are provided as separate adapter placeholders in ./baselines."
            )
    return models
