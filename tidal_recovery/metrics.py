from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import torch


def denormalize(x: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    return x * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1)


def _safe_mape(pred: np.ndarray, target: np.ndarray, eps: float = 1e-3) -> float:
    denom = np.maximum(np.abs(target), eps)
    return float(np.mean(np.abs((pred - target) / denom)))


def compute_metrics(pred: np.ndarray, target: np.ndarray, miss_mask: np.ndarray | None = None) -> Dict[str, float]:
    err = pred - target
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    mape = _safe_mape(pred, target)
    out = {"MAE": mae, "RMSE": rmse, "MAPE": mape}
    if miss_mask is not None and miss_mask.sum() > 0:
        m = miss_mask.astype(bool)
        me = err[m]
        out.update({
            "Missing_MAE": float(np.mean(np.abs(me))),
            "Missing_RMSE": float(np.sqrt(np.mean(me ** 2))),
            "Missing_MAPE": _safe_mape(pred[m], target[m]),
        })
    else:
        out.update({"Missing_MAE": np.nan, "Missing_RMSE": np.nan, "Missing_MAPE": np.nan})
    return out


def per_channel_rmse(pred: np.ndarray, target: np.ndarray, channel_names=None) -> Dict[str, float]:
    rmse = np.sqrt(np.mean((pred - target) ** 2, axis=(0, 1)))
    if channel_names is None:
        channel_names = [f"ch_{i}" for i in range(pred.shape[-1])]
    return {str(c): float(v) for c, v in zip(channel_names, rmse)}
