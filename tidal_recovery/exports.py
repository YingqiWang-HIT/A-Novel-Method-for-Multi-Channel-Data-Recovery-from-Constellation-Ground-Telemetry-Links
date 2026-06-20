from __future__ import annotations

from pathlib import Path
from typing import Dict, List

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import yaml

from .metrics import per_channel_rmse
from .utils import ensure_dir, write_json


def export_run(output_dir: str, cfg, data, metrics_rows: List[dict], trained: Dict[str, dict], predictions: Dict[str, dict]) -> None:
    out = ensure_dir(output_dir)
    pd.DataFrame(metrics_rows).to_csv(out / "metrics_summary.csv", index=False)
    pd.DataFrame(metrics_rows).to_excel(out / "metrics_summary.xlsx", index=False)
    with (out / "run_config.yaml").open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg.__dict__, f, sort_keys=False, allow_unicode=True)
    write_json(data.get("metadata", {}), out / "dataset_metadata.json")
    np.save(out / "adjacency.npy", data["A"])

    if cfg.save_model:
        model_dir = ensure_dir(out / "trained_models")
        hist_dir = ensure_dir(out / "training_history")
        for name, pack in trained.items():
            safe = name.replace("/", "-")
            torch.save(pack["model"].state_dict(), model_dir / f"{safe}.pt")
            pd.DataFrame(pack["info"]["history"]).to_csv(hist_dir / f"{safe}_history.csv", index=False)

    if cfg.save_predictions:
        pred_dir = ensure_dir(out / "predictions")
        for name, pack in predictions.items():
            safe = name.replace("/", "-")
            np.savez_compressed(pred_dir / f"{safe}_predictions.npz", pred=pack["pred"], target=pack["target"], mask=pack["mask"])
            ch_rmse = per_channel_rmse(pack["pred"], pack["target"], data.get("channel_names"))
            pd.DataFrame([{"Channel": k, "RMSE": v} for k, v in ch_rmse.items()]).to_csv(pred_dir / f"{safe}_per_channel_rmse.csv", index=False)
            plot_prediction(pack["pred"], pack["target"], pack["mask"], data.get("channel_names"), pred_dir / f"{safe}_example.png", cfg)
    plot_metrics(metrics_rows, out / "rmse_comparison.png")


def plot_metrics(rows: List[dict], path: Path) -> None:
    df = pd.DataFrame(rows)
    if df.empty:
        return
    plt.figure(figsize=(8, 4))
    plt.bar(df["Model"], df["RMSE"])
    plt.ylabel("RMSE")
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(path, dpi=300)
    plt.close()


def plot_prediction(pred: np.ndarray, target: np.ndarray, mask: np.ndarray, channel_names, path: Path, cfg) -> None:
    # Flatten sample-horizon for a compact visualization.
    y = target.reshape(-1, target.shape[-1])
    p = pred.reshape(-1, pred.shape[-1])
    m = mask.reshape(-1, mask.shape[-1])
    L = min(cfg.plot_len, len(y))
    n_ch = min(cfg.plot_channels, y.shape[-1])
    for ch in range(n_ch):
        plt.figure(figsize=(8, 3))
        plt.plot(y[:L, ch], label="Target", linewidth=1.3)
        plt.plot(p[:L, ch], label="Prediction", linewidth=1.0)
        idx = np.where(m[:L, ch] > 0.5)[0]
        if len(idx):
            plt.scatter(idx, y[idx, ch], s=12, label="Anomaly region")
        title = channel_names[ch] if channel_names else f"Channel {ch}"
        plt.title(title)
        plt.xlabel("Flattened prediction index")
        plt.ylabel("Value")
        plt.legend()
        plt.tight_layout()
        stem = path.with_suffix("")
        plt.savefig(stem.parent / f"{stem.name}_ch{ch}.png", dpi=300)
        plt.close()
