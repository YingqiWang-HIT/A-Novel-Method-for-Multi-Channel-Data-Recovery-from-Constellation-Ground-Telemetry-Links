#!/usr/bin/env python
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tidal_recovery.config import load_config
from tidal_recovery.data import prepare_dataset, make_loaders
from tidal_recovery.exports import export_run
from tidal_recovery.models import build_models
from tidal_recovery.trainer import train_evaluate_all
from tidal_recovery.utils import get_device, set_seed, ensure_dir


def parse_args():
    p = argparse.ArgumentParser(description="Train and test TiDAL-Net for irregular satellite telemetry reconstruction.")
    p.add_argument("--config", type=str, default="configs/default.yaml")
    p.add_argument("--excel_path", type=str, default=None, help="Path to Excel/CSV telemetry data. First column is time by default.")
    p.add_argument("--output_dir", type=str, default=None)
    p.add_argument("--models", type=str, default=None, help="Comma-separated model names, e.g. TiDAL-Full,TiDAL-w/o-BiNODE")
    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch_size", type=int, default=None)
    p.add_argument("--use_gpu", type=str, default=None, choices=["true", "false", "True", "False"])
    p.add_argument("--anomaly_rate", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    overrides = {
        "excel_path": args.excel_path,
        "output_dir": args.output_dir,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "anomaly_rate": args.anomaly_rate,
    }
    if args.models:
        overrides["models"] = [m.strip() for m in args.models.split(",") if m.strip()]
    if args.use_gpu is not None:
        overrides["use_gpu"] = args.use_gpu.lower() == "true"
    cfg = load_config(args.config, overrides)
    if not cfg.excel_path:
        raise ValueError("Please provide --excel_path or set excel_path in the config file.")
    ensure_dir(cfg.output_dir)
    set_seed(cfg.seed)
    device = get_device(cfg.use_gpu)
    print(f"[Device] {device}")
    print(f"[Data] {cfg.excel_path}")
    data = prepare_dataset(cfg)
    print(f"[Data] T={data['T']}, N={data['N']}, input_dim={data['input_dim']}, train={len(data['X_train'])}, val={len(data['X_val'])}, test={len(data['X_test'])}")
    loaders = make_loaders(data, cfg)
    model_names = cfg.models if cfg.models else ["TiDAL-Full"]
    models = build_models(model_names, data, cfg)
    print("[Models]", ", ".join(models.keys()))
    rows, trained, predictions = train_evaluate_all(models, loaders, data, cfg, device)
    export_run(cfg.output_dir, cfg, data, rows, trained, predictions)
    print("\n[Done] Metrics:")
    for r in rows:
        print(f"  {r['Model']}: RMSE={r['RMSE']:.6f}, MAE={r['MAE']:.6f}, MAPE={r['MAPE']:.6f}, latency={r['Latency_s_per_sample']*1000:.3f} ms/sample")
    print(f"[Output] {cfg.output_dir}")


if __name__ == "__main__":
    main()
