#!/usr/bin/env python
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "examples" / "synthetic_tidal_demo.xlsx"
OUT = ROOT / "outputs" / "demo"


def make_demo_data(path: Path, T: int = 48, N: int = 3):
    rng = np.random.default_rng(7)
    t = np.arange(T)
    cols = {"time": t}
    base = np.sin(2 * np.pi * t / 60) + 0.4 * np.sin(2 * np.pi * t / 144)
    switch = (t > T // 2).astype(float)
    for i in range(N):
        phase = i / max(1, N) * np.pi
        signal = (1 + 0.08 * i) * np.sin(2 * np.pi * t / (48 + i * 3) + phase)
        coupled = 0.35 * base + 0.25 * switch * np.cos(2 * np.pi * t / (36 + i))
        cols[f"sensor_{i+1:02d}"] = signal + coupled + 0.03 * rng.normal(size=T)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(cols).to_excel(path, index=False)


def main():
    make_demo_data(DATA)
    cmd = [
        sys.executable, str(ROOT / "scripts" / "run_train_test.py"),
        "--config", str(ROOT / "examples" / "demo_config.yaml"),
        "--excel_path", str(DATA),
        "--output_dir", str(OUT),
        "--models", "TiDAL-Full",
        "--epochs", "1",
        "--batch_size", "32",
        "--use_gpu", "false",
    ]
    subprocess.run(cmd, check=True)
    print(f"Demo finished. Outputs are saved to {OUT}")


if __name__ == "__main__":
    main()
