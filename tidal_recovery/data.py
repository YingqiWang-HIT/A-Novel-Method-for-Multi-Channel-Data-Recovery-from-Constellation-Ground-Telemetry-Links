from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Dict, Optional, Tuple, List

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from .config import Config


def read_telemetry_table(excel_path: str, sheet_name: int | str = 0, time_col: Optional[str] = None,
                         first_col_is_time: bool = True) -> Tuple[np.ndarray, List[str], Optional[np.ndarray]]:
    path = Path(excel_path)
    if not path.exists():
        raise FileNotFoundError(f"Telemetry file not found: {excel_path}")
    if path.suffix.lower() in {".xlsx", ".xls"}:
        df = pd.read_excel(path, sheet_name=sheet_name)
    elif path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        raise ValueError("Supported file formats: .xlsx, .xls, .csv")
    df = df.dropna(axis=0, how="all").dropna(axis=1, how="all")
    if df.shape[1] < 2:
        raise ValueError("The file should contain one time/index column and at least one telemetry channel.")

    time_values = None
    if time_col is not None:
        if time_col not in df.columns:
            raise KeyError(f"time_col={time_col!r} not found in columns")
        time_values = df[time_col].to_numpy()
        df = df.drop(columns=[time_col])
    elif first_col_is_time:
        time_values = df.iloc[:, 0].to_numpy()
        df = df.iloc[:, 1:].copy()

    for c in df.columns:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(axis=1, how="all")
    if df.shape[1] == 0:
        raise ValueError("No numeric telemetry channel remains after cleaning.")
    return df.to_numpy(dtype=np.float32), [str(c) for c in df.columns], time_values


def interpolate_target(raw: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(raw)
    df = df.interpolate(method="linear", axis=0, limit_direction="both")
    df = df.ffill().bfill().fillna(0.0)
    return df.to_numpy(dtype=np.float32)


def forward_fill_observation(x: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(x)
    df = df.ffill().bfill().fillna(0.0)
    return df.to_numpy(dtype=np.float32)


def simulate_timestamp_anomalies(target: np.ndarray, cfg: Config, rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Create missing and timestamp-drift anomalies.

    Returns
    -------
    corrupted : [T,N]
        Observed sequence after noise, missing values, and drifted segments.
    observed_mask : [T,N]
        1 for observed samples, 0 for deleted samples.
    anomaly_mask : [T,N]
        1 for artificially missing or drifted locations, used for missing-region metrics.
    """
    T, N = target.shape
    corrupted = target.copy()
    observed_mask = np.ones((T, N), dtype=np.float32)
    anomaly_mask = np.zeros((T, N), dtype=np.float32)

    if cfg.add_noise_std > 0:
        scale = np.nanstd(target, axis=0, keepdims=True)
        scale = np.where(scale < 1e-6, 1.0, scale)
        corrupted = corrupted + rng.normal(0.0, cfg.add_noise_std, size=target.shape).astype(np.float32) * scale.astype(np.float32)

    total_points = T * N
    n_anom = max(1, int(total_points * cfg.anomaly_rate))
    n_missing = int(n_anom * cfg.missing_fraction)
    flat = rng.choice(total_points, size=min(n_missing, total_points), replace=False)
    rows, cols = np.unravel_index(flat, (T, N))
    corrupted[rows, cols] = np.nan
    observed_mask[rows, cols] = 0.0
    anomaly_mask[rows, cols] = 1.0

    # Block missing / drift is closer to link instability than purely point-wise deletion.
    for _ in range(cfg.block_count):
        length = int(rng.integers(cfg.block_min_len, cfg.block_max_len + 1))
        start = int(rng.integers(0, max(1, T - length)))
        ch = int(rng.integers(0, N))
        if rng.random() < cfg.missing_fraction:
            corrupted[start:start + length, ch] = np.nan
            observed_mask[start:start + length, ch] = 0.0
        else:
            lag = int(rng.integers(1, cfg.drift_max_lag + 1))
            sign = -1 if rng.random() < 0.5 else 1
            src_start = np.clip(start + sign * lag, 0, max(0, T - length))
            corrupted[start:start + length, ch] = target[src_start:src_start + length, ch]
        anomaly_mask[start:start + length, ch] = 1.0

    # Additional drift points.
    n_drift = int(n_anom * cfg.drift_fraction)
    if n_drift > 0:
        flat = rng.choice(total_points, size=min(n_drift, total_points), replace=False)
        rows, cols = np.unravel_index(flat, (T, N))
        for r, c in zip(rows, cols):
            lag = int(rng.integers(1, cfg.drift_max_lag + 1))
            rr = int(np.clip(r + (lag if rng.random() < 0.5 else -lag), 0, T - 1))
            corrupted[r, c] = target[rr, c]
            anomaly_mask[r, c] = 1.0

    return corrupted.astype(np.float32), observed_mask, anomaly_mask


def compute_delta(mask: np.ndarray, clip_value: int = 80) -> np.ndarray:
    T, N = mask.shape
    delta = np.zeros((T, N), dtype=np.float32)
    for t in range(1, T):
        delta[t] = np.where(mask[t] > 0.5, 0.0, delta[t - 1] + 1.0)
    return np.clip(delta, 0, clip_value).astype(np.float32) / float(clip_value)


def time_features(T: int, periods=(24, 60, 180, 360)) -> np.ndarray:
    t = np.arange(T, dtype=np.float32)
    feats = []
    for p in periods:
        feats.append(np.sin(2 * np.pi * t / p))
        feats.append(np.cos(2 * np.pi * t / p))
    feats.append(t / max(1, T - 1))
    return np.stack(feats, axis=-1).astype(np.float32)


def build_correlation_graph(target_norm: np.ndarray, train_end: int, top_k: int, threshold: float) -> np.ndarray:
    train = target_norm[:train_end]
    corr = np.corrcoef(train.T)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    corr = np.abs(corr).astype(np.float32)
    np.fill_diagonal(corr, 1.0)
    N = corr.shape[0]
    k = min(max(1, top_k), N)
    A = np.zeros_like(corr, dtype=np.float32)
    for i in range(N):
        idx = np.argsort(corr[i])[-k:]
        A[i, idx] = corr[i, idx]
    A[A < threshold] = 0.0
    np.fill_diagonal(A, 1.0)
    A = np.maximum(A, A.T)
    return (A / (A.sum(axis=1, keepdims=True) + 1e-8)).astype(np.float32)


def make_windows(features: np.ndarray, target: np.ndarray, anomaly_mask: np.ndarray, seq_len: int, pred_len: int):
    X, Y, M = [], [], []
    max_i = len(features) - seq_len - pred_len + 1
    for i in range(max_i):
        X.append(features[i:i + seq_len])
        Y.append(target[i + seq_len:i + seq_len + pred_len])
        M.append(anomaly_mask[i + seq_len:i + seq_len + pred_len])
    return np.asarray(X, np.float32), np.asarray(Y, np.float32), np.asarray(M, np.float32)


def prepare_dataset(cfg: Config) -> Dict[str, object]:
    raw, channel_names, time_values = read_telemetry_table(
        cfg.excel_path, cfg.sheet_name, cfg.time_col, cfg.first_col_is_time
    )
    target = interpolate_target(raw)
    T, N = target.shape
    if T < cfg.seq_len + cfg.pred_len + 3:
        raise ValueError(f"Time series too short: T={T}, seq_len={cfg.seq_len}, pred_len={cfg.pred_len}")

    rng = np.random.default_rng(cfg.seed)
    corrupted, observed_mask, anomaly_mask = simulate_timestamp_anomalies(target, cfg, rng)
    filled = forward_fill_observation(corrupted)

    train_time_end = int(T * cfg.train_ratio)
    mean = target[:train_time_end].mean(axis=0).astype(np.float32)
    std = target[:train_time_end].std(axis=0).astype(np.float32)
    std = np.where(std < 1e-6, 1.0, std).astype(np.float32)

    target_norm = ((target - mean) / std).astype(np.float32)
    filled_norm = ((filled - mean) / std).astype(np.float32)
    delta = compute_delta(observed_mask)
    tf = time_features(T)
    features = np.concatenate([filled_norm, observed_mask, delta, tf], axis=-1).astype(np.float32)

    A = build_correlation_graph(target_norm, train_time_end, cfg.top_k_graph, cfg.graph_threshold)
    X, Y, M = make_windows(features, target_norm, anomaly_mask, cfg.seq_len, cfg.pred_len)
    n = len(X)
    n_train = int(n * cfg.train_ratio)
    n_val = int(n * cfg.val_ratio)
    n_train = min(max(1, n_train), n - 2)
    n_val = min(max(1, n_val), n - n_train - 1)

    split = {
        "X_train": torch.tensor(X[:n_train]),
        "Y_train": torch.tensor(Y[:n_train]),
        "M_train": torch.tensor(M[:n_train]),
        "X_val": torch.tensor(X[n_train:n_train + n_val]),
        "Y_val": torch.tensor(Y[n_train:n_train + n_val]),
        "M_val": torch.tensor(M[n_train:n_train + n_val]),
        "X_test": torch.tensor(X[n_train + n_val:]),
        "Y_test": torch.tensor(Y[n_train + n_val:]),
        "M_test": torch.tensor(M[n_train + n_val:]),
        "mean": mean,
        "std": std,
        "A": A,
        "channel_names": channel_names,
        "time_values": time_values,
        "N": N,
        "T": T,
        "input_dim": features.shape[-1],
        "seq_len": cfg.seq_len,
        "pred_len": cfg.pred_len,
        "metadata": {
            "source": cfg.excel_path,
            "T": int(T),
            "N": int(N),
            "anomaly_rate": float(anomaly_mask.mean()),
            "observed_rate": float(observed_mask.mean()),
        },
    }
    return split


def make_loaders(data: Dict[str, object], cfg: Config):
    train = DataLoader(TensorDataset(data["X_train"], data["Y_train"], data["M_train"]),
                       batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    val = DataLoader(TensorDataset(data["X_val"], data["Y_val"], data["M_val"]),
                     batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    test = DataLoader(TensorDataset(data["X_test"], data["Y_test"], data["M_test"]),
                      batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train, val, test
