from __future__ import annotations

import copy
import time
from typing import Dict, Tuple

import numpy as np
import torch
from torch import amp

from .losses import reconstruction_loss
from .metrics import compute_metrics, denormalize
from .utils import count_parameters


def _forward_model(model, x):
    out = model(x)
    if isinstance(out, tuple):
        return out[0], out[1]
    return out, {}


def train_one_model(model, loaders, cfg, device: torch.device) -> Tuple[torch.nn.Module, Dict[str, object]]:
    train_loader, val_loader, _ = loaders
    model = model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    scaler = amp.GradScaler("cuda", enabled=cfg.amp and device.type == "cuda")
    best_state = copy.deepcopy(model.state_dict())
    best_val = float("inf")
    bad = 0
    history = {"train_loss": [], "val_loss": [], "epoch_time": []}
    started = time.time()

    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        model.train()
        train_losses = []
        for xb, yb, mb in train_loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            mb = mb.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with amp.autocast("cuda", enabled=cfg.amp and device.type == "cuda"):
                pred, _ = _forward_model(model, xb)
                loss = reconstruction_loss(pred, yb, mb, cfg)
            scaler.scale(loss).backward()
            if cfg.grad_clip and cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()
            train_losses.append(float(loss.detach().cpu()))

        val_loss = evaluate_loss(model, val_loader, cfg, device)
        history["train_loss"].append(float(np.mean(train_losses)))
        history["val_loss"].append(float(val_loss))
        history["epoch_time"].append(float(time.time() - t0))

        if val_loss < best_val - cfg.min_delta:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())
            bad = 0
        else:
            bad += 1
        if bad >= cfg.patience:
            break

    model.load_state_dict(best_state)
    info = {
        "best_val_loss": float(best_val),
        "epochs_trained": len(history["train_loss"]),
        "train_seconds": float(time.time() - started),
        "parameters": int(count_parameters(model)),
        "history": history,
    }
    return model, info


@torch.no_grad()
def evaluate_loss(model, loader, cfg, device: torch.device) -> float:
    model.eval()
    losses = []
    for xb, yb, mb in loader:
        xb = xb.to(device, non_blocking=True)
        yb = yb.to(device, non_blocking=True)
        mb = mb.to(device, non_blocking=True)
        pred, _ = _forward_model(model, xb)
        loss = reconstruction_loss(pred, yb, mb, cfg)
        losses.append(float(loss.detach().cpu()))
    return float(np.mean(losses)) if losses else float("nan")


@torch.no_grad()
def predict(model, loader, device: torch.device):
    model.eval()
    preds, targets, masks = [], [], []
    infer_times = []
    for xb, yb, mb in loader:
        xb = xb.to(device, non_blocking=True)
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.time()
        pred, _ = _forward_model(model, xb)
        if device.type == "cuda":
            torch.cuda.synchronize()
        infer_times.append((time.time() - t0) / max(1, xb.size(0)))
        preds.append(pred.detach().cpu().numpy())
        targets.append(yb.numpy())
        masks.append(mb.numpy())
    return np.concatenate(preds, axis=0), np.concatenate(targets, axis=0), np.concatenate(masks, axis=0), float(np.mean(infer_times))


def train_evaluate_all(models: Dict[str, torch.nn.Module], loaders, data, cfg, device):
    rows = []
    trained = {}
    predictions = {}
    for name, model in models.items():
        model, info = train_one_model(model, loaders, cfg, device)
        pred, target, mask, latency = predict(model, loaders[2], device)
        pred_den = denormalize(pred, data["mean"], data["std"])
        target_den = denormalize(target, data["mean"], data["std"])
        metric = compute_metrics(pred_den, target_den, mask)
        row = {"Model": name, **metric, "Latency_s_per_sample": latency, **{k: v for k, v in info.items() if k != "history"}}
        rows.append(row)
        trained[name] = {"model": model, "info": info}
        predictions[name] = {"pred": pred_den, "target": target_den, "mask": mask}
    return rows, trained, predictions
