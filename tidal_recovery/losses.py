from __future__ import annotations

import torch
import torch.nn.functional as F


def reconstruction_loss(pred: torch.Tensor, target: torch.Tensor, miss_mask: torch.Tensor, cfg) -> torch.Tensor:
    mse = F.mse_loss(pred, target)
    mae = F.l1_loss(pred, target)
    loss = mse + cfg.lambda_abs * mae
    if miss_mask is not None and miss_mask.sum() > 0:
        miss = ((pred - target).abs() * miss_mask).sum() / (miss_mask.sum() + 1e-8)
        loss = loss + cfg.missing_loss_weight * miss
    if getattr(cfg, "smooth_loss_weight", 0.0) > 0 and pred.size(1) > 1:
        dp = pred[:, 1:] - pred[:, :-1]
        dt = target[:, 1:] - target[:, :-1]
        loss = loss + cfg.smooth_loss_weight * F.l1_loss(dp, dt)
    return loss
