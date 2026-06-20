from __future__ import annotations

import math
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2, dropout: float = 0.0):
        super().__init__()
        layers = []
        d = in_dim
        for _ in range(max(1, depth - 1)):
            layers += [nn.Linear(d, hidden_dim), nn.GELU(), nn.Dropout(dropout)]
            d = hidden_dim
        layers.append(nn.Linear(d, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BiNODECompensator(nn.Module):
    """
    Bidirectional neural ODE-like timestamp compensation.

    The paper formulates forward/backward NODE integrals with an attention fusion network.
    This implementation uses a stable explicit Euler/RK-style recurrent integration over the
    observed window and blends the directional estimates at low-reliability positions.
    """
    def __init__(self, n_channels: int, hidden_dim: int = 64, ode_steps: int = 4, dropout: float = 0.1):
        super().__init__()
        self.n_channels = n_channels
        self.ode_steps = max(1, ode_steps)
        self.forward_drift = MLP(n_channels + 1, hidden_dim, n_channels, depth=3, dropout=dropout)
        self.backward_drift = MLP(n_channels + 1, hidden_dim, n_channels, depth=3, dropout=dropout)
        # local features: forward/backward distance, local variance, local change, mask, delta
        self.fusion = nn.Sequential(
            nn.Linear(n_channels * 4 + 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_channels),
            nn.Sigmoid(),
        )
        self.residual = nn.Sequential(
            nn.Linear(n_channels * 3, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_channels),
        )

    def _integrate(self, obs: torch.Tensor, direction: str) -> torch.Tensor:
        B, L, N = obs.shape
        if direction == "forward":
            xs = obs
            drift = self.forward_drift
        else:
            xs = torch.flip(obs, dims=[1])
            drift = self.backward_drift
        state = xs[:, 0]
        outs = [state]
        for t in range(1, L):
            time_token = torch.full((B, 1), float(t) / max(1, L - 1), device=obs.device, dtype=obs.dtype)
            step = (xs[:, t] - state) / float(self.ode_steps)
            for _ in range(self.ode_steps):
                ode_in = torch.cat([state, time_token], dim=-1)
                state = state + (drift(ode_in) / float(L) + step)
            outs.append(state)
        out = torch.stack(outs, dim=1)
        return out if direction == "forward" else torch.flip(out, dims=[1])

    def forward(self, obs: torch.Tensor, mask: torch.Tensor, delta: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, L, N = obs.shape
        fwd = self._integrate(obs, "forward")
        bwd = self._integrate(obs, "backward")
        diff = torch.zeros_like(obs)
        diff[:, 1:] = obs[:, 1:] - obs[:, :-1]
        local_var = F.avg_pool1d(diff.abs().transpose(1, 2), kernel_size=5, stride=1, padding=2).transpose(1, 2)
        global_mask = mask.mean(dim=-1, keepdim=True)
        global_delta = delta.mean(dim=-1, keepdim=True)
        fuse_in = torch.cat([fwd, bwd, local_var, diff.abs(), global_mask, global_delta], dim=-1)
        alpha = self.fusion(fuse_in)
        compensated = alpha * fwd + (1.0 - alpha) * bwd
        reliability = mask * (1.0 - delta.clamp(0.0, 1.0))
        residual = 0.05 * self.residual(torch.cat([obs, mask, delta], dim=-1))
        out = reliability * obs + (1.0 - reliability) * compensated + residual
        return out, {"binode_alpha": alpha, "reliability": reliability}


class LiquidGRUCell(nn.Module):
    """GRU cell with input-dependent liquid time constants."""
    def __init__(self, input_dim: int, hidden_dim: int, tau_min: float = 0.1, dropout: float = 0.1):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.tau_min = tau_min
        tau_in = 2 * input_dim + hidden_dim
        self.tau_z = nn.Linear(tau_in, hidden_dim)
        self.tau_r = nn.Linear(tau_in, hidden_dim)
        self.tau_h = nn.Linear(tau_in, hidden_dim)
        self.x_z = nn.Linear(input_dim, hidden_dim)
        self.x_r = nn.Linear(input_dim, hidden_dim)
        self.x_h = nn.Linear(input_dim, hidden_dim)
        self.h_z = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.h_r = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.h_h = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.norm = nn.LayerNorm(hidden_dim)
        self.drop = nn.Dropout(dropout)

    def _tau(self, layer: nn.Linear, x: torch.Tensor, dx: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
        return F.softplus(layer(torch.cat([x, dx, h], dim=-1))) + self.tau_min

    def forward(self, x: torch.Tensor, dx: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        tau_z = self._tau(self.tau_z, x, dx, h)
        tau_r = self._tau(self.tau_r, x, dx, h)
        tau_h = self._tau(self.tau_h, x, dx, h)
        z = torch.sigmoid((self.x_z(x) + self.h_z(h)) / tau_z)
        r = torch.sigmoid((self.x_r(x) + self.h_r(h)) / tau_r)
        cand = torch.tanh((self.x_h(x) + self.h_h(r * h)) / tau_h)
        dh = z * (cand - h) / tau_h
        h_new = self.norm(h + dh)
        h_new = self.drop(h_new)
        return h_new, {"tau_z": tau_z, "tau_r": tau_r, "tau_h": tau_h, "update_gate": z}


class StandardNodeGRUCell(nn.Module):
    """A standard GRU-cell fallback used by TiDAL w/o Li-GRU."""
    def __init__(self, input_dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.cell = nn.GRUCell(input_dim, hidden_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, dx: torch.Tensor, h: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, N, D = x.shape
        h_new = self.cell(x.reshape(B * N, D), h.reshape(B * N, -1)).reshape_as(h)
        return self.drop(h_new), {}


class LiquidGraphAttention(nn.Module):
    """Dynamic local graph attention with time-constant synchronization."""
    def __init__(self, A: torch.Tensor, hidden_dim: int, k_max: int = 6, dropout: float = 0.1):
        super().__init__()
        A = torch.as_tensor(A, dtype=torch.float32)
        self.register_buffer("A0", A)
        self.hidden_dim = hidden_dim
        self.k_max = max(1, min(k_max, A.shape[0]))
        self.tau_net = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Linear(hidden_dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, 1), nn.Softplus())
        self.state_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim * 2),
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.gamma = nn.Parameter(torch.tensor(1.0))
        self.drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        B, N, H = h.shape
        tau = self.tau_net(h).squeeze(-1) + 1e-3  # [B,N]
        tau_sync = torch.exp(-F.softplus(self.gamma) * torch.abs(tau[:, :, None] - tau[:, None, :]))
        sim = F.cosine_similarity(h[:, :, None, :], h[:, None, :, :], dim=-1)
        prior = self.A0.to(h.device).unsqueeze(0).expand(B, -1, -1)
        score = sim * tau_sync + torch.log(prior + 1e-6)
        k = min(self.k_max, N)
        top_idx = torch.topk(score, k=k, dim=-1).indices
        mask = torch.zeros_like(score, dtype=torch.bool)
        mask.scatter_(-1, top_idx, True)
        score = score.masked_fill(~mask, -1e4 if h.dtype in (torch.float16, torch.bfloat16) else -1e9)
        att = torch.softmax(score.float(), dim=-1).to(h.dtype)
        att = self.drop(att)
        msg = torch.bmm(att, self.state_proj(h))
        h_plus = h + self.out(torch.cat([h, msg], dim=-1))
        return h_plus, {"dynamic_attention": att, "liquid_tau": tau}


class StaticGraphAttention(nn.Module):
    """Fixed graph attention fallback used by TiDAL w/o Li-GAT."""
    def __init__(self, A: torch.Tensor, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        A = torch.as_tensor(A, dtype=torch.float32)
        self.register_buffer("A0", A)
        self.proj = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim * 2), nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(), nn.Dropout(dropout))

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        A = self.A0.to(h.device)
        msg = torch.einsum("ij,bjh->bih", A, self.proj(h))
        return h + self.out(torch.cat([h, msg], dim=-1)), {"static_attention": A}
