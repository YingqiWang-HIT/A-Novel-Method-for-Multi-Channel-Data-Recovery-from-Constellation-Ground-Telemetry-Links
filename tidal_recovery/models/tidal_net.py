from __future__ import annotations

from typing import Dict, Optional

import torch
import torch.nn as nn

from .components import (
    BiNODECompensator,
    LiquidGRUCell,
    LiquidGraphAttention,
    StandardNodeGRUCell,
    StaticGraphAttention,
)


class TiDALNet(nn.Module):
    """
    Time-Drift-Aware Liquid Spatio-Temporal Graph Network (TiDAL-Net).

    This implementation follows the manuscript-level module design:
    1. Bi-NODE: bidirectional continuous-time timestamp compensation.
    2. Li-GRU: liquid time-constant recurrent temporal reconstruction.
    3. Li-GAT: dynamic local graph attention for time-varying inter-channel coupling.

    Input shape is [B, L, 3N + T_f]: normalized observation, mask, delta, and global time features.
    Output shape is [B, pred_len, N].
    """
    def __init__(
        self,
        A,
        input_dim: int,
        seq_len: int,
        pred_len: int,
        n_channels: int,
        hidden_dim: int = 64,
        node_input_dim: int = 3,
        dropout: float = 0.10,
        ode_steps: int = 4,
        tau_min: float = 0.10,
        k_max: int = 6,
        use_binode: bool = True,
        use_li_gru: bool = True,
        use_li_gat: bool = True,
        unidirectional_node: bool = False,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.n_channels = n_channels
        self.hidden_dim = hidden_dim
        self.use_binode = use_binode
        self.use_li_gru = use_li_gru
        self.use_li_gat = use_li_gat
        self.unidirectional_node = unidirectional_node

        self.binode = BiNODECompensator(n_channels, hidden_dim, ode_steps=ode_steps, dropout=dropout)
        if use_li_gru:
            self.temporal_cell = LiquidGRUCell(node_input_dim, hidden_dim, tau_min=tau_min, dropout=dropout)
        else:
            self.temporal_cell = StandardNodeGRUCell(node_input_dim, hidden_dim, dropout=dropout)
        if use_li_gat:
            self.graph = LiquidGraphAttention(torch.as_tensor(A, dtype=torch.float32), hidden_dim, k_max=k_max, dropout=dropout)
        else:
            self.graph = StaticGraphAttention(torch.as_tensor(A, dtype=torch.float32), hidden_dim, dropout=dropout)
        self.decoder = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, pred_len),
        )
        self.shortcut = nn.Linear(seq_len, pred_len)
        self.output_gate = nn.Sequential(
            nn.Linear(hidden_dim + 3, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
            nn.Sigmoid(),
        )

    def _split(self, x: torch.Tensor):
        N = self.n_channels
        obs = x[:, :, :N]
        mask = x[:, :, N:2 * N]
        delta = x[:, :, 2 * N:3 * N]
        return obs, mask, delta

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        obs, mask, delta = self._split(x)
        extras: Dict[str, torch.Tensor] = {}
        if self.use_binode:
            compensated, info = self.binode(obs, mask, delta)
            if self.unidirectional_node:
                # Approximate w/o NODE variant: keep the forward NODE side dominant.
                compensated = info["binode_alpha"] * compensated + (1.0 - info["binode_alpha"]) * obs
            extras.update(info)
        else:
            compensated = obs

        B, L, N = compensated.shape
        h = torch.zeros(B, N, self.hidden_dim, device=x.device, dtype=x.dtype)
        prev_node = torch.zeros(B, N, 3, device=x.device, dtype=x.dtype)
        last_graph_info = {}
        last_cell_info = {}
        for t in range(L):
            node = torch.stack([compensated[:, t], mask[:, t], delta[:, t]], dim=-1)
            dx = node - prev_node if t > 0 else torch.zeros_like(node)
            h, cell_info = self.temporal_cell(node, dx, h)
            h, graph_info = self.graph(h)
            prev_node = node
            last_graph_info = graph_info
            last_cell_info = cell_info

        node_pred = self.decoder(h).permute(0, 2, 1)
        last_value = compensated[:, -1:, :]
        shortcut = self.shortcut((compensated - last_value).transpose(1, 2)).transpose(1, 2) + last_value
        summary = torch.cat([h, prev_node], dim=-1)
        gate = self.output_gate(summary).permute(0, 2, 1)
        pred = gate * node_pred + (1.0 - gate) * shortcut
        extras.update(last_cell_info)
        extras.update(last_graph_info)
        extras["output_gate"] = gate
        return pred, extras


def build_tidal_variant(name: str, A, input_dim: int, seq_len: int, pred_len: int, n_channels: int, **kwargs) -> TiDALNet:
    key = name.lower()
    flags = dict(use_binode=True, use_li_gru=True, use_li_gat=True, unidirectional_node=False)
    if key in {"tidal-w/o-node", "tidal-without-node"}:
        flags["unidirectional_node"] = True
    elif key in {"tidal-w/o-binode", "tidal-without-binode"}:
        flags["use_binode"] = False
    elif key in {"tidal-w/o-gru", "tidal-without-gru"}:
        flags["use_li_gru"] = False
    elif key in {"tidal-w/o-li-gru", "tidal-without-li-gru"}:
        flags["use_li_gru"] = False
    elif key in {"tidal-w/o-gat", "tidal-without-gat"}:
        flags["use_li_gat"] = False
    elif key in {"tidal-w/o-li-gat", "tidal-without-li-gat"}:
        flags["use_li_gat"] = False
    elif key not in {"tidal-full", "tidal-net", "tidalnet"}:
        raise KeyError(f"Unknown TiDAL-Net variant: {name}")
    return TiDALNet(A, input_dim, seq_len, pred_len, n_channels, **flags, **kwargs)
