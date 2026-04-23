"""
models/snn_model.py — SNN Classifier with readout-window masking.

Spike rates for classification are computed ONLY over the readout window (mask=True).
Input-window spikes are explicitly excluded from the classification readout.
With all_recurrent=True, both layers have recurrent connections — necessary for a
45-step working memory delay.
"""
from __future__ import annotations
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Dict, List, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.lif_cell import LIFLayer, LIFState
from configs import SNNConfig


class SNNClassifier(nn.Module):
    def __init__(self, cfg: SNNConfig):
        super().__init__()
        self.cfg = cfg
        H, I, O, L = cfg.hidden_dim, cfg.input_dim, cfg.output_dim, cfg.n_layers
        all_rec = getattr(cfg, "all_recurrent", False)

        self.lif_layers = nn.ModuleList()
        for l in range(L):
            use_rec = all_rec or (l == L - 1)
            self.lif_layers.append(
                LIFLayer(
                    in_dim=(I if l == 0 else H),
                    out_dim=H,
                    tau_mem=cfg.tau_mem,
                    tau_syn=cfg.tau_syn,
                    v_th=cfg.v_th,
                    alpha=cfg.alpha_surr,
                    recurrent=use_rec,
                )
            )

        self.head = nn.Linear(H, O, bias=True)
        nn.init.xavier_uniform_(self.head.weight, gain=0.5)
        nn.init.zeros_(self.head.bias)

    @property
    def n_layers(self):
        return len(self.lif_layers)

    def initial_states(self, B: int, device: torch.device) -> List[LIFState]:
        return [layer.initial_state(B, device) for layer in self.lif_layers]

    def _masked_rate(self, spike_stack: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Compute mean firing rate over readout window only.

        Args:
            spike_stack: (B, T, H)
            mask:        (T,) bool, True in readout window

        Returns:
            (B, H) mean firing rate restricted to readout timesteps.
        """
        m = mask.unsqueeze(0).unsqueeze(-1).float()  # (1, T, 1)
        return (spike_stack * m).sum(dim=1) / mask.sum().clamp(min=1).float()

    def run_bptt(
        self, x: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        B, T, _ = x.shape
        device = x.device
        states = self.initial_states(B, device)
        prev   = [torch.zeros(B, self.cfg.hidden_dim, device=device) for _ in self.lif_layers]
        last: List[torch.Tensor] = []

        for t in range(T):
            h = x[:, t, :]
            for l, layer in enumerate(self.lif_layers):
                sp, ns = layer(h, states[l], prev_spikes=prev[l] if layer.recurrent else None)
                prev[l] = sp
                states[l] = ns
                h = sp
            last.append(h)

        spike_stack = torch.stack(last, dim=1)  # (B, T, H)
        rates  = self._masked_rate(spike_stack, mask.to(device))
        logits = self.head(rates)
        return logits, rates, spike_stack

    @torch.no_grad()
    def run_no_grad(self, x: torch.Tensor, mask: torch.Tensor) -> Dict[str, torch.Tensor]:
        B, T, _ = x.shape
        device = x.device
        L, H = self.n_layers, self.cfg.hidden_dim
        all_spikes = torch.zeros(L, B, T, H, device=device)
        all_vmem   = torch.zeros(L, B, T, H, device=device)
        states = self.initial_states(B, device)
        prev   = [torch.zeros(B, H, device=device) for _ in self.lif_layers]

        for t in range(T):
            h = x[:, t, :]
            for l, layer in enumerate(self.lif_layers):
                sp, ns = layer(h, states[l], prev_spikes=prev[l] if layer.recurrent else None)
                all_spikes[l, :, t, :] = sp
                all_vmem[l, :, t, :]   = ns.v
                prev[l]  = sp
                states[l] = ns
                h = sp

        last_sp = all_spikes[-1]  # (B, T, H)
        rates   = self._masked_rate(last_sp, mask.to(device))
        logits  = self.head(rates)
        return {"spikes": all_spikes, "vmem": all_vmem, "rates": rates, "logits": logits}

    @torch.no_grad()
    def check_dead_neurons(self, x, mask, threshold=0.005):
        out = self.run_no_grad(x, mask)
        lr  = out["spikes"].mean(dim=[1, 2])
        return {
            f"layer{l}": (lr[l] < threshold).sum().item() / self.cfg.hidden_dim
            for l in range(self.n_layers)
        }

    def count_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


def build_model(cfg: SNNConfig) -> SNNClassifier:
    assert cfg.hidden_dim > 0 and cfg.n_layers >= 1
    assert cfg.T > 0 and cfg.readout_len > 0 and cfg.readout_len < cfg.T
    return SNNClassifier(cfg)
