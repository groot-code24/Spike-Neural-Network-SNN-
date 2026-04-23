
from __future__ import annotations
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Dict, List
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.snn_model import SNNClassifier
from models.lif_cell import LIFState
from configs import BPTTConfig, TruncBPTTConfig


class BPTTTrainer:
    """Full BPTT — ground-truth gradient reference."""

    def __init__(self, model: SNNClassifier, cfg: BPTTConfig):
        self.model = model
        self.cfg   = cfg
        self.opt   = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self._last_grads: Dict[str, torch.Tensor] = {}

    def step(
        self, x: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> Dict[str, float]:
        self.opt.zero_grad()
        logits, rates, _ = self.model.run_bptt(x, mask)

        ce_loss   = F.cross_entropy(logits, labels)
        mean_rate = rates.mean()
        rate_loss = self.model.cfg.rate_penalty * (mean_rate - self.model.cfg.target_rate) ** 2
        loss      = ce_loss + rate_loss
        loss.backward()

        nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.grad_clip)
        self.opt.step()

        self._last_grads = {
            name: p.grad.detach().clone()
            for name, p in self.model.named_parameters()
            if p.grad is not None
        }
        acc = (logits.argmax(-1) == labels).float().mean().item()
        return {"loss": loss.item(), "accuracy": acc,
                "mean_rate": mean_rate.item(), "method": "bptt"}

    def get_param_gradients(self):
        return dict(self._last_grads)

    @torch.no_grad()
    def evaluate(self, x, labels, mask) -> Dict[str, float]:
        logits, _, _ = self.model.run_bptt(x, mask)
        return {
            "val_loss":     F.cross_entropy(logits, labels).item(),
            "val_accuracy": (logits.argmax(-1) == labels).float().mean().item(),
        }


class TruncBPTTTrainer:
    """Truncated BPTT — states detached at chunk boundaries, loss from last chunk only."""

    def __init__(self, model: SNNClassifier, cfg: TruncBPTTConfig):
        self.model = model
        self.cfg   = cfg
        self.K     = cfg.trunc_len
        self.opt   = torch.optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
        self._last_grads: Dict[str, torch.Tensor] = {}

    def step(
        self, x: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> Dict[str, float]:
        B, T, _ = x.shape
        K = self.K
        device = x.device

        self.opt.zero_grad()
        model = self.model

        states      = model.initial_states(B, device)
        prev_spikes = [
            torch.zeros(B, model.cfg.hidden_dim, device=device)
            for _ in model.lif_layers
        ]
        all_last_spikes: List[torch.Tensor] = []

        chunks           = list(range(0, T, K))
        last_chunk_start = chunks[-1]
        loss             = torch.tensor(0.0, device=device)

        for chunk_start in chunks:
            chunk_end = min(chunk_start + K, T)

            # Detach state at every chunk boundary (truncation)
            states      = [LIFState(s.v.detach(), s.i.detach()) for s in states]
            prev_spikes = [s.detach() for s in prev_spikes]

            chunk_spikes: List[torch.Tensor] = []
            for t_local in range(chunk_end - chunk_start):
                h = x[:, chunk_start + t_local, :]
                for l, layer in enumerate(model.lif_layers):
                    spikes, new_state = layer(
                        h, states[l],
                        prev_spikes=prev_spikes[l] if layer.recurrent else None,
                    )
                    prev_spikes[l] = spikes
                    states[l]      = new_state
                    h = spikes
                chunk_spikes.append(h)

            for s in chunk_spikes:
                all_last_spikes.append(s.detach())

            if chunk_start == last_chunk_start:
                chunk_spike_stack = torch.stack(chunk_spikes, dim=1)  # (B, K', H)
                global_ts  = torch.arange(chunk_start, chunk_end)
                mask_device = mask.to(device)
                local_mask = mask_device[global_ts]  # (K',)

                if local_mask.any():
                    masked      = chunk_spike_stack * local_mask.unsqueeze(0).unsqueeze(-1).float()
                    n_ro        = local_mask.sum().clamp(min=1).float()
                    last_rates  = masked.sum(dim=1) / n_ro  # (B, H)
                    last_logits = model.head(last_rates)
                    loss        = F.cross_entropy(last_logits, labels)

        with torch.no_grad():
            all_stack = torch.stack(all_last_spikes, dim=1)
            mean_rate = all_stack.mean().item()

        if loss.requires_grad:
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), self.cfg.grad_clip)
            self.opt.step()

        self._last_grads = {
            name: p.grad.detach().clone()
            for name, p in model.named_parameters()
            if p.grad is not None
        }

        with torch.no_grad():
            if "last_logits" in dir():
                acc = (last_logits.argmax(-1) == labels).float().mean().item()
            else:
                acc = 0.5

        return {
            "loss": loss.item(),
            "accuracy": acc,
            "mean_rate": mean_rate,
            "method": "tbptt",
            "trunc_len": K,
        }

    def get_param_gradients(self):
        return dict(self._last_grads)

    @torch.no_grad()
    def evaluate(self, x, labels, mask) -> Dict[str, float]:
        logits, _, _ = self.model.run_bptt(x, mask)
        return {
            "val_loss":     F.cross_entropy(logits, labels).item(),
            "val_accuracy": (logits.argmax(-1) == labels).float().mean().item(),
        }
