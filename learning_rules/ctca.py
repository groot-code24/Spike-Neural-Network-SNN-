
from __future__ import annotations
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.snn_model import SNNClassifier
from models.lif_cell import LIFState
from configs import CTCAConfig


@torch.no_grad()
def _surrogate(v: torch.Tensor, v_th: float, alpha: float) -> torch.Tensor:
    return 1.0 / (1.0 + alpha * (v - v_th).abs()) ** 2


class CTCATrainer:
    def __init__(self, model: SNNClassifier, cfg: CTCAConfig):
        self.model  = model
        self.cfg    = cfg
        self.device = next(model.parameters()).device

        self.head_opt = torch.optim.Adam(
            model.head.parameters(),
            lr=cfg.lr * cfg.head_lr_mult,
            weight_decay=cfg.weight_decay,
        )
        self._approx_grads: Dict[str, torch.Tensor] = {}

    def step(
        self,
        x: torch.Tensor,
        labels: torch.Tensor,
        mask: torch.Tensor,
    ) -> Dict[str, float]:
        model = self.model
        cfg   = self.cfg
        B, T, _ = x.shape
        L    = model.n_layers
        H    = model.cfg.hidden_dim
        v_th = model.cfg.v_th

        mask_dev = mask.to(self.device)

        # ── Forward pass ─────────────────────────────────────────────────
        with torch.no_grad():
            all_spikes: List[List[torch.Tensor]] = [[] for _ in range(L)]
            all_vmem:   List[List[torch.Tensor]] = [[] for _ in range(L)]
            all_inputs: List[List[torch.Tensor]] = [[] for _ in range(L)]

            states      = model.initial_states(B, self.device)
            prev_spikes = [torch.zeros(B, H, device=self.device) for _ in range(L)]

            for t in range(T):
                h = x[:, t, :]
                for l, layer in enumerate(model.lif_layers):
                    all_inputs[l].append(h.clone())
                    spikes, new_state = layer(
                        h, states[l],
                        prev_spikes=prev_spikes[l] if layer.recurrent else None,
                    )
                    all_spikes[l].append(spikes)
                    all_vmem[l].append(new_state.v)
                    prev_spikes[l] = spikes
                    states[l]      = new_state
                    h = spikes

            # Readout-masked firing rate
            last_stack = torch.stack(all_spikes[-1], dim=1)  # (B, T, H)
            masked     = last_stack * mask_dev.unsqueeze(0).unsqueeze(-1).float()
            n_ro       = mask_dev.sum().clamp(min=1).float()
            rate_vec   = masked.sum(dim=1) / n_ro            # (B, H)
            logits     = model.head(rate_vec)
            probs      = F.softmax(logits.float(), dim=-1)

        # ── Output error (softmax – one-hot) ────────────────────────────
        with torch.no_grad():
            one_hot   = F.one_hot(labels.long(), model.cfg.output_dim).float()
            delta_out = probs - one_hot  # (B, O)

        # ── Manual backward sweep ────────────────────────────────────────
        trunc   = min(cfg.trunc_len, T)
        t_start = max(0, T - trunc)
        gamma   = cfg.influence_decay

        with torch.no_grad():
            dW_accum  = [torch.zeros_like(model.lif_layers[l].W.weight)     for l in range(L)]
            act_accum = [torch.zeros_like(model.lif_layers[l].W.weight)     for l in range(L)]
            dW_rec_accum  = [
                torch.zeros_like(model.lif_layers[l].W_rec.weight)
                if model.lif_layers[l].W_rec is not None else None
                for l in range(L)
            ]
            act_rec_accum = [
                torch.zeros_like(model.lif_layers[l].W_rec.weight)
                if model.lif_layers[l].W_rec is not None else None
                for l in range(L)
            ]

            W_head        = model.head.weight          # (O, H)
            delta_last_in = delta_out @ W_head         # (B, H)

            causal_inf = [torch.zeros(B, H, device=self.device) for _ in range(L)]

            # Only readout-window timesteps contribute error to the backward sweep.
            # Non-readout timesteps set ro_scale=0 so their errors do not propagate.
            n_ro_inv = 1.0 / mask_dev.sum().clamp(min=1).float()

            for t in range(T - 1, t_start - 1, -1):
                ro_scale = n_ro_inv if mask_dev[t] else torch.tensor(0.0, device=self.device)

                h_last    = _surrogate(all_vmem[-1][t], v_th, cfg.alpha_surr)
                delta_last = h_last * delta_last_in * ro_scale

                causal_inf[-1] = gamma * causal_inf[-1] + delta_last

                pre = all_inputs[-1][t]
                dW_accum[-1]  += (causal_inf[-1].unsqueeze(2) * pre.unsqueeze(1)).mean(0)
                act_accum[-1] += (h_last.unsqueeze(2)          * pre.unsqueeze(1)).mean(0)

                if dW_rec_accum[-1] is not None and t > 0:
                    pre_rec = all_spikes[-1][t - 1]
                    dW_rec_accum[-1]  += (causal_inf[-1].unsqueeze(2) * pre_rec.unsqueeze(1)).mean(0)
                    act_rec_accum[-1] += (h_last.unsqueeze(2)          * pre_rec.unsqueeze(1)).mean(0)

                delta_upstream = causal_inf[-1]
                for l in range(L - 2, -1, -1):
                    W_next  = model.lif_layers[l + 1].W.weight  # true W^T feedback
                    h_l     = _surrogate(all_vmem[l][t], v_th, cfg.alpha_surr)
                    delta_l = h_l * (delta_upstream @ W_next)

                    causal_inf[l] = gamma * causal_inf[l] + delta_l

                    pre_l = all_inputs[l][t]
                    dW_accum[l]  += (causal_inf[l].unsqueeze(2) * pre_l.unsqueeze(1)).mean(0)
                    act_accum[l] += (h_l.unsqueeze(2)            * pre_l.unsqueeze(1)).mean(0)

                    if dW_rec_accum[l] is not None and t > 0:
                        pr = all_spikes[l][t - 1]
                        dW_rec_accum[l]  += (causal_inf[l].unsqueeze(2) * pr.unsqueeze(1)).mean(0)
                        act_rec_accum[l] += (h_l.unsqueeze(2)            * pr.unsqueeze(1)).mean(0)

                    delta_upstream = causal_inf[l]

            mean_rate      = rate_vec.mean().item()
            rate_error     = mean_rate - model.cfg.target_rate
            rate_reg_coeff = 2.0 * model.cfg.rate_penalty * rate_error
            scale          = -cfg.lr / max(trunc, 1)

            for l, layer in enumerate(model.lif_layers):
                dW   = scale * dW_accum[l] + (-cfg.lr * rate_reg_coeff * act_accum[l] / max(trunc, 1))
                dW_n = dW.norm().item()
                if dW_n > cfg.grad_clip:
                    dW = dW * (cfg.grad_clip / (dW_n + 1e-8))
                layer.W.weight.data.add_(dW)
                layer.W.weight.data.mul_(1.0 - cfg.weight_decay)
                self._approx_grads[f"lif_layers.{l}.W.weight"] = dW.clone()

                if dW_rec_accum[l] is not None and layer.W_rec is not None:
                    dW_r = scale * dW_rec_accum[l]
                    if act_rec_accum[l] is not None:
                        dW_r += -cfg.lr * rate_reg_coeff * act_rec_accum[l] / max(trunc, 1)
                    dW_rn = dW_r.norm().item()
                    if dW_rn > cfg.grad_clip:
                        dW_r = dW_r * (cfg.grad_clip / (dW_rn + 1e-8))
                    layer.W_rec.weight.data.add_(dW_r)
                    layer.W_rec.weight.data.mul_(1.0 - cfg.weight_decay)
                    self._approx_grads[f"lif_layers.{l}.W_rec.weight"] = dW_r.clone()

            ce_loss = F.cross_entropy(logits.float(), labels)
            acc     = (logits.argmax(-1) == labels).float().mean().item()

        # ── Head update via autograd ─────────────────────────────────────
        self.head_opt.zero_grad()
        with torch.enable_grad():
            logits2 = model.head(rate_vec.detach())
            F.cross_entropy(logits2.float(), labels).backward()
        nn.utils.clip_grad_norm_(model.head.parameters(), cfg.grad_clip)
        self.head_opt.step()
        model.head.weight.data.mul_(1.0 - cfg.weight_decay)
        if model.head.weight.grad is not None:
            self._approx_grads["head.weight"] = model.head.weight.grad.detach().clone()

        return {
            "loss": ce_loss.item(),
            "accuracy": acc,
            "mean_rate": mean_rate,
            "method": "ctca",
            "trunc_len": cfg.trunc_len,
        }

    def get_param_gradients(self):
        return dict(self._approx_grads)

    @torch.no_grad()
    def evaluate(self, x, labels, mask) -> Dict[str, float]:
        out = self.model.run_no_grad(x, mask)
        return {
            "val_loss":     F.cross_entropy(out["logits"].float(), labels).item(),
            "val_accuracy": (out["logits"].argmax(-1) == labels).float().mean().item(),
        }
