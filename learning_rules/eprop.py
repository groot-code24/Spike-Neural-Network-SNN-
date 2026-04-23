
from __future__ import annotations
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import math
from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.snn_model import SNNClassifier
from models.lif_cell import LIFState
from configs import EpropConfig


@torch.no_grad()
def _surrogate(v: torch.Tensor, v_th: float, alpha: float) -> torch.Tensor:
    return 1.0 / (1.0 + alpha * (v - v_th).abs()) ** 2


class EligibilityTrace:
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        tau_e: float,
        alpha: float,
        dt: float = 1.0,
        device: torch.device = torch.device("cpu"),
    ):
        self.alpha = alpha
        self.decay = math.exp(-dt / tau_e)
        self.trace_ff  = torch.zeros(out_dim, in_dim,  device=device)
        self.trace_rec: Optional[torch.Tensor] = None

    def allocate_rec(self, out_dim: int, device: torch.device):
        self.trace_rec = torch.zeros(out_dim, out_dim, device=device)

    def reset(self, device: torch.device):
        self.trace_ff.zero_()
        if self.trace_rec is not None:
            self.trace_rec.zero_()

    @torch.no_grad()
    def update(
        self,
        pre: torch.Tensor,
        post_v: torch.Tensor,
        v_th: float,
        pre_rec: Optional[torch.Tensor] = None,
    ):
        h = _surrogate(post_v, v_th, self.alpha)
        self.trace_ff = (
            self.decay * self.trace_ff + (h.unsqueeze(2) * pre.unsqueeze(1)).mean(0)
        )
        if self.trace_rec is not None and pre_rec is not None:
            self.trace_rec = (
                self.decay * self.trace_rec + (h.unsqueeze(2) * pre_rec.unsqueeze(1)).mean(0)
            )


class EpropTrainer:
    def __init__(self, model: SNNClassifier, cfg: EpropConfig):
        self.model  = model
        self.cfg    = cfg
        self.device = next(model.parameters()).device

        L = model.n_layers
        H = model.cfg.hidden_dim
        O = model.cfg.output_dim

        in_dims = [model.cfg.input_dim] + [H] * (L - 1)
        self.traces: List[EligibilityTrace] = []
        for l in range(L):
            tr = EligibilityTrace(
                in_dim=in_dims[l], out_dim=H,
                tau_e=cfg.tau_e, alpha=cfg.alpha_surr,
                device=self.device,
            )
            if model.lif_layers[l].recurrent:
                tr.allocate_rec(H, self.device)
            self.traces.append(tr)

        # Fixed random feedback matrices (orthogonalised for stability)
        self.feedback_B: List[torch.Tensor] = []
        for _ in range(L):
            B_mat = torch.randn(H, O, device=self.device)
            Q, _  = torch.linalg.qr(B_mat)
            r     = min(cfg.n_feedback_cols, O)
            self.feedback_B.append(Q[:, :r] @ Q[:, :r].T @ B_mat)

        self.head_opt = torch.optim.Adam(
            model.head.parameters(),
            lr=cfg.lr * cfg.head_lr_mult,
            weight_decay=cfg.weight_decay,
        )
        self._approx_grads: Dict[str, torch.Tensor] = {}

    def step(
        self, x: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor
    ) -> Dict[str, float]:
        model = self.model
        cfg   = self.cfg
        B, T, _ = x.shape
        L = model.n_layers
        H = model.cfg.hidden_dim

        for tr in self.traces:
            tr.reset(self.device)
        states      = model.initial_states(B, self.device)
        prev_spikes = [torch.zeros(B, H, device=self.device) for _ in range(L)]

        readout_spikes: List[torch.Tensor] = []
        mask_dev = mask.to(self.device)

        with torch.no_grad():
            layer_inputs: List[torch.Tensor]
            for t in range(T):
                h = x[:, t, :]
                layer_inputs = [h]
                for l, layer in enumerate(model.lif_layers):
                    spikes, new_state = layer(
                        h, states[l],
                        prev_spikes=prev_spikes[l] if layer.recurrent else None,
                    )
                    self.traces[l].update(
                        pre=layer_inputs[l],
                        post_v=new_state.v,
                        v_th=model.cfg.v_th,
                        pre_rec=prev_spikes[l] if layer.W_rec is not None else None,
                    )
                    prev_spikes[l] = spikes
                    states[l]      = new_state
                    h = spikes
                    layer_inputs.append(spikes)

                if mask_dev[t]:
                    readout_spikes.append(h)

            if not readout_spikes:
                return {"loss": float("nan"), "accuracy": 0.5,
                        "mean_rate": 0.0, "method": "eprop"}

            readout_stack = torch.stack(readout_spikes, dim=1)  # (B, n_ro, H)
            rate_vec      = readout_stack.mean(dim=1)            # (B, H)
            logits        = model.head(rate_vec)
            probs         = F.softmax(logits.float(), dim=-1)
            one_hot       = F.one_hot(labels.long(), model.cfg.output_dim).float()
            delta_out     = (probs - one_hot).mean(0)            # (O,)

            mean_rate      = rate_vec.mean().item()
            rate_error     = mean_rate - model.cfg.target_rate
            rate_reg_coeff = 2.0 * model.cfg.rate_penalty * rate_error

            for l, (layer, trace, B_l) in enumerate(
                zip(model.lif_layers, self.traces, self.feedback_B)
            ):
                delta_l = B_l @ delta_out
                dW      = -cfg.lr * delta_l.unsqueeze(1) * trace.trace_ff
                dW     += -cfg.lr * rate_reg_coeff * trace.trace_ff

                dW_n = dW.norm().item()
                if dW_n > cfg.grad_clip:
                    dW = dW * (cfg.grad_clip / (dW_n + 1e-8))

                layer.W.weight.data.add_(dW)
                layer.W.weight.data.mul_(1.0 - cfg.weight_decay)
                self._approx_grads[f"lif_layers.{l}.W.weight"] = dW.clone()

                if trace.trace_rec is not None and layer.W_rec is not None:
                    dW_rec  = -cfg.lr * delta_l.unsqueeze(1) * trace.trace_rec
                    dW_rec += -cfg.lr * rate_reg_coeff * trace.trace_rec
                    dW_rn   = dW_rec.norm().item()
                    if dW_rn > cfg.grad_clip:
                        dW_rec = dW_rec * (cfg.grad_clip / (dW_rn + 1e-8))
                    layer.W_rec.weight.data.add_(dW_rec)
                    layer.W_rec.weight.data.mul_(1.0 - cfg.weight_decay)
                    self._approx_grads[f"lif_layers.{l}.W_rec.weight"] = dW_rec.clone()

            ce_loss = F.cross_entropy(logits.float(), labels)
            acc     = (logits.argmax(-1) == labels).float().mean().item()

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
            "method": "eprop",
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
