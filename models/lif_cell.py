"""models/lif_cell.py — Leaky Integrate-and-Fire neuron with SuperSpike surrogate gradient."""
from __future__ import annotations
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import NamedTuple, Optional, Tuple
import torch
import torch.nn as nn


class _SuperSpike(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v_shifted: torch.Tensor, alpha: float) -> torch.Tensor:
        ctx.save_for_backward(v_shifted)
        ctx.alpha = alpha
        return (v_shifted >= 0.0).float()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (v_shifted,) = ctx.saved_tensors
        surrogate = 1.0 / (1.0 + ctx.alpha * v_shifted.abs()) ** 2
        return grad_output * surrogate, None


def super_spike(v: torch.Tensor, v_th: float, alpha: float = 1.0) -> torch.Tensor:
    return _SuperSpike.apply(v - v_th, alpha)


class LIFState(NamedTuple):
    v: torch.Tensor
    i: torch.Tensor

    @staticmethod
    def initial(batch_size: int, n_neurons: int, device: torch.device) -> "LIFState":
        z = torch.zeros(batch_size, n_neurons, device=device)
        return LIFState(v=z, i=z.clone())


def lif_step(
    input_current: torch.Tensor,
    state: LIFState,
    tau_mem: float,
    tau_syn: float,
    v_th: float,
    v_reset: float,
    alpha: float,
    dt: float = 1.0,
) -> Tuple[torch.Tensor, LIFState]:
    new_i = (1.0 - dt / tau_syn) * state.i + input_current
    new_v = (1.0 - dt / tau_mem) * state.v + (dt / tau_mem) * new_i
    spikes = super_spike(new_v, v_th, alpha)
    new_v  = new_v - v_th * spikes  # soft reset
    return spikes, LIFState(v=new_v, i=new_i)


class LIFLayer(nn.Module):
    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        tau_mem: float = 20.0,
        tau_syn: float = 10.0,
        v_th: float = 0.5,
        v_reset: float = 0.0,
        alpha: float = 1.0,
        recurrent: bool = False,
        dt: float = 1.0,
    ):
        super().__init__()
        self.in_dim    = in_dim
        self.out_dim   = out_dim
        self.tau_mem   = tau_mem
        self.tau_syn   = tau_syn
        self.v_th      = v_th
        self.v_reset   = v_reset
        self.alpha     = alpha
        self.recurrent = recurrent
        self.dt        = dt

        self.W = nn.Linear(in_dim, out_dim, bias=False)
        nn.init.xavier_uniform_(self.W.weight, gain=2.0)  # gain=2: targets ~15-25% firing rate at init

        if recurrent:
            self.W_rec = nn.Linear(out_dim, out_dim, bias=False)
            nn.init.orthogonal_(self.W_rec.weight, gain=0.5)  # gain=0.5: stable recurrent dynamics at init
        else:
            self.W_rec = None

    def initial_state(self, batch_size: int, device: torch.device) -> LIFState:
        return LIFState.initial(batch_size, self.out_dim, device)

    def forward(
        self,
        x: torch.Tensor,
        state: LIFState,
        prev_spikes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, LIFState]:
        inp = self.W(x)
        if self.W_rec is not None and prev_spikes is not None:
            inp = inp + self.W_rec(prev_spikes)
        return lif_step(inp, state, self.tau_mem, self.tau_syn,
                        self.v_th, self.v_reset, self.alpha, self.dt)

    def extra_repr(self) -> str:
        return f"in={self.in_dim}, out={self.out_dim}, v_th={self.v_th}, rec={self.recurrent}"
