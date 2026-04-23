"""utils/grad_validation.py — Gradient validation against BPTT ground truth."""
from __future__ import annotations
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
import torch.nn.functional as F
from typing import Dict, List, Optional
from models.snn_model import SNNClassifier


def compute_bptt_gradients(model: SNNClassifier, x: torch.Tensor,
                            labels: torch.Tensor, mask: torch.Tensor
                            ) -> Dict[str, torch.Tensor]:
    model.zero_grad()
    logits, _, _ = model.run_bptt(x, mask)
    F.cross_entropy(logits, labels).backward()
    grads = {name: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p.data)
             for name, p in model.named_parameters()}
    model.zero_grad()
    return grads


def gradient_error(true_g: Dict[str, torch.Tensor],
                   approx_g: Dict[str, torch.Tensor],
                   keys: Optional[List[str]] = None) -> Dict[str, float]:
    if keys is None: keys = list(true_g.keys())
    errors = {}
    for k in keys:
        if k not in approx_g: errors[k] = float('nan'); continue
        gt, ga = true_g[k].float(), approx_g[k].float()
        if gt.shape != ga.shape: errors[k] = float('nan'); continue
        errors[k] = (ga - gt).norm(p='fro').item() / (gt.norm(p='fro').item() + 1e-8)
    valid = [v for v in errors.values() if v == v]
    errors['mean'] = float(sum(valid) / len(valid)) if valid else float('nan')
    return errors


def cosine_similarity_gradients(true_g: Dict[str, torch.Tensor],
                                 approx_g: Dict[str, torch.Tensor],
                                 keys: Optional[List[str]] = None) -> Dict[str, float]:
    if keys is None: keys = list(true_g.keys())
    sims = {}
    for k in keys:
        if k not in approx_g: sims[k] = float('nan'); continue
        gt = true_g[k].float().reshape(-1)
        ga = approx_g[k].float().reshape(-1)
        sims[k] = F.cosine_similarity(gt.unsqueeze(0), ga.unsqueeze(0)).item()
    valid = [v for v in sims.values() if v == v]
    sims['mean'] = float(sum(valid) / len(valid)) if valid else float('nan')
    return sims
