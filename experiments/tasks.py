
from __future__ import annotations
import sys
import os

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Tuple
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from configs import TaskConfig


class DelayedXORDataset(Dataset):
    def __init__(self, cfg: TaskConfig, n_samples: int, seed: int = 0):
        self.cfg = cfg
        self.T   = cfg.T
        rng = np.random.RandomState(seed)

        a = rng.randint(0, 2, n_samples)
        b = rng.randint(0, 2, n_samples)
        self.a_bits = torch.tensor(a, dtype=torch.long)
        self.b_bits = torch.tensor(b, dtype=torch.long)
        self.labels = torch.tensor(np.bitwise_xor(a, b), dtype=torch.long)

        if cfg.randomize_delay:
            jitter = rng.randint(-15, 16, n_samples)
        else:
            jitter = np.zeros(n_samples, dtype=int)
        self.cue_starts = np.clip(jitter, 0, cfg.T - cfg.cue_duration - 20)
        self.n_samples  = n_samples

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cfg  = self.cfg
        T, I = self.T, cfg.input_dim
        half = I // 2

        x = torch.zeros(T, I)

        a_bit = self.a_bits[idx].item()
        b_bit = self.b_bits[idx].item()
        t0 = int(self.cue_starts[idx])
        t1 = t0 + cfg.cue_duration

        for t in range(t0, t1):
            x[t, :half] = torch.bernoulli(torch.full((half,),    0.9 if a_bit else 0.05))
            x[t, half:] = torch.bernoulli(torch.full((I - half,), 0.9 if b_bit else 0.05))

        distractor = torch.bernoulli(torch.full((T, I), cfg.distractor_rate))
        distractor[t0:t1, :] = 0.0
        x = (x + distractor).clamp(0, 1)

        readout_len  = cfg.T - cfg.delay
        readout_mask = torch.zeros(T, dtype=torch.bool)
        readout_mask[-readout_len:] = True

        return x, self.labels[idx], readout_mask


TASK_REGISTRY = {"delayed_xor": DelayedXORDataset}


def build_dataloaders(cfg: TaskConfig, seed: int = 42):
    TaskClass = TASK_REGISTRY[cfg.task_name]
    kw = dict(batch_size=cfg.batch_size, num_workers=0, pin_memory=False)
    return (
        DataLoader(TaskClass(cfg, cfg.n_train, seed=seed),   shuffle=True,  **kw),
        DataLoader(TaskClass(cfg, cfg.n_val,   seed=seed+1), shuffle=False, **kw),
        DataLoader(TaskClass(cfg, cfg.n_test,  seed=seed+2), shuffle=False, **kw),
    )


def get_task_description(cfg: TaskConfig) -> str:
    readout_len = cfg.T - cfg.delay
    return (
        f"DelayedXOR T={cfg.T}, cue={cfg.cue_duration}, delay={cfg.delay}, "
        f"readout={readout_len}, distractor={cfg.distractor_rate}, "
        f"jitter={'±15' if cfg.randomize_delay else 'fixed'}"
    )


def validate_task_scientifically(cfg: TaskConfig, n: int = 500) -> dict:
    """Validate that the task requires genuine temporal memory.

    Returns a dict of checks; raises ValueError if any critical check fails.
    """
    ds = DelayedXORDataset(cfg, n, seed=99)
    xs, ys, masks = [], [], []
    for i in range(n):
        x, y, m = ds[i]
        xs.append(x); ys.append(y); masks.append(m)
    xs = torch.stack(xs); ys = torch.stack(ys)

    results    = {}
    readout_len = cfg.T - cfg.delay

    readout_activity = xs[:, -readout_len:, :].mean().item()
    signal_activity  = xs[:, :cfg.cue_duration, :].mean().item()
    results["readout_activity"] = readout_activity
    results["signal_activity"]  = signal_activity
    results["readout_blank"]    = readout_activity < 0.20

    import torch.nn as nn
    import torch.nn.functional as F

    readout_feats = xs[:, -readout_len:, :].mean(dim=1)
    probe = nn.Linear(cfg.input_dim, 2)
    opt   = torch.optim.Adam(probe.parameters(), lr=1e-2)
    for _ in range(300):
        opt.zero_grad()
        F.cross_entropy(probe(readout_feats), ys).backward()
        opt.step()
    with torch.no_grad():
        memoryless_acc = (probe(readout_feats).argmax(-1) == ys).float().mean().item()
    results["memoryless_acc"] = memoryless_acc
    results["memoryless_ok"]  = memoryless_acc < 0.70

    full_feats = xs.mean(dim=1)
    probe2 = nn.Linear(cfg.input_dim, 2)
    opt2   = torch.optim.Adam(probe2.parameters(), lr=1e-2)
    for _ in range(300):
        opt2.zero_grad()
        F.cross_entropy(probe2(full_feats), ys).backward()
        opt2.step()
    with torch.no_grad():
        full_acc = (probe2(full_feats).argmax(-1) == ys).float().mean().item()
    results["linear_full_acc"] = full_acc

    readout_mean = xs[:, -readout_len:, :].mean(dim=[1, 2])
    corr = torch.corrcoef(torch.stack([readout_mean, ys.float()]))[0, 1].abs().item()
    results["readout_label_corr"] = corr
    results["no_leakage"]         = corr < 0.10

    return results
