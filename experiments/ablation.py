
from __future__ import annotations
import sys
import os
import json
import copy

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import torch
from typing import Dict, List
from configs import ExperimentConfig
from experiments.run_comparison import run_single_method, _validate_config
from experiments.tasks import build_dataloaders
from models.snn_model import build_model


def _base_cfg(
    T: int = 60,
    delay: int = 50,
    K: int = 10,
    distractor: float = 0.10,
    recurrent: bool = True,
) -> ExperimentConfig:
    cfg = ExperimentConfig()
    cfg.snn.T           = T
    cfg.snn.hidden_dim  = 32
    cfg.snn.n_layers    = 2
    cfg.snn.readout_len = max(5, T - delay)
    cfg.snn.tau_mem     = 20.0
    cfg.snn.tau_syn     = 10.0
    cfg.snn.v_th        = 0.5
    cfg.snn.alpha_surr  = 1.0
    cfg.snn.target_rate  = 0.10
    cfg.snn.rate_penalty = 0.005
    if not recurrent:
        cfg.snn.n_layers = 1
    cfg.task.T               = T
    cfg.task.cue_duration    = 5
    cfg.task.delay           = delay
    cfg.task.distractor_rate = distractor
    cfg.task.randomize_delay = False
    cfg.task.n_train         = 200
    cfg.task.n_val           = 80
    cfg.task.batch_size      = 32
    cfg.bptt.epochs          = 8
    cfg.bptt.lr              = 5e-4
    cfg.tbptt.epochs         = 8
    cfg.tbptt.trunc_len      = K
    cfg.tbptt.lr             = 5e-4
    cfg.eprop.epochs         = 8
    cfg.eprop.lr             = 2e-3
    cfg.eprop.tau_e          = 30.0
    cfg.eprop.alpha_surr     = 1.0
    cfg.eprop.head_lr_mult   = 5.0
    cfg.eprop.n_feedback_cols = 8
    cfg.ctca.epochs          = 8
    cfg.ctca.lr              = 2e-3
    cfg.ctca.trunc_len       = T
    cfg.ctca.influence_decay = 0.99
    cfg.ctca.head_lr_mult    = 5.0
    cfg.log_every            = 100
    cfg.device               = "cpu"
    cfg.seed                 = 42
    return cfg


def _run_pair(cfg: ExperimentConfig, methods: List[str] = None) -> Dict:
    if methods is None:
        methods = ["bptt", "tbptt", "eprop", "ctca"]
    _validate_config(cfg)
    train_l, val_l, _ = build_dataloaders(cfg.task, cfg.seed)
    gv_x, gv_y, gv_m  = next(iter(val_l))
    results = {}
    for m in methods:
        _, s = run_single_method(
            m, cfg, train_l, val_l,
            seed=cfg.seed, gv_x=gv_x, gv_y=gv_y, gv_mask=gv_m,
        )
        results[m] = {
            "acc":  s["best_val_acc"],
            "rate": s["mean_spike_rate"],
            "cos":  s.get("max_grad_cosine", float("nan")),
        }
    return results


def run_delay_sweep(delays: List[int] = None) -> Dict:
    if delays is None:
        delays = [10, 20, 30, 50]
    results = {}
    print("\n── Delay Sweep ──")
    for d in delays:
        T   = d + 15
        cfg = _base_cfg(T=T, delay=d)
        r   = _run_pair(cfg)
        results[d] = r
        print(
            f"  delay={d:3d}  bptt={r['bptt']['acc']:.3f}  tbptt={r['tbptt']['acc']:.3f}  "
            f"eprop={r['eprop']['acc']:.3f}  ctca={r['ctca']['acc']:.3f}"
        )
    return results


def run_tbptt_window_sweep(Ks: List[int] = None, delay: int = 40) -> Dict:
    if Ks is None:
        Ks = [5, 10, 20, 50]
    T       = delay + 15
    results = {}
    print(f"\n── TBPTT Window Sweep (delay={delay}) ──")
    for K in Ks:
        cfg = _base_cfg(T=T, delay=delay, K=K)
        r   = _run_pair(cfg, methods=["bptt", "tbptt"])
        results[K] = r
        verdict = "FAILS" if r["tbptt"]["acc"] < 0.65 else "LEAKS"
        print(
            f"  K={K:3d}  bptt={r['bptt']['acc']:.3f}  tbptt={r['tbptt']['acc']:.3f} "
            f"({'K>=delay' if K >= delay else verdict})"
        )
    return results


def run_noise_sweep(rates: List[float] = None) -> Dict:
    if rates is None:
        rates = [0.05, 0.10, 0.20]
    results = {}
    print("\n── Distractor Noise Sweep ──")
    for dr in rates:
        cfg = _base_cfg(distractor=dr)
        r   = _run_pair(cfg)
        results[dr] = r
        print(
            f"  distractor={dr:.2f}  bptt={r['bptt']['acc']:.3f}  "
            f"eprop={r['eprop']['acc']:.3f}  ctca={r['ctca']['acc']:.3f}"
        )
    return results


def run_recurrence_ablation() -> Dict:
    results = {}
    print("\n── Recurrence Ablation ──")
    for rec_label, rec in [("with_recurrence", True), ("no_recurrence", False)]:
        cfg = _base_cfg(recurrent=rec)
        r   = _run_pair(cfg, methods=["bptt", "ctca"])
        results[rec_label] = r
        print(f"  {rec_label}: bptt={r['bptt']['acc']:.3f}  ctca={r['ctca']['acc']:.3f}")
    return results


def run_all_ablations(output_dir: str = "/tmp/snn_v2") -> Dict:
    os.makedirs(output_dir, exist_ok=True)
    all_ablations = {
        "delay_sweep":  run_delay_sweep(),
        "tbptt_window": run_tbptt_window_sweep(),
        "noise_sweep":  run_noise_sweep(),
        "recurrence":   run_recurrence_ablation(),
    }

    def _convert(obj):
        if isinstance(obj, float):
            return obj
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        return str(obj)

    with open(os.path.join(output_dir, "ablations.json"), "w") as f:
        json.dump(all_ablations, f, indent=2, default=_convert)

    print(f"\nAblation results saved to {output_dir}/ablations.json")
    return all_ablations
