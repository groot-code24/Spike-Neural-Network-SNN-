"""utils/logging.py — Experiment logging."""
from __future__ import annotations
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import time
from collections import defaultdict
from typing import Dict, List, Optional
import torch


class ExperimentLogger:
    def __init__(self, method_name: str, log_every: int = 10):
        self.method_name = method_name
        self.log_every   = log_every
        self.history: Dict[str, List[float]] = defaultdict(list)
        self._batch_buf:  Dict[str, List[float]] = defaultdict(list)
        self._epoch_start = time.time()
        self._t_start     = time.time()

    def log_batch(self, metrics: Dict[str, float]):
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                self._batch_buf[k].append(float(v))

    def end_epoch(self, val_metrics: Optional[Dict[str, float]] = None):
        for k, vals in self._batch_buf.items():
            if vals: self.history[f"train_{k}"].append(sum(vals) / len(vals))
        self._batch_buf.clear()
        if val_metrics:
            for k, v in val_metrics.items():
                self.history[k].append(float(v))
        self.history["epoch_time_s"].append(time.time() - self._epoch_start)
        self._epoch_start = time.time()

    def log_grad_error(self, grad_metrics: Dict[str, float], epoch: int):
        for k, v in grad_metrics.items():
            self.history[k].append(float(v))

    def print_epoch(self, epoch: int):
        if epoch % self.log_every != 0 and epoch != 0: return
        def last(key): vals = self.history.get(key, []); return f"{vals[-1]:.4f}" if vals else "N/A"
        print(f"[{self.method_name.upper():8s}] Ep {epoch:03d} | "
              f"Loss: {last('train_loss')} | ValAcc: {last('val_accuracy')} | "
              f"Rate: {last('train_mean_rate')} | GradErr: {last('grad_error_mean')} | "
              f"T: {last('epoch_time_s')}s")

    def get_final_summary(self) -> Dict:
        def best(key, hi=True):
            vals = [v for v in self.history.get(key, [float('nan')]) if v == v]
            return (max(vals) if hi else min(vals)) if vals else float('nan')
        return {
            "method":           self.method_name,
            "best_val_acc":     best("val_accuracy", hi=True),
            "final_train_loss": self.history.get("train_loss", [float('nan')])[-1],
            "min_grad_error":   best("grad_error_mean", hi=False),
            "max_grad_cosine":  best("grad_cosine_mean", hi=True),
            "mean_spike_rate":  float(sum(self.history.get("train_mean_rate", [0.1])) /
                                      max(1, len(self.history.get("train_mean_rate", [0.1])))),
            "total_time_s":     time.time() - self._t_start,
        }


def print_comparison_table(summaries: Dict):
    print("\n" + "="*90)
    print("EXPERIMENT RESULTS — SCIENTIFIC COMPARISON")
    print("="*90)
    print(f"{'Method':<12} {'ValAcc':>8} {'Loss':>8} {'GradErr':>9} {'GradCos':>9} {'Rate':>7}")
    print("-"*90)
    for method, s in summaries.items():
        na = lambda v: f"{v:.4f}" if v == v else "  N/A "
        print(f"{s.get('method', method):<12} "
              f"{na(s.get('best_val_acc', float('nan'))):>8} "
              f"{na(s.get('final_train_loss', float('nan'))):>8} "
              f"{na(s.get('min_grad_error', float('nan'))):>9} "
              f"{na(s.get('max_grad_cosine', float('nan'))):>9} "
              f"{na(s.get('mean_spike_rate', float('nan'))):>7}")
    print("="*90)
