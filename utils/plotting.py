"""utils/plotting.py — Visualization for scientific SNN comparison."""
from __future__ import annotations
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from typing import Dict, List, Optional
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

METHOD_COLORS  = {"bptt": "#2196F3", "tbptt": "#FF9800", "eprop": "#4CAF50", "ctca": "#E91E63"}
METHOD_LABELS  = {"bptt": "BPTT (reference)", "tbptt": f"TBPTT (K=10)",
                  "eprop": "E-prop", "ctca": "CTCA (ours)"}


def _smooth(arr, w=3):
    out = []
    for i in range(len(arr)):
        lo, hi = max(0, i-w//2), min(len(arr), i+w//2+1)
        out.append(sum(arr[lo:hi])/(hi-lo))
    return out


def plot_training_curves(histories, task_name, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Training Curves — {task_name}", fontsize=14, fontweight="bold")
    for method, hist in histories.items():
        c = METHOD_COLORS.get(method, "gray"); lb = METHOD_LABELS.get(method, method)
        if hist.get("train_loss"):     axes[0].plot(_smooth(hist["train_loss"]),     color=c, label=lb, lw=2)
        if hist.get("val_accuracy"):   axes[1].plot(_smooth(hist["val_accuracy"]),   color=c, label=lb, lw=2)
    axes[0].set(title="Training Loss", xlabel="Epoch", ylabel="Cross-Entropy")
    axes[0].legend(); axes[0].grid(True, alpha=0.3)
    axes[1].set(title="Validation Accuracy", xlabel="Epoch", ylabel="Accuracy", ylim=(0, 1.05))
    axes[1].axhline(0.5, color="gray", ls="--", alpha=0.4, label="Chance")
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_gradient_quality(histories, task_name, save_path=None):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Gradient Quality vs BPTT — {task_name}", fontsize=14, fontweight="bold")
    for method, hist in histories.items():
        if method == "bptt": continue
        c = METHOD_COLORS.get(method, "gray"); lb = METHOD_LABELS.get(method, method)
        for ax_i, key in [(0, "grad_error_mean"), (1, "grad_cosine_mean")]:
            vals = [v for v in hist.get(key, []) if v == v]
            if vals: axes[ax_i].plot(vals, color=c, label=lb, lw=2, marker="o", ms=4)
    axes[0].set(title="Gradient Error (lower=better)", xlabel="Check #",
                ylabel="||g_approx - g_true|| / ||g_true||")
    axes[0].legend(); axes[0].grid(True, alpha=0.3); axes[0].set_ylim(bottom=0)
    axes[1].set(title="Gradient Cosine Similarity (higher=better)", xlabel="Check #",
                ylabel="Cosine Similarity", ylim=(-0.1, 1.1))
    axes[1].axhline(0, color="gray", ls="--", alpha=0.4)
    axes[1].legend(); axes[1].grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_final_comparison(summaries, save_path=None):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Final Method Comparison", fontsize=14, fontweight="bold")
    methods = list(summaries.keys())
    colors  = [METHOD_COLORS.get(m, "gray") for m in methods]
    labels  = [METHOD_LABELS.get(m, m) for m in methods]
    def _bar(ax, vals, title, ylabel, hline=None):
        valid = [(l,c,v) for l,c,v in zip(labels,colors,vals) if v==v]
        if valid:
            ls,cs,vs = zip(*valid); ax.bar(ls, vs, color=cs, edgecolor="white", lw=1.5)
        ax.set(title=title, ylabel=ylabel); ax.tick_params(axis="x", rotation=20)
        if hline is not None: ax.axhline(hline, color="red", ls="--", alpha=0.6)
    _bar(axes[0], [summaries[m].get("best_val_acc",float("nan")) for m in methods],
         "Best Validation Accuracy", "Accuracy", hline=0.5); axes[0].set_ylim(0,1.05)
    ge_m = [m for m in methods if m != "bptt"]
    vals = [summaries[m].get("min_grad_error",float("nan")) for m in ge_m]
    valid = [(METHOD_LABELS.get(m,m), METHOD_COLORS.get(m,"gray"), v) for m,v in zip(ge_m,vals) if v==v]
    if valid:
        ls,cs,vs = zip(*valid); axes[1].bar(ls, vs, color=cs, edgecolor="white", lw=1.5)
    axes[1].set(title="Min Gradient Error vs BPTT\n(lower=better)", ylabel="Relative Error")
    axes[1].tick_params(axis="x", rotation=20)
    _bar(axes[2], [summaries[m].get("mean_spike_rate",float("nan")) for m in methods],
         f"Mean Spike Rate\n(target=0.10)", "Rate", hline=0.10)
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_ablation_delay(ablation_results, save_path=None):
    """Plot performance vs delay for all methods."""
    fig, ax = plt.subplots(figsize=(8, 5))
    delays = sorted(ablation_results.keys())
    for method in ["bptt", "tbptt", "eprop", "ctca"]:
        accs = [ablation_results[d].get(method, {}).get("acc", float("nan")) for d in delays]
        valid = [(d, a) for d, a in zip(delays, accs) if a == a]
        if valid:
            ds, acs = zip(*valid)
            ax.plot(ds, acs, color=METHOD_COLORS.get(method, "gray"),
                    label=METHOD_LABELS.get(method, method), lw=2, marker="o")
    ax.axhline(0.5, color="gray", ls="--", alpha=0.5, label="Chance")
    ax.set(title="Performance vs Delay Length", xlabel="Delay (timesteps)",
           ylabel="Validation Accuracy", ylim=(0.4, 1.05))
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_tbptt_window(window_results, delay, save_path=None):
    """Plot TBPTT accuracy vs window size relative to delay."""
    fig, ax = plt.subplots(figsize=(7, 5))
    Ks   = sorted(window_results.keys())
    accs = [window_results[K]["tbptt"]["acc"] for K in Ks]
    bptt_acc = window_results[Ks[0]]["bptt"]["acc"]
    ax.plot(Ks, accs, color=METHOD_COLORS["tbptt"], lw=2, marker="o", label="TBPTT")
    ax.axhline(bptt_acc, color=METHOD_COLORS["bptt"], ls="--", lw=1.5, label="BPTT")
    ax.axhline(0.5, color="gray", ls=":", alpha=0.5, label="Chance")
    ax.axvline(delay, color="red", ls="--", alpha=0.7, label=f"delay={delay}")
    ax.set(title=f"TBPTT: Accuracy vs Window Size (delay={delay})",
           xlabel="Truncation Window K", ylabel="Validation Accuracy", ylim=(0.4, 1.05))
    ax.legend(); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path: fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig
