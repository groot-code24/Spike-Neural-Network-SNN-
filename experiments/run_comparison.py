
from __future__ import annotations
import sys, os
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

import copy, gc, json, time
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from configs import ExperimentConfig, SNNConfig
from models.snn_model import SNNClassifier, build_model
from learning_rules.bptt  import BPTTTrainer, TruncBPTTTrainer
from learning_rules.eprop import EpropTrainer
from learning_rules.ctca  import CTCATrainer
from experiments.tasks import (build_dataloaders, get_task_description,
                                validate_task_scientifically)
from utils.logging import ExperimentLogger, print_comparison_table
from utils.grad_validation import (compute_bptt_gradients, gradient_error,
                                   cosine_similarity_gradients)


def build_trainer(method: str, model: SNNClassifier, cfg: ExperimentConfig):
    if method == "bptt":   return BPTTTrainer(model, cfg.bptt)
    if method == "tbptt":  return TruncBPTTTrainer(model, cfg.tbptt)
    if method == "eprop":  return EpropTrainer(model, cfg.eprop)
    if method == "ctca":   return CTCATrainer(model, cfg.ctca)
    raise ValueError(f"Unknown method: {method}")


def _n_epochs(method: str, cfg: ExperimentConfig) -> int:
    return {"bptt": cfg.bptt.epochs, "tbptt": cfg.tbptt.epochs,
            "eprop": cfg.eprop.epochs, "ctca": cfg.ctca.epochs}[method]


def _validate_config(cfg: ExperimentConfig):
    if cfg.snn.T != cfg.task.T:
        raise ValueError(f"cfg.snn.T ({cfg.snn.T}) != cfg.task.T ({cfg.task.T})")
    if cfg.snn.input_dim != cfg.task.input_dim:
        raise ValueError(f"input_dim mismatch")
    if cfg.snn.readout_len >= cfg.snn.T:
        raise ValueError(f"readout_len must be < T")


def run_single_method(method, cfg, train_loader, val_loader, seed,
                      gv_x=None, gv_y=None, gv_mask=None):
    torch.manual_seed(seed)
    device = torch.device(cfg.device)
    model   = build_model(cfg.snn).to(device)
    trainer = build_trainer(method, model, cfg)
    logger  = ExperimentLogger(method, log_every=cfg.log_every)
    n_ep    = _n_epochs(method, cfg)

    print(f"\n{'─'*60}")
    print(f"Method: {method.upper():8s} | Params: {model.count_parameters():,} | Epochs: {n_ep}")
    print(f"{'─'*60}")

    for epoch in range(n_ep):
        model.train()
        for batch in train_loader:
            x_b, y_b, mask_b = batch
            x_b, y_b = x_b.to(device), y_b.to(device)
            # Use first sample's mask (all masks are same for fixed delay)
            mask_b = mask_b[0].to(device)
            metrics = trainer.step(x_b, y_b, mask_b)
            logger.log_batch(metrics)

        model.eval()
        val_accs, val_losses = [], []
        for x_v, y_v, mask_v in val_loader:
            x_v, y_v = x_v.to(device), y_v.to(device)
            mask_v = mask_v[0].to(device)
            vm = trainer.evaluate(x_v, y_v, mask_v)
            val_accs.append(vm["val_accuracy"])
            val_losses.append(vm["val_loss"])

        val_m = {
            "val_accuracy": sum(val_accs)  / len(val_accs),
            "val_loss":     sum(val_losses) / len(val_losses),
        }

        # Gradient validation
        if method != "bptt" and epoch % cfg.log_every == 0 and gv_x is not None:
            approx_g = trainer.get_param_gradients()
            if approx_g:
                _m = copy.deepcopy(model)
                true_g = compute_bptt_gradients(
                    _m, gv_x[:16].to(device), gv_y[:16].to(device),
                    gv_mask[:16][0].to(device))
                del _m; gc.collect()
                common = [k for k in true_g if k in approx_g]
                if common:
                    ge  = gradient_error(true_g, approx_g, keys=common)
                    gcs = cosine_similarity_gradients(true_g, approx_g, keys=common)
                    val_m["grad_error_mean"]  = ge.get("mean", float("nan"))
                    val_m["grad_cosine_mean"] = gcs.get("mean", float("nan"))
                    logger.log_grad_error(val_m, epoch)

        logger.end_epoch(val_m)
        logger.print_epoch(epoch)

    summary = logger.get_final_summary()
    print(f"✓ {method.upper()} done  |  best_val_acc = {summary['best_val_acc']:.4f}  "
          f"|  rate = {summary['mean_spike_rate']:.4f}")
    return logger, summary


def run_full_comparison(cfg: ExperimentConfig) -> Dict[str, Dict]:
    _validate_config(cfg)

    # Scientific task validation
    print("\n── Task Scientific Validation ──")
    task_checks = validate_task_scientifically(cfg.task)
    for k, v in task_checks.items():
        sym = "✓" if v is True else ("✗" if v is False else f"{v:.4f}")
        print(f"  {sym}  {k}")
    if not task_checks.get('memoryless_ok', True):
        raise ValueError("Task fails memoryless check — shortcut learning possible!")
    if not task_checks.get('no_leakage', True):
        raise ValueError("Task has label leakage in readout window!")
    if not task_checks.get('readout_blank', True):
        raise ValueError("Readout window has signal — masks are wrong!")

    device = torch.device(
        "cuda" if (torch.cuda.is_available() and cfg.device in ("auto", "cuda")) else "cpu")
    cfg.device = str(device)

    print(f"\n{'═'*70}")
    print("SNN TEMPORAL CREDIT ASSIGNMENT — SCIENTIFIC COMPARISON v2")
    print(f"{'═'*70}")
    print(f"Device: {device}  |  Task: {get_task_description(cfg.task)}")
    print()

    train_loader, val_loader, _ = build_dataloaders(cfg.task, cfg.seed)
    gv_batch = next(iter(val_loader))
    gv_x, gv_y, gv_mask = gv_batch

    results, histories = {}, {}
    for method in cfg.methods:
        logger, summary = run_single_method(
            method, cfg, train_loader, val_loader,
            seed=cfg.seed, gv_x=gv_x, gv_y=gv_y, gv_mask=gv_mask)
        results[method]   = summary
        histories[method] = dict(logger.history)

    print_comparison_table(results)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    task = cfg.task.task_name
    with open(out_dir / f"results_{task}.json", "w") as f:
        json.dump(results, f, indent=2, default=str)
    with open(out_dir / f"history_{task}.json", "w") as f:
        json.dump(histories, f, indent=2, default=float)

    return results


def sanity_check(cfg: Optional[ExperimentConfig] = None) -> bool:
    if cfg is None:
        cfg = ExperimentConfig()
        cfg.snn.hidden_dim  = 16; cfg.snn.T = 20
        cfg.snn.readout_len = 5
        cfg.task.T          = 20; cfg.task.cue_duration = 3
        cfg.task.delay      = 15; cfg.task.n_train = 32
        cfg.task.n_val      = 16; cfg.task.n_test  = 16
        cfg.task.batch_size = 4;  cfg.task.randomize_delay = False

    passed = failed = 0
    def chk(name, fn):
        nonlocal passed, failed
        try: fn(); print(f"  ✓ {name}"); passed += 1
        except Exception as e: print(f"  ✗ {name}: {e}"); failed += 1

    print("\n── Sanity Check ──")
    _validate_config(cfg)
    model = build_model(cfg.snn)
    chk("Model builds", lambda: None)

    train_l, val_l, _ = build_dataloaders(cfg.task, seed=0)
    xb, yb, mb = next(iter(train_l))
    mb0 = mb[0]
    chk("DataLoader returns (x,y,mask)", lambda: assert_(mb.shape[-1] == cfg.snn.T))
    chk("Mask has readout steps", lambda: assert_(mb0.sum() > 0))

    logits, rates, _ = model.run_bptt(xb, mb0)
    chk("run_bptt shape", lambda: assert_(logits.shape == (cfg.task.batch_size, cfg.snn.output_dim)))
    import torch.nn.functional as _F
    _F.cross_entropy(logits, yb).backward()
    chk("BPTT backward", lambda: None)

    for method in ["bptt", "tbptt", "eprop", "ctca"]:
        m2 = build_model(cfg.snn)
        tr = build_trainer(method, m2, cfg)
        mt = tr.step(xb, yb, mb0)
        chk(f"{method}.step()", lambda mt=mt: assert_(0 <= mt.get("accuracy", 0) <= 1))
        ev = tr.evaluate(xb, yb, mb0)
        chk(f"{method}.evaluate()", lambda ev=ev: assert_("val_accuracy" in ev))

    print(f"\nSanity: {passed} passed, {failed} failed → {'✅ OK' if failed==0 else '❌ FAILED'}")
    return failed == 0


def assert_(cond, msg=""):
    if not cond: raise AssertionError(msg)
