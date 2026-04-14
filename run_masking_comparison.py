"""
All masking variants compared: Magma vs SkipUpdate, block vs element.
Fine LR grid, scale=0.01 (noise-floor regime), best-vs-best comparison.

Methods:
  1. AdamW (baseline)
  2. Magma(AdamW, block)
  3. Magma(AdamW, element)
  4. SkipUpdate(AdamW, block)
  5. SkipUpdate(AdamW, element)
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmark import QuadraticBenchmark, DIM
from optimizers import AdamW, Magma, SkipUpdate
from run_experiments import LOG_EVERY, RESULTS_DIR
import os

LRS = np.logspace(-3, -0.3, 20).tolist()
SCALE = 0.01
TAU = 2.0
NUM_SEEDS = 20
N_ITERS = 10000
HESSIAN = "heterogeneous"
K = 3

METHODS = {
    "AdamW":                    lambda lr: AdamW(DIM, lr),
    "Magma(AdamW,block)":       lambda lr: Magma(AdamW(DIM, lr), masking="block", tau=TAU),
    "Magma(AdamW,element)":     lambda lr: Magma(AdamW(DIM, lr), masking="element", tau=TAU),
    "SkipUpdate(AdamW,block)":  lambda lr: SkipUpdate(AdamW(DIM, lr), masking="block"),
    "SkipUpdate(AdamW,element)":lambda lr: SkipUpdate(AdamW(DIM, lr), masking="element"),
}

COLORS = {
    "AdamW":                    "#1f77b4",
    "Magma(AdamW,block)":       "#2ca02c",
    "Magma(AdamW,element)":     "#8cc63f",
    "SkipUpdate(AdamW,block)":  "#d62728",
    "SkipUpdate(AdamW,element)":"#ff7f0e",
}


def ema_smooth(x, alpha=0.02):
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i-1] if np.isfinite(x[i]) else out[i-1]
    return out


def run_one(optimizer, seed):
    bench = QuadraticBenchmark(HESSIAN, seed=seed)
    rng = np.random.default_rng(seed * 10000 + 9999)
    w = np.ones(DIM) * SCALE
    losses = []
    for t in range(N_ITERS):
        if t % LOG_EVERY == 0:
            losses.append(bench.loss(w))
        grad = bench.stochastic_gradient(w, K, rng)
        w = optimizer.step(w, grad, rng)
        if not np.isfinite(w).all() or bench.loss(w) > 1e20:
            remaining = (N_ITERS - t - 1) // LOG_EVERY
            losses.extend([float("nan")] * remaining)
            break
    return np.array(losses)


def run_seeds(make_opt):
    all_losses = []
    for seed in range(NUM_SEEDS):
        losses = run_one(make_opt(), seed)
        all_losses.append(losses)
    max_len = max(len(l) for l in all_losses)
    padded = [np.concatenate([l, np.full(max_len - len(l), np.nan)])
              if len(l) < max_len else l for l in all_losses]
    return np.stack(padded)


def median_final(losses):
    finals = [l[np.isfinite(l)][-1] if np.any(np.isfinite(l)) else np.nan for l in losses]
    return np.nanmedian(finals)


def main():
    t0 = time.time()
    total = len(METHODS) * len(LRS) * NUM_SEEDS
    print(f"Masking comparison: {total} runs ({len(METHODS)} methods × {len(LRS)} LRs × {NUM_SEEDS} seeds)")

    # Run all methods
    results = {}  # results[method_name][lr] = losses array (seeds × steps)
    for name, make_fn in METHODS.items():
        results[name] = {}
        for lr in LRS:
            results[name][lr] = run_seeds(lambda _lr=lr, _fn=make_fn: _fn(_lr))
        print(f"  {name} done ({time.time()-t0:.0f}s)")

    # Find best LR per method
    best = {}
    for name in METHODS:
        finals = {lr: median_final(results[name][lr]) for lr in LRS}
        best_lr = min(finals, key=finals.get)
        best[name] = {"lr": best_lr, "loss": finals[best_lr], "all_finals": finals}

    # Summary table
    print(f"\n{'='*75}")
    print(f"Best-vs-best comparison (scale={SCALE}, {N_ITERS} iters, {NUM_SEEDS} seeds)")
    print(f"{'='*75}")
    print(f"{'Method':<28} {'Best LR':>10} {'Median Loss':>14} {'vs AdamW':>10}")
    print("-" * 65)
    adamw_loss = best["AdamW"]["loss"]
    for name in METHODS:
        ratio = best[name]["loss"] / adamw_loss
        print(f"{name:<28} {best[name]['lr']:>10.5f} {best[name]['loss']:>14.8f} {ratio:>10.3f}")

    # Plot 1: LR sweep (all methods)
    fig, ax = plt.subplots(figsize=(12, 7))
    for name in METHODS:
        lr_arr = np.array(LRS)
        med_arr = np.array([best[name]["all_finals"][lr] for lr in LRS])
        ax.loglog(lr_arr, med_arr, "o-", color=COLORS[name], label=name,
                  linewidth=2, markersize=4)
        ax.axvline(best[name]["lr"], color=COLORS[name], linestyle=":", alpha=0.3)

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Median Final Loss @10k (20 seeds)")
    ax.set_title(f"All Masking Variants: Fine LR Sweep (scale={SCALE}, k={K})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "masking_comparison_lr_sweep.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved: {path}")

    # Plot 2: Convergence curves at each method's best LR
    fig, ax = plt.subplots(figsize=(12, 7))
    for name in METHODS:
        lr = best[name]["lr"]
        losses = results[name][lr]
        iters = np.arange(losses.shape[1]) * LOG_EVERY
        med = ema_smooth(np.nanmedian(losses, axis=0))
        q25 = ema_smooth(np.nanpercentile(losses, 25, axis=0))
        q75 = ema_smooth(np.nanpercentile(losses, 75, axis=0))
        ax.semilogy(iters, med, label=f"{name} (lr={lr:.4f})",
                    color=COLORS[name], linewidth=2)
        ax.fill_between(iters, q25, q75, alpha=0.08, color=COLORS[name])

    ax.set_xlabel("Gradient Steps")
    ax.set_ylabel("Loss (EMA-smoothed median, 20 seeds)")
    ax.set_title(f"Best LR per method (scale={SCALE}, k={K})")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "masking_comparison_curves.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
