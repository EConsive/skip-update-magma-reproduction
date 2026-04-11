"""
Single-block experiment: 3D quadratic with one heterogeneous block.
Tests whether Magma's advantage requires inter-block selectivity.

With 1 block, Magma has a single alignment score and a single coin flip —
no inter-block selectivity possible. If Magma still wins, the advantage
is purely from within-block alignment-based noise suppression.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.linalg import block_diag
import os
import sys

# We need to monkeypatch the block structure before importing optimizers
import optimizers
optimizers.BLOCK_SIZE = 3
optimizers.NUM_BLOCKS = 1
optimizers.BLOCK_SLICES = [slice(0, 3)]

from optimizers import AdamW, Magma
from run_experiments import LOG_EVERY, RESULTS_DIR


DIM = 3
EIGS = [1, 99, 4998]  # one heterogeneous block
SCALE = 0.01
TAU = 2.0
NUM_SEEDS = 20
N_ITERS = 10000
K = 1  # 1/3 of rows, same fraction as k=3 out of 9

LRS = np.logspace(-3, -0.3, 20).tolist()


def random_orthogonal(n, rng):
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q


class SingleBlockBenchmark:
    def __init__(self, seed=0):
        rng = np.random.default_rng(seed)
        Q = random_orthogonal(3, rng)
        self.H = Q @ np.diag(EIGS) @ Q.T
        rng2 = np.random.default_rng(seed)
        Q2 = random_orthogonal(3, rng2)
        self.X = Q2 @ np.diag(np.sqrt(EIGS)) @ Q2.T

    def loss(self, w):
        return 0.5 * w @ self.H @ w

    def stochastic_gradient(self, w, k, rng):
        n = self.X.shape[0]
        indices = rng.choice(n, size=k, replace=False)
        X_S = self.X[indices]
        return (n / k) * (X_S.T @ X_S @ w)


def ema_smooth(x, alpha=0.02):
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i-1] if np.isfinite(x[i]) else out[i-1]
    return out


def run_one(optimizer, seed):
    bench = SingleBlockBenchmark(seed=seed)
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

    METHODS = {
        "AdamW":              lambda lr: AdamW(DIM, lr),
        "Magma(AdamW,block)": lambda lr: Magma(AdamW(DIM, lr), masking="block", tau=TAU),
    }
    COLORS = {"AdamW": "#1f77b4", "Magma(AdamW,block)": "#2ca02c"}

    total = len(METHODS) * len(LRS) * NUM_SEEDS
    print(f"Single-block experiment: {total} runs")
    print(f"  dim={DIM}, eigs={EIGS}, scale={SCALE}, k={K}, {N_ITERS} iters")

    results = {}
    for name, make_fn in METHODS.items():
        results[name] = {}
        for lr in LRS:
            results[name][lr] = run_seeds(lambda _lr=lr, _fn=make_fn: _fn(_lr))
        print(f"  {name} done ({time.time()-t0:.0f}s)")

    # Best LR per method
    best = {}
    for name in METHODS:
        finals = {lr: median_final(results[name][lr]) for lr in LRS}
        best_lr = min(finals, key=finals.get)
        best[name] = {"lr": best_lr, "loss": finals[best_lr], "all_finals": finals}

    adamw_loss = best["AdamW"]["loss"]
    print(f"\n{'='*70}")
    print(f"Single-block best-vs-best (dim=3, scale={SCALE}, k={K})")
    print(f"{'='*70}")
    for name in METHODS:
        ratio = best[name]["loss"] / adamw_loss
        print(f"  {name:<25} lr={best[name]['lr']:.5f}  loss={best[name]['loss']:.8f}  vs AdamW={ratio:.3f}")

    # Plot 1: LR sweep
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in METHODS:
        lr_arr = np.array(LRS)
        med_arr = np.array([best[name]["all_finals"][lr] for lr in LRS])
        ax.loglog(lr_arr, med_arr, "o-", color=COLORS[name], label=name, linewidth=2, markersize=5)
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Median Final Loss @10k (20 seeds)")
    ax.set_title(f"Single Block (dim=3, eigs={EIGS}, scale={SCALE}, k={K})")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "single_block_lr_sweep.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved: {path}")

    # Plot 2: Convergence at best LRs
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in METHODS:
        lr = best[name]["lr"]
        losses = results[name][lr]
        iters = np.arange(losses.shape[1]) * LOG_EVERY
        med = ema_smooth(np.nanmedian(losses, axis=0))
        q25 = ema_smooth(np.nanpercentile(losses, 25, axis=0))
        q75 = ema_smooth(np.nanpercentile(losses, 75, axis=0))
        ax.semilogy(iters, med, label=f"{name} (lr={lr:.4f})", color=COLORS[name], linewidth=2)
        ax.fill_between(iters, q25, q75, alpha=0.12, color=COLORS[name])
    ax.set_xlabel("Gradient Steps")
    ax.set_ylabel("Loss (EMA-smoothed median)")
    ax.set_title(f"Single Block: Best LR per method (dim=3, k={K})")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "single_block_curves.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    print(f"Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
