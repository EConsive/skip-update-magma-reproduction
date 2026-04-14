"""
Definitive test: Magma(AdamW) vs AdamW with fine LR grid at scale=0.01.
Each method picks its best LR independently. 20 log-spaced LRs.
"""

import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from benchmark import QuadraticBenchmark, DIM
from optimizers import AdamW, Magma
from run_experiments import LOG_EVERY, RESULTS_DIR
import os

LRS = np.logspace(-3, -0.3, 20).tolist()  # 0.001 to ~0.5, 20 points
SCALE = 0.01
TAU = 2.0
NUM_SEEDS = 20
N_ITERS = 10000
HESSIAN = "heterogeneous"
K = 3


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
    total = 2 * len(LRS) * NUM_SEEDS
    print(f"Fine LR sweep: {total} runs ({len(LRS)} LRs × {NUM_SEEDS} seeds × 2)")
    print(f"  scale={SCALE}, tau={TAU}, k={K}, {N_ITERS} iters")

    adamw_results = {}
    magma_results = {}

    for lr in LRS:
        adamw_results[lr] = run_seeds(lambda _lr=lr: AdamW(DIM, _lr))
        magma_results[lr] = run_seeds(
            lambda _lr=lr: Magma(AdamW(DIM, _lr), masking="block", tau=TAU))

    elapsed = time.time() - t0
    print(f"  Done ({elapsed:.0f}s)")

    # Find best LR for each method
    adamw_finals = {lr: median_final(adamw_results[lr]) for lr in LRS}
    magma_finals = {lr: median_final(magma_results[lr]) for lr in LRS}

    best_adamw_lr = min(adamw_finals, key=adamw_finals.get)
    best_magma_lr = min(magma_finals, key=magma_finals.get)

    print(f"\n{'='*70}")
    print(f"Best LR comparison (scale={SCALE}, 10k iters, 20 seeds)")
    print(f"{'='*70}")
    print(f"  AdamW:        lr={best_adamw_lr:.5f}  median final loss = {adamw_finals[best_adamw_lr]:.8f}")
    print(f"  Magma(AdamW): lr={best_magma_lr:.5f}  median final loss = {magma_finals[best_magma_lr]:.8f}")
    ratio = magma_finals[best_magma_lr] / adamw_finals[best_adamw_lr]
    winner = "MAGMA" if ratio < 1.0 else "AdamW"
    print(f"  Ratio: {ratio:.4f}  Winner: {winner}")

    # Full LR sweep table
    print(f"\n{'LR':>10} {'AdamW Med':>14} {'Magma Med':>14} {'Ratio':>8}")
    print("-" * 50)
    for lr in LRS:
        a = adamw_finals[lr]
        m = magma_finals[lr]
        r = m / a if a > 0 and np.isfinite(a) and np.isfinite(m) else np.inf
        marker = " *" if r < 1.0 else ""
        print(f"{lr:>10.5f} {a:>14.8f} {m:>14.8f} {r:>8.3f}{marker}")

    # Plot 1: LR sweep (final loss vs LR)
    fig, ax = plt.subplots(figsize=(10, 6))
    lr_arr = np.array(LRS)
    a_arr = np.array([adamw_finals[lr] for lr in LRS])
    m_arr = np.array([magma_finals[lr] for lr in LRS])

    # Also compute IQR for error bars
    a_q25 = np.array([np.nanpercentile([l[np.isfinite(l)][-1] if np.any(np.isfinite(l)) else np.nan
                       for l in adamw_results[lr]], 25) for lr in LRS])
    a_q75 = np.array([np.nanpercentile([l[np.isfinite(l)][-1] if np.any(np.isfinite(l)) else np.nan
                       for l in adamw_results[lr]], 75) for lr in LRS])
    m_q25 = np.array([np.nanpercentile([l[np.isfinite(l)][-1] if np.any(np.isfinite(l)) else np.nan
                       for l in magma_results[lr]], 25) for lr in LRS])
    m_q75 = np.array([np.nanpercentile([l[np.isfinite(l)][-1] if np.any(np.isfinite(l)) else np.nan
                       for l in magma_results[lr]], 75) for lr in LRS])

    ax.loglog(lr_arr, a_arr, "o-", color="#1f77b4", label="AdamW", linewidth=2, markersize=5)
    ax.fill_between(lr_arr, a_q25, a_q75, alpha=0.15, color="#1f77b4")
    ax.loglog(lr_arr, m_arr, "s-", color="#2ca02c", label="Magma(AdamW)", linewidth=2, markersize=5)
    ax.fill_between(lr_arr, m_q25, m_q75, alpha=0.15, color="#2ca02c")

    ax.axvline(best_adamw_lr, color="#1f77b4", linestyle=":", alpha=0.5)
    ax.axvline(best_magma_lr, color="#2ca02c", linestyle=":", alpha=0.5)

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Median Final Loss @10k (20 seeds)")
    ax.set_title(f"Fine LR Sweep: scale={SCALE}, tau={TAU}, k={K}")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "magma_fine_lr_sweep.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved: {path}")

    # Plot 2: Convergence curves at best LRs
    fig, ax = plt.subplots(figsize=(10, 6))
    for name, results, best_lr, color in [
        ("AdamW", adamw_results, best_adamw_lr, "#1f77b4"),
        ("Magma(AdamW)", magma_results, best_magma_lr, "#2ca02c"),
    ]:
        losses = results[best_lr]
        iters = np.arange(losses.shape[1]) * LOG_EVERY
        med = ema_smooth(np.nanmedian(losses, axis=0))
        q25 = ema_smooth(np.nanpercentile(losses, 25, axis=0))
        q75 = ema_smooth(np.nanpercentile(losses, 75, axis=0))
        ax.semilogy(iters, med, label=f"{name} (lr={best_lr:.4f})", color=color, linewidth=2)
        ax.fill_between(iters, q25, q75, alpha=0.12, color=color)

    ax.set_xlabel("Gradient Steps")
    ax.set_ylabel("Loss (EMA-smoothed median, 20 seeds)")
    ax.set_title(f"Best LR for each method: scale={SCALE}")
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "magma_fine_lr_curves.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()
