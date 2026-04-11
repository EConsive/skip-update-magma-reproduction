"""
Quick experiment: start closer to optimum to reach Magma crossover faster.
w_init = scale * ones(9) for different scales.
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

LRS = [0.003, 0.01, 0.03, 0.1]
SCALES = [1.0, 0.1, 0.01]  # w_init = scale * ones(9)
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


def run_one(optimizer, seed, w_scale):
    bench = QuadraticBenchmark(HESSIAN, seed=seed)
    rng = np.random.default_rng(seed * 10000 + 9999)
    w = np.ones(DIM) * w_scale
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


def run_seeds(make_opt, w_scale):
    all_losses = []
    for seed in range(NUM_SEEDS):
        losses = run_one(make_opt(), seed, w_scale)
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
    results = {}

    for scale in SCALES:
        results[scale] = {}
        for lr in LRS:
            a = run_seeds(lambda _lr=lr: AdamW(DIM, _lr), scale)
            m = run_seeds(lambda _lr=lr: Magma(AdamW(DIM, _lr), masking="block", tau=TAU), scale)
            results[scale][lr] = {"adamw": a, "magma": m}
        print(f"  scale={scale} done ({time.time()-t0:.0f}s)")

    # Summary table
    checkpoints = [500, 1000, 2000, 5000, 10000]
    print(f"\n{'='*85}")
    print(f"Start-closer experiment: Magma(AdamW) vs AdamW, 10k iters, tau={TAU}, k={K}")
    print(f"{'='*85}")

    for scale in SCALES:
        for lr in LRS:
            a_losses = results[scale][lr]["adamw"]
            m_losses = results[scale][lr]["magma"]
            # Find first crossover
            a_med = np.nanmedian(a_losses, axis=0)
            m_med = np.nanmedian(m_losses, axis=0)
            crossover = None
            for i in range(len(a_med)):
                if np.isfinite(a_med[i]) and np.isfinite(m_med[i]) and m_med[i] < a_med[i]:
                    crossover = i * LOG_EVERY
                    break

            a_final = median_final(a_losses)
            m_final = median_final(m_losses)
            ratio = m_final / a_final if a_final > 0 else np.inf
            cross_str = f"{crossover}" if crossover else "never"
            print(f"  scale={scale:<5} lr={lr:<6} AdamW={a_final:>10.6f}  "
                  f"Magma={m_final:>10.6f}  ratio={ratio:>6.3f}  crossover@{cross_str}")

    # Plot: one row per scale, one col per LR
    fig, axes = plt.subplots(len(SCALES), len(LRS), figsize=(20, 4*len(SCALES)))
    for si, scale in enumerate(SCALES):
        for li, lr in enumerate(LRS):
            ax = axes[si][li]
            a_losses = results[scale][lr]["adamw"]
            m_losses = results[scale][lr]["magma"]
            iters = np.arange(a_losses.shape[1]) * LOG_EVERY

            for name, losses, color in [
                ("AdamW", a_losses, "#1f77b4"),
                ("Magma(AdamW)", m_losses, "#2ca02c"),
            ]:
                med = ema_smooth(np.nanmedian(losses, axis=0))
                q25 = ema_smooth(np.nanpercentile(losses, 25, axis=0))
                q75 = ema_smooth(np.nanpercentile(losses, 75, axis=0))
                ax.semilogy(iters, med, label=name, color=color, linewidth=1.5)
                ax.fill_between(iters, q25, q75, alpha=0.1, color=color)

            ax.set_title(f"scale={scale}, lr={lr}", fontsize=9)
            if si == len(SCALES)-1:
                ax.set_xlabel("Steps")
            if li == 0:
                ax.set_ylabel("Loss")
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)

    fig.suptitle(f"Effect of initialization scale on Magma crossover (tau={TAU}, k={K})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "magma_init_scale.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved: {path}")
    print(f"Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
