"""
Magma(AdamW) vs AdamW at 50k iterations — looking for late-stage crossover.
Paper's Figure 4 shows Magma winning only late, at specific LRs.
"""

import time
import numpy as np
from benchmark import QuadraticBenchmark, DIM
from optimizers import AdamW, Magma
from run_experiments import LOG_EVERY, save_results

LRS = [0.003, 0.01, 0.03, 0.1]
TAU = 2.0  # paper's LLM default (most natural choice)
NUM_SEEDS = 20
N_ITERS = 50000
HESSIAN = "heterogeneous"
K = 3


def run_one(optimizer, seed):
    bench = QuadraticBenchmark(HESSIAN, seed=seed)
    rng = np.random.default_rng(seed * 10000 + 9999)
    w = bench.initial_point(rng)
    losses = []
    alignment_scores = []

    for t in range(N_ITERS):
        if t % LOG_EVERY == 0:
            losses.append(bench.loss(w))
            if isinstance(optimizer, Magma):
                alignment_scores.append(optimizer.get_alignment_scores())
        grad = bench.stochastic_gradient(w, K, rng)
        w = optimizer.step(w, grad, rng)
        if not np.isfinite(w).all() or bench.loss(w) > 1e20:
            remaining = (N_ITERS - t - 1) // LOG_EVERY
            losses.extend([float("nan")] * remaining)
            break

    result = {"losses": np.array(losses)}
    if alignment_scores:
        result["alignment_scores"] = np.array(alignment_scores)
    return result


def run_seeds(make_opt):
    all_losses, all_align = [], []
    for seed in range(NUM_SEEDS):
        res = run_one(make_opt(), seed)
        all_losses.append(res["losses"])
        if "alignment_scores" in res:
            all_align.append(res["alignment_scores"])
    max_len = max(len(l) for l in all_losses)
    padded = [np.concatenate([l, np.full(max_len - len(l), np.nan)])
              if len(l) < max_len else l for l in all_losses]
    entry = {"losses": np.stack(padded)}
    if all_align:
        entry["alignment_scores"] = np.stack(all_align)
    return entry


def median_at(entry, step):
    idx = step // LOG_EVERY
    if idx >= entry["losses"].shape[1]:
        idx = entry["losses"].shape[1] - 1
    col = entry["losses"][:, idx]
    return np.nanmedian(col)


def main():
    t0 = time.time()
    total = 2 * len(LRS) * NUM_SEEDS
    print(f"50k iteration experiment: {total} runs")

    results = {}

    for lr in LRS:
        adamw_entry = run_seeds(lambda _lr=lr: AdamW(DIM, _lr))
        magma_entry = run_seeds(
            lambda _lr=lr: Magma(AdamW(DIM, _lr), masking="block", tau=TAU))
        results[lr] = {"adamw": adamw_entry, "magma": magma_entry}
        print(f"  lr={lr} done ({time.time()-t0:.0f}s)")

    # Print summary at multiple checkpoints
    checkpoints = [500, 1000, 2000, 5000, 10000, 20000, 50000]
    print(f"\n{'='*90}")
    print(f"Magma(AdamW) vs AdamW: 50k iters, tau={TAU}, k={K}, 20 seeds")
    print(f"{'='*90}")

    for lr in LRS:
        print(f"\n--- LR = {lr} ---")
        print(f"{'Iter':>8} {'AdamW Med':>12} {'Magma Med':>12} {'Ratio':>8} {'Winner':>8}")
        print("-" * 55)
        for cp in checkpoints:
            a = median_at(results[lr]["adamw"], cp)
            m = median_at(results[lr]["magma"], cp)
            ratio = m / a if a > 0 and np.isfinite(a) and np.isfinite(m) else np.inf
            winner = "MAGMA" if ratio < 1.0 else "AdamW"
            print(f"{cp:>8} {a:>12.6f} {m:>12.6f} {ratio:>8.3f} {winner:>8}")

    # Save for plotting
    save_data = {HESSIAN: {}}
    for lr in LRS:
        save_data[HESSIAN][f"adamw_lr{lr}"] = {lr: {K: results[lr]["adamw"]}}
        save_data[HESSIAN][f"magma_adamw_lr{lr}"] = {lr: {K: results[lr]["magma"]}}
    save_results(save_data, "magma_50k.npz")

    # Plot convergence curves
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from run_experiments import RESULTS_DIR
    import os

    def ema_smooth(x, alpha=0.02):
        """Exponential moving average for temporal smoothing."""
        out = np.empty_like(x)
        out[0] = x[0]
        for i in range(1, len(x)):
            if np.isfinite(x[i]):
                out[i] = alpha * x[i] + (1 - alpha) * out[i - 1]
            else:
                out[i] = out[i - 1]
        return out

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    for idx, lr in enumerate(LRS):
        ax = axes[idx // 2][idx % 2]
        iters = np.arange(results[lr]["adamw"]["losses"].shape[1]) * LOG_EVERY

        for name, entry, color, ls in [
            ("AdamW", results[lr]["adamw"], "#1f77b4", "-"),
            ("Magma(AdamW)", results[lr]["magma"], "#2ca02c", "-"),
        ]:
            losses = entry["losses"]
            median = np.nanmedian(losses, axis=0)
            q25 = np.nanpercentile(losses, 25, axis=0)
            q75 = np.nanpercentile(losses, 75, axis=0)

            # Apply EMA smoothing for cleaner curves
            median_s = ema_smooth(median)
            q25_s = ema_smooth(q25)
            q75_s = ema_smooth(q75)

            ax.semilogy(iters, median_s, label=name, color=color, linewidth=2, linestyle=ls)
            ax.fill_between(iters, q25_s, q75_s, alpha=0.1, color=color)

        ax.set_title(f"lr={lr}")
        ax.set_xlabel("Gradient steps")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.axvline(500, color="gray", linestyle=":", alpha=0.5, label="500 iters")

    fig.suptitle(f"Magma(AdamW) vs AdamW: 50k steps (tau={TAU}, k={K}, 20 seeds)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "magma_50k_curves.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved: {path}")
    print(f"Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
