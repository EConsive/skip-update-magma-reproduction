"""
Sweep all k values (1-8) plus Bernoulli(0.5) row sampling.
Magma(AdamW) vs AdamW, 500 iters, heterogeneous, tau=0.1, 20 seeds.
"""

import time
import numpy as np
from benchmark import QuadraticBenchmark, DIM
from optimizers import AdamW, Magma
from run_experiments import LOG_EVERY, save_results

LRS = [0.003, 0.01, 0.03, 0.1]
TAU = 0.1
KS = list(range(1, 9)) + ["bernoulli"]  # k=1..8 + Bernoulli(0.5)
NUM_SEEDS = 20
N_ITERS = 500
HESSIAN = "heterogeneous"


def run_one(optimizer, seed, k):
    bench = QuadraticBenchmark(HESSIAN, seed=seed)
    rng = np.random.default_rng(seed * 10000 + 9999)
    w = bench.initial_point(rng)
    losses = []
    for t in range(N_ITERS):
        if t % LOG_EVERY == 0:
            losses.append(bench.loss(w))
        grad = bench.stochastic_gradient(w, k, rng)
        w = optimizer.step(w, grad, rng)
        if not np.isfinite(w).all() or bench.loss(w) > 1e20:
            remaining = (N_ITERS - t - 1) // LOG_EVERY
            losses.extend([float("nan")] * remaining)
            break
    return np.array(losses)


def run_seeds(make_opt, k):
    all_losses = []
    for seed in range(NUM_SEEDS):
        losses = run_one(make_opt(), seed, k)
        all_losses.append(losses)
    max_len = max(len(l) for l in all_losses)
    padded = [np.concatenate([l, np.full(max_len - len(l), np.nan)])
              if len(l) < max_len else l for l in all_losses]
    return {"losses": np.stack(padded)}


def median_final(entry):
    losses = entry["losses"]
    finals = [l[np.isfinite(l)][-1] if np.any(np.isfinite(l)) else np.nan
              for l in losses]
    return np.nanmedian(finals)


def main():
    t0 = time.time()
    total = len(KS) * len(LRS) * NUM_SEEDS * 2
    print(f"k sweep: {total} runs ({len(KS)} k-values × {len(LRS)} LRs × {NUM_SEEDS} seeds × 2)")

    results = []

    for k in KS:
        k_label = str(k)
        for lr in LRS:
            adamw_entry = run_seeds(lambda _lr=lr: AdamW(DIM, _lr), k)
            magma_entry = run_seeds(
                lambda _lr=lr: Magma(AdamW(DIM, _lr), masking="block", tau=TAU), k)

            a_med = median_final(adamw_entry)
            m_med = median_final(magma_entry)
            ratio = m_med / a_med if a_med > 0 else np.inf
            results.append((k_label, lr, a_med, m_med, ratio))

        elapsed = time.time() - t0
        print(f"  k={k_label} done ({elapsed:.0f}s)")

    # Summary: best ratio per k
    print(f"\n{'='*75}")
    print(f"Magma(AdamW) vs AdamW: k sweep (tau=0.1, 500 iters, 20 seeds)")
    print(f"{'='*75}")
    print(f"{'k':>10} {'Best LR':>8} {'AdamW Med':>12} {'Magma Med':>12} {'Ratio':>8}")
    print("-" * 55)

    for k in KS:
        k_label = str(k)
        k_results = [(lr, a, m, r) for kl, lr, a, m, r in results if kl == k_label]
        best = min(k_results, key=lambda x: x[3])  # best Magma
        best_adamw = min(k_results, key=lambda x: x[1])  # best AdamW
        ratio = best[3] / best_adamw[1] if best_adamw[1] > 0 else np.inf
        marker = " <-- WINS" if ratio < 1.0 else ""
        print(f"{k_label:>10} {best[0]:>8.3f} {best_adamw[1]:>12.4f} {best[3]:>12.4f} {ratio:>8.3f}{marker}")

    # Full per-LR table
    print(f"\nPer-LR breakdown:")
    print(f"{'k':>10} {'LR':>8} {'AdamW':>12} {'Magma':>12} {'Ratio':>8}")
    print("-" * 55)
    for k_label, lr, a, m, r in results:
        marker = " *" if r < 1.0 else ""
        print(f"{k_label:>10} {lr:>8.3f} {a:>12.4f} {m:>12.4f} {r:>8.3f}{marker}")

    print(f"\nTotal: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
