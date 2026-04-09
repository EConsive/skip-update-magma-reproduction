"""
Quick diagnostic sweep: Magma(AdamW) with different tau and k values.
Paper's exact setup: 500 iters, heterogeneous, LRs={0.003, 0.01, 0.03, 0.1}.
"""

import time
import numpy as np
from benchmark import QuadraticBenchmark, DIM
from optimizers import AdamW, Magma
from run_experiments import LOG_EVERY, save_results

LRS = [0.003, 0.01, 0.03, 0.1]
TAUS = [0.1, 0.5, 1.0, 2.0]
KS = [1, 3, 5]
NUM_SEEDS = 20
N_ITERS = 500
HESSIAN = "heterogeneous"


def run_one(optimizer, seed, k):
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
        grad = bench.stochastic_gradient(w, k, rng)
        w = optimizer.step(w, grad, rng)
        if not np.isfinite(w).all() or bench.loss(w) > 1e20:
            remaining = (N_ITERS - t - 1) // LOG_EVERY
            losses.extend([float("nan")] * remaining)
            break

    result = {"losses": np.array(losses)}
    if alignment_scores:
        result["alignment_scores"] = np.array(alignment_scores)
    return result


def median_final(entries):
    losses = entries["losses"]
    finals = [l[np.isfinite(l)][-1] if np.any(np.isfinite(l)) else np.nan
              for l in losses]
    return np.nanmedian(finals)


def run_seeds(make_opt, k):
    all_losses, all_align = [], []
    for seed in range(NUM_SEEDS):
        res = run_one(make_opt(seed), seed, k)
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


def main():
    t0 = time.time()
    results = {HESSIAN: {}}

    # AdamW baselines for each k
    for k in KS:
        key = f"adamw_k{k}"
        results[HESSIAN][key] = {}
        for lr in LRS:
            results[HESSIAN][key][lr] = {k: run_seeds(
                lambda s, _lr=lr: AdamW(DIM, _lr), k)}
        print(f"  AdamW k={k} done ({time.time()-t0:.1f}s)")

    # Magma(AdamW) for each tau × k
    for tau in TAUS:
        for k in KS:
            key = f"magma_adamw_tau{tau}_k{k}"
            results[HESSIAN][key] = {}
            for lr in LRS:
                results[HESSIAN][key][lr] = {k: run_seeds(
                    lambda s, _lr=lr, _tau=tau: Magma(AdamW(DIM, _lr), masking="block", tau=_tau),
                    k)}
            print(f"  Magma(AdamW) tau={tau} k={k} done ({time.time()-t0:.1f}s)")

    save_results(results, "magma_adamw_sweep.npz")

    # Summary: for each (tau, k), show best Magma LR vs best AdamW LR
    print(f"\n{'='*80}")
    print(f"Magma(AdamW) vs AdamW: tau × k sweep (500 iters, heterogeneous, 20 seeds)")
    print(f"{'='*80}")
    print(f"{'k':>3} {'tau':>5} {'Best Magma LR':>14} {'Magma Med':>12} {'Best AdamW LR':>14} {'AdamW Med':>12} {'Ratio':>8}")
    print("-" * 80)

    for k in KS:
        adamw_key = f"adamw_k{k}"
        best_adamw_lr, best_adamw = None, np.inf
        for lr in LRS:
            med = median_final(results[HESSIAN][adamw_key][lr][k])
            if med < best_adamw:
                best_adamw, best_adamw_lr = med, lr

        for tau in TAUS:
            magma_key = f"magma_adamw_tau{tau}_k{k}"
            best_magma_lr, best_magma = None, np.inf
            for lr in LRS:
                med = median_final(results[HESSIAN][magma_key][lr][k])
                if med < best_magma:
                    best_magma, best_magma_lr = med, lr

            ratio = best_magma / best_adamw if best_adamw > 0 else np.inf
            marker = " <-- WINS" if ratio < 1.0 else ""
            print(f"{k:>3} {tau:>5} {best_magma_lr:>14.3f} {best_magma:>12.4f} "
                  f"{best_adamw_lr:>14.3f} {best_adamw:>12.4f} {ratio:>8.2f}{marker}")
        print()

    # Also print per-LR breakdown for the most interesting combos
    print(f"\nPer-LR breakdown (all combos):")
    print(f"{'k':>3} {'tau':>5} {'LR':>6} {'Magma Med':>12} {'AdamW Med':>12} {'Ratio':>8}")
    print("-" * 55)
    for k in KS:
        adamw_key = f"adamw_k{k}"
        for tau in TAUS:
            magma_key = f"magma_adamw_tau{tau}_k{k}"
            for lr in LRS:
                m_med = median_final(results[HESSIAN][magma_key][lr][k])
                a_med = median_final(results[HESSIAN][adamw_key][lr][k])
                ratio = m_med / a_med if a_med > 0 else np.inf
                marker = " *" if ratio < 1.0 else ""
                print(f"{k:>3} {tau:>5} {lr:>6.3f} {m_med:>12.4f} {a_med:>12.4f} {ratio:>8.2f}{marker}")
            print()

    print(f"\nTotal time: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
