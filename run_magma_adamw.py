"""
Magma(AdamW) vs AdamW — matching the paper's Figure 4 exactly.

Paper shows: AdamW+Magma vs AdamW, 500 iterations, heterogeneous,
LRs = {0.003, 0.01, 0.03, 0.1}, block-wise masking.
"""

import time
import numpy as np
from benchmark import QuadraticBenchmark, DIM
from optimizers import AdamW, Magma, Magma as _Magma
from run_experiments import LOG_EVERY, RESULTS_DIR, save_results

LRS = [0.003, 0.01, 0.03, 0.1]
NUM_SEEDS = 20
N_ITERS = 500
HESSIAN = "heterogeneous"
K = 3


def run_one(optimizer, n_iters, seed):
    bench = QuadraticBenchmark(HESSIAN, seed=seed)
    rng = np.random.default_rng(seed * 10000 + 9999)
    w = bench.initial_point(rng)
    losses = []
    alignment_scores = []

    for t in range(n_iters):
        if t % LOG_EVERY == 0:
            losses.append(bench.loss(w))
            if isinstance(optimizer, _Magma):
                alignment_scores.append(optimizer.get_alignment_scores())
        grad = bench.stochastic_gradient(w, K, rng)
        w = optimizer.step(w, grad, rng)
        if not np.isfinite(w).all() or bench.loss(w) > 1e20:
            remaining = (n_iters - t - 1) // LOG_EVERY
            losses.extend([float("nan")] * remaining)
            break

    result = {"losses": np.array(losses)}
    if alignment_scores:
        result["alignment_scores"] = np.array(alignment_scores)
    return result


def run_variant(make_opt, label):
    """Run across seeds, return stacked losses."""
    all_losses = []
    all_align = []
    for seed in range(NUM_SEEDS):
        res = run_one(make_opt(seed), N_ITERS, seed)
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
    total = 2 * len(LRS) * NUM_SEEDS
    print(f"Magma(AdamW) reproduction: {total} runs")
    print(f"  {len(LRS)} LRs × {NUM_SEEDS} seeds × 2 (AdamW, Magma(AdamW))")
    print(f"  {N_ITERS} iters, heterogeneous, k={K}")

    results = {HESSIAN: {}}
    t0 = time.time()

    # AdamW baseline
    results[HESSIAN]["adamw"] = {}
    for lr in LRS:
        entry = run_variant(lambda s, _lr=lr: AdamW(DIM, _lr), f"AdamW lr={lr}")
        results[HESSIAN]["adamw"][lr] = {K: entry}
    print(f"  AdamW done ({time.time()-t0:.0f}s)")

    # Magma(AdamW, block)
    results[HESSIAN]["magma_adamw_block"] = {}
    for lr in LRS:
        entry = run_variant(
            lambda s, _lr=lr: Magma(AdamW(DIM, _lr), masking="block"),
            f"Magma(AdamW) lr={lr}"
        )
        results[HESSIAN]["magma_adamw_block"][lr] = {K: entry}
    print(f"  Magma(AdamW) done ({time.time()-t0:.0f}s)")

    save_results(results, "magma_adamw.npz")

    # Quick summary
    print(f"\n{'Optimizer':<25} {'LR':>6} {'Median@500':>12}")
    print("-" * 50)
    for opt in ["adamw", "magma_adamw_block"]:
        for lr in LRS:
            losses = results[HESSIAN][opt][lr][K]["losses"]
            finals = [l[np.isfinite(l)][-1] if np.any(np.isfinite(l)) else np.nan
                      for l in losses]
            med = np.nanmedian(finals)
            print(f"{opt:<25} {lr:>6.3f} {med:>12.4f}")
    print("Done!")


if __name__ == "__main__":
    main()
