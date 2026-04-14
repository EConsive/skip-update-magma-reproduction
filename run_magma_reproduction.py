"""
Magma reproduction: sweep tau × LR to find the regime where Magma beats AdamW.

The paper claims Magma outperforms AdamW on heterogeneous quadratics (Figure 4,
Section 4.4) but our default tau=2.0 yields s≈0.5 always (no discrimination).
This experiment sweeps tau from 0.01 to 5.0 alongside LR to find the right combo.

Also records alignment scores for diagnostic plots.
"""

import time
import numpy as np
from benchmark import QuadraticBenchmark, DIM
from optimizers import RMSProp, AdamW, Magma, create_optimizer
from run_experiments import save_results, LOG_EVERY, RESULTS_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

TAUS = [0.01, 0.05, 0.1, 0.5, 1.0, 2.0, 5.0]
LRS = [0.003, 0.01, 0.03, 0.1, 0.3, 0.5, 1.0]

NUM_SEEDS = 20
N_ITERS = 10000
HESSIAN = "heterogeneous"
K = 3


def run_one(hessian_type, optimizer, lr, n_iters, seed, subsample_k):
    """Run a single experiment with an already-constructed optimizer."""
    bench = QuadraticBenchmark(hessian_type, seed=seed)
    rng = np.random.default_rng(seed * 10000 + 9999)

    w = bench.initial_point(rng)
    losses = []
    alignment_scores = []

    for t in range(n_iters):
        if t % LOG_EVERY == 0:
            losses.append(bench.loss(w))
            if isinstance(optimizer, Magma):
                alignment_scores.append(optimizer.get_alignment_scores())

        grad = bench.stochastic_gradient(w, subsample_k, rng)
        w = optimizer.step(w, grad, rng)

        if not np.isfinite(w).all() or bench.loss(w) > 1e20:
            remaining = (n_iters - t - 1) // LOG_EVERY
            losses.extend([float("nan")] * remaining)
            break

    result = {"losses": np.array(losses)}
    if alignment_scores:
        result["alignment_scores"] = np.array(alignment_scores)
    return result


def run_seeds(make_opt, n_seeds, label=""):
    """Run across seeds, return stacked losses and alignment scores."""
    seed_losses = []
    seed_alignments = []
    for seed in range(n_seeds):
        opt = make_opt()
        res = run_one(HESSIAN, opt, None, N_ITERS, seed, K)
        seed_losses.append(res["losses"])
        if "alignment_scores" in res:
            seed_alignments.append(res["alignment_scores"])

    max_len = max(len(l) for l in seed_losses)
    padded = []
    for l in seed_losses:
        if len(l) < max_len:
            l = np.concatenate([l, np.full(max_len - len(l), np.nan)])
        padded.append(l)

    entry = {"losses": np.stack(padded)}
    if seed_alignments:
        entry["alignment_scores"] = np.stack(seed_alignments)
    return entry


def main():
    total_magma = len(TAUS) * len(LRS) * NUM_SEEDS
    total_baselines = 2 * len(LRS) * NUM_SEEDS  # RMSProp + AdamW
    total = total_magma + total_baselines
    print(f"Magma reproduction experiment: {total} runs")
    print(f"  {len(TAUS)} taus × {len(LRS)} LRs × {NUM_SEEDS} seeds = {total_magma} Magma runs")
    print(f"  2 baselines × {len(LRS)} LRs × {NUM_SEEDS} seeds = {total_baselines} baseline runs")

    results = {HESSIAN: {}}
    done = 0
    t_start = time.time()

    # --- Magma sweep ---
    for tau in TAUS:
        key = f"magma_block_tau{tau}"
        results[HESSIAN][key] = {}
        for lr in LRS:
            entry = run_seeds(
                lambda _lr=lr, _tau=tau: Magma(RMSProp(DIM, _lr), masking="block", tau=_tau),
                NUM_SEEDS
            )
            results[HESSIAN][key][lr] = {K: entry}
            done += NUM_SEEDS

        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else 0
        print(f"  [{done}/{total}] Magma tau={tau} done "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    # --- RMSProp baseline ---
    results[HESSIAN]["rmsprop"] = {}
    for lr in LRS:
        entry = run_seeds(
            lambda _lr=lr: RMSProp(DIM, _lr),
            NUM_SEEDS
        )
        results[HESSIAN]["rmsprop"][lr] = {K: entry}
        done += NUM_SEEDS

    elapsed = time.time() - t_start
    rate = done / elapsed if elapsed > 0 else 0
    remaining = (total - done) / rate if rate > 0 else 0
    print(f"  [{done}/{total}] RMSProp done ({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    # --- AdamW baseline ---
    results[HESSIAN]["adamw"] = {}
    for lr in LRS:
        entry = run_seeds(
            lambda _lr=lr: AdamW(DIM, _lr),
            NUM_SEEDS
        )
        results[HESSIAN]["adamw"][lr] = {K: entry}
        done += NUM_SEEDS

    elapsed = time.time() - t_start
    print(f"  [{done}/{total}] AdamW done ({elapsed:.0f}s elapsed)")

    save_results(results, "magma_reproduction.npz")
    print("Done!")


if __name__ == "__main__":
    main()
