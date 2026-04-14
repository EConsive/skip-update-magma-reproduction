"""
Focused experiment: block-wise vs element-wise masking for SGD+Momentum and AdamW.

Fills the gap from earlier focused experiments where SGD+Mom was only tested
with element-wise and AdamW was only tested with block-wise.

30 seeds × 13-14 LRs × 6 variants, heterogeneous Hessian, k=3, 10000 iters.
"""

import os
import time
import numpy as np
from run_experiments import run_single, save_results, RESULTS_DIR

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

NUM_SEEDS = 30
N_ITERS = 10000
HESSIAN = "heterogeneous"
K = 3

# SGD family LRs (from focused_sgdmom experiment)
LRS_SGD = [0.00005, 0.00008, 0.0001, 0.00012, 0.00015, 0.00018,
           0.0002, 0.00022, 0.00025, 0.00028, 0.0003, 0.00032, 0.00035]

# AdamW family LRs (from focused_adamw experiment)
LRS_ADAMW = [0.001, 0.003, 0.005, 0.008, 0.01, 0.015, 0.02,
             0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3]

VARIANTS = [
    # (optimizer_name, lr_grid)
    ("sgd_momentum",                LRS_SGD),
    ("skipupdate_sgd_momentum",     LRS_SGD),      # element-wise (existing)
    ("skipupdate_sgd_momentum_block", LRS_SGD),     # block-wise (new)
    ("adamw",                       LRS_ADAMW),
    ("skipupdate_adamw_element",    LRS_ADAMW),     # element-wise
    ("skipupdate_adamw_block",      LRS_ADAMW),     # block-wise
]

# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------

def main():
    seeds = list(range(NUM_SEEDS))

    # Count total runs
    total = sum(len(lrs) * len(seeds) for _, lrs in VARIANTS)
    print(f"Focused granularity experiment: {total} runs")
    print(f"  {len(VARIANTS)} variants, {NUM_SEEDS} seeds, heterogeneous, k={K}")

    results = {}
    results[HESSIAN] = {}
    done = 0
    t_start = time.time()

    for opt_name, lr_grid in VARIANTS:
        results[HESSIAN][opt_name] = {}
        for lr in lr_grid:
            results[HESSIAN][opt_name][lr] = {}
            seed_losses = []
            for seed in seeds:
                res = run_single(HESSIAN, opt_name, lr, N_ITERS, seed, K)
                seed_losses.append(res["losses"])
                done += 1

            max_len = max(len(l) for l in seed_losses)
            padded = []
            for l in seed_losses:
                if len(l) < max_len:
                    l = np.concatenate([l, np.full(max_len - len(l), np.nan)])
                padded.append(l)

            results[HESSIAN][opt_name][lr][K] = {"losses": np.stack(padded)}

        elapsed = time.time() - t_start
        rate = done / elapsed if elapsed > 0 else 0
        remaining = (total - done) / rate if rate > 0 else 0
        print(f"  [{done}/{total}] {opt_name} done "
              f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    save_results(results, "focused_granularity.npz")
    print("Done!")


if __name__ == "__main__":
    main()
