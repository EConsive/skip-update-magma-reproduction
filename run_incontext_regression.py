"""
Section 4.3 reproduction — in-context linear regression with light- vs
heavy-tailed covariates. Pass A: AdamW vs Magma(AdamW, block-mask) only.

Pass A gates Pass B: if Magma(AdamW) reproduces the paper's heavy-tailed
advantage, we run Pass B adding SkipUpdate(AdamW) and RMSProp. Otherwise we
document the second non-reproduction.

Architecture note: we use the exact Ahn et al. 2024 single-layer linear
attention parameterization (row_p ∈ R^{d+1}, Q ∈ R^{(d+1)×(d+1)}), 42 params
total for d=5. See incontext_benchmark.py docstring.
"""

import os
import time
import numpy as np

# Monkeypatch BLOCK_SLICES BEFORE importing optimizer classes.
# 42 params = 7 blocks of 6 under the Ahn et al. layout:
#   block 0   = row_p          (params[0:6])
#   blocks 1..6 = rows of Q    (params[6+6*i : 6+6*(i+1)] for i in 0..5)
# Each block is one semantically meaningful row of the parameterization.
import optimizers
optimizers.BLOCK_SIZE = 6
optimizers.NUM_BLOCKS = 7
optimizers.BLOCK_SLICES = [slice(6 * i, 6 * (i + 1)) for i in range(7)]

from optimizers import AdamW, Magma  # noqa: E402
from run_experiments import RESULTS_DIR  # noqa: E402
from incontext_benchmark import LinearAttentionRegression, PARAM_DIM  # noqa: E402


REGIMES = ["light", "heavy"]
LRS = np.logspace(-4, 0, 12).tolist()  # 1e-4 to 1.0
NUM_SEEDS = 15
N_ITERS = 3000
LOG_EVERY = 30
N_EVAL = 128

OPTIMS = {
    "adamw": lambda lr: AdamW(PARAM_DIM, lr),
    "magma_adamw_block": lambda lr: Magma(AdamW(PARAM_DIM, lr), masking="block"),
}


def run_one(regime, opt_name, lr, seed):
    benchmark = LinearAttentionRegression(regime)
    rng = np.random.default_rng(seed * 100000 + 12345)
    W = benchmark.warm_start_point(scale=0.5, rng=rng)
    opt = OPTIMS[opt_name](lr)
    losses = []
    diverged = False
    for t in range(N_ITERS + 1):
        if t % LOG_EVERY == 0:
            if diverged:
                losses.append(float("nan"))
            else:
                # Deterministic eval seed per checkpoint — same across runs,
                # for a low-variance comparison.
                eval_rng = np.random.default_rng(12345678 + t)
                losses.append(benchmark.loss(W, eval_rng, n_eval=N_EVAL))
        if t == N_ITERS:
            break
        if not diverged:
            g = benchmark.stochastic_gradient(W, rng)
            W = opt.step(W, g, rng)
            if not np.isfinite(W).all():
                diverged = True
    return np.array(losses, dtype=float)


def run_seeds(regime, opt_name, lr):
    rows = [run_one(regime, opt_name, lr, seed) for seed in range(NUM_SEEDS)]
    return np.stack(rows)


def final_metric(losses_mat):
    """Median over seeds of (mean of last 5 log points per seed)."""
    tail = losses_mat[:, -5:]
    per_seed = np.nanmean(tail, axis=1)
    return np.nanmedian(per_seed)


def main():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    t0 = time.time()
    total_runs = len(REGIMES) * len(OPTIMS) * len(LRS) * NUM_SEEDS
    print(f"Section 4.3 Pass A: {total_runs} runs")
    print(f"  regimes={REGIMES}, opts={list(OPTIMS)}, "
          f"LRs={len(LRS)}, seeds={NUM_SEEDS}, iters={N_ITERS}")
    print(f"  PARAM_DIM={PARAM_DIM}  (d=5, W in R^(5x5))")
    print(f"  LOG_EVERY={LOG_EVERY}, N_EVAL={N_EVAL}")

    results = {r: {o: {} for o in OPTIMS} for r in REGIMES}
    for regime in REGIMES:
        for opt_name in OPTIMS:
            for lr in LRS:
                results[regime][opt_name][lr] = run_seeds(regime, opt_name, lr)
            print(f"  [{regime:6} | {opt_name:<20}] done ({time.time() - t0:.0f}s)")

    # Summary table
    print()
    print("=" * 80)
    print("Best-vs-best (median of seed-tails) — Pass A")
    print("=" * 80)
    print(f"{'regime':>6} {'method':>22}  {'best LR':>10}  {'tail loss':>14}")
    print("-" * 80)
    best = {r: {} for r in REGIMES}
    for regime in REGIMES:
        for opt_name in OPTIMS:
            per_lr = {lr: final_metric(results[regime][opt_name][lr]) for lr in LRS}
            best_lr = min(per_lr, key=lambda k: per_lr[k] if np.isfinite(per_lr[k]) else np.inf)
            best[regime][opt_name] = {"lr": best_lr, "loss": per_lr[best_lr], "per_lr": per_lr}
            print(f"{regime:>6} {opt_name:>22}  {best_lr:>10.5f}  {per_lr[best_lr]:>14.6f}")
    print()
    for regime in REGIMES:
        a = best[regime]["adamw"]["loss"]
        m = best[regime]["magma_adamw_block"]["loss"]
        ratio = m / a if a > 0 else np.inf
        winner = "Magma" if ratio < 1.0 else "AdamW"
        print(f"  [{regime:6}] Magma/AdamW = {ratio:.3f}   ({winner} wins)")

    # Save raw data
    save = {
        "regimes": np.array(REGIMES),
        "lrs": np.array(LRS),
        "n_iters": N_ITERS,
        "log_every": LOG_EVERY,
        "num_seeds": NUM_SEEDS,
        "param_dim": PARAM_DIM,
    }
    for regime in REGIMES:
        for opt_name in OPTIMS:
            for lr_i, lr in enumerate(LRS):
                save[f"{regime}__{opt_name}__lr{lr_i}"] = results[regime][opt_name][lr]
    path = os.path.join(RESULTS_DIR, "incontext_regression_passA_warmstart.npz")
    np.savez(path, **save)
    print(f"\nSaved: {path}")
    print(f"Total Pass A time: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
