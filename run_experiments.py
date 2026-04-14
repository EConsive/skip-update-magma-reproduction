"""
Experiment runner for the quadratic benchmark.

Sweeps over optimizer × learning rate × hessian type × seed × subsample_k.
Saves results as .npz files for later plotting.
"""

import os
import json
import time
import numpy as np
from benchmark import QuadraticBenchmark, DIM
from optimizers import create_optimizer, Magma, ALL_OPTIMIZERS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

LEARNING_RATES_ADAPTIVE = [0.003, 0.01, 0.03, 0.1, 0.3]
# SGD needs much smaller LRs: max eigenvalue=5000, stable LR < 2/5000=0.0004
# SkipUpdate(SGD) halves stability boundary due to 2x rescaling
LEARNING_RATES_SGD = [0.00001, 0.00003, 0.0001, 0.0002]
HESSIAN_TYPES = ["homogeneous", "heterogeneous"]
SUBSAMPLE_KS = [1, 3, 5]
NUM_SEEDS = 5
N_ITERS = 10000
LOG_EVERY = 10  # log loss every N iterations to keep arrays manageable

RESULTS_DIR = os.path.join(os.path.dirname(__file__), "results")

# Optimizer families that need SGD-scale LRs (no adaptive preconditioning)
SGD_FAMILY = {"sgd", "sgd_momentum", "skipupdate_sgd", "skipupdate_sgd_momentum"}

def get_learning_rates(optimizer_name):
    if optimizer_name in SGD_FAMILY:
        return LEARNING_RATES_SGD
    return LEARNING_RATES_ADAPTIVE


# ---------------------------------------------------------------------------
# Single experiment run
# ---------------------------------------------------------------------------

def run_single(hessian_type, optimizer_name, lr, n_iters, seed, subsample_k):
    """
    Run one experiment: optimizer on quadratic benchmark.

    Returns dict with:
      'losses': array of loss values (logged every LOG_EVERY steps)
      'alignment_scores': array of alignment scores per block (Magma only)
    """
    bench = QuadraticBenchmark(hessian_type, seed=seed)
    opt = create_optimizer(optimizer_name, dim=DIM, lr=lr)
    rng = np.random.default_rng(seed * 10000 + hash(optimizer_name) % 10000)

    w = bench.initial_point(rng)
    losses = []
    alignment_scores = []

    for t in range(n_iters):
        if t % LOG_EVERY == 0:
            losses.append(bench.loss(w))
            if isinstance(opt, Magma):
                alignment_scores.append(opt.get_alignment_scores())

        grad = bench.stochastic_gradient(w, subsample_k, rng)
        w = opt.step(w, grad, rng)

        # Divergence check
        if not np.isfinite(w).all() or bench.loss(w) > 1e20:
            # Fill remaining with NaN
            remaining = (n_iters - t - 1) // LOG_EVERY
            losses.extend([float("nan")] * remaining)
            break

    result = {"losses": np.array(losses)}
    if alignment_scores:
        result["alignment_scores"] = np.array(alignment_scores)
    return result


# ---------------------------------------------------------------------------
# Full sweep
# ---------------------------------------------------------------------------

def run_sweep(optimizer_names=None, hessian_types=None, learning_rates=None,
              subsample_ks=None, seeds=None, n_iters=N_ITERS):
    """
    Run the full experiment sweep. Saves results to RESULTS_DIR.

    Returns a nested dict: results[hessian][optimizer][lr][k] = {
        'losses': (num_seeds, n_logged_steps) array,
        'alignment_scores': ... (Magma only)
    }
    """
    if optimizer_names is None:
        optimizer_names = ALL_OPTIMIZERS
    if hessian_types is None:
        hessian_types = HESSIAN_TYPES
    if subsample_ks is None:
        subsample_ks = SUBSAMPLE_KS
    if seeds is None:
        seeds = list(range(NUM_SEEDS))

    # Count total runs (account for per-optimizer LR grids)
    total = 0
    for opt_name in optimizer_names:
        lrs = learning_rates if learning_rates else get_learning_rates(opt_name)
        total += len(hessian_types) * len(lrs) * len(subsample_ks) * len(seeds)
    print(f"Total runs: {total}")

    results = {}
    done = 0
    t_start = time.time()

    for hessian in hessian_types:
        results[hessian] = {}
        for opt_name in optimizer_names:
            results[hessian][opt_name] = {}
            opt_lrs = learning_rates if learning_rates else get_learning_rates(opt_name)
            for lr in opt_lrs:
                results[hessian][opt_name][lr] = {}
                for k in subsample_ks:
                    seed_losses = []
                    seed_alignments = []
                    for seed in seeds:
                        res = run_single(hessian, opt_name, lr, n_iters, seed, k)
                        seed_losses.append(res["losses"])
                        if "alignment_scores" in res:
                            seed_alignments.append(res["alignment_scores"])
                        done += 1

                    # Stack across seeds (pad with NaN if different lengths)
                    max_len = max(len(l) for l in seed_losses)
                    padded = []
                    for l in seed_losses:
                        if len(l) < max_len:
                            l = np.concatenate([l, np.full(max_len - len(l), np.nan)])
                        padded.append(l)

                    entry = {"losses": np.stack(padded)}
                    if seed_alignments:
                        entry["alignment_scores"] = np.stack(seed_alignments)

                    results[hessian][opt_name][lr][k] = entry

            elapsed = time.time() - t_start
            rate = done / elapsed if elapsed > 0 else 0
            remaining = (total - done) / rate if rate > 0 else 0
            print(f"  [{done}/{total}] {hessian}/{opt_name} done "
                  f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining)")

    return results


def save_results(results, filename="results.npz"):
    """Save results dict as a compressed npz file."""
    os.makedirs(RESULTS_DIR, exist_ok=True)
    filepath = os.path.join(RESULTS_DIR, filename)

    # Flatten nested dict for npz storage
    flat = {}
    meta_keys = []
    for hessian in results:
        for opt_name in results[hessian]:
            for lr in results[hessian][opt_name]:
                for k in results[hessian][opt_name][lr]:
                    entry = results[hessian][opt_name][lr][k]
                    key_base = f"{hessian}__{opt_name}__lr{lr}__k{k}"
                    flat[f"{key_base}__losses"] = entry["losses"]
                    if "alignment_scores" in entry:
                        flat[f"{key_base}__alignment"] = entry["alignment_scores"]
                    meta_keys.append(key_base)

    flat["_meta_keys"] = np.array(meta_keys, dtype=object)
    np.savez_compressed(filepath, **flat)
    print(f"Results saved to {filepath}")
    return filepath


def load_results(filename="results.npz"):
    """Load results from npz file back into nested dict."""
    filepath = os.path.join(RESULTS_DIR, filename)
    data = np.load(filepath, allow_pickle=True)

    results = {}
    for meta_key in data["_meta_keys"]:
        parts = meta_key.split("__")
        hessian, opt_name = parts[0], parts[1]
        lr = float(parts[2].replace("lr", ""))
        k = int(parts[3].replace("k", ""))

        results.setdefault(hessian, {})
        results[hessian].setdefault(opt_name, {})
        results[hessian][opt_name].setdefault(lr, {})

        entry = {"losses": data[f"{meta_key}__losses"]}
        align_key = f"{meta_key}__alignment"
        if align_key in data:
            entry["alignment_scores"] = data[align_key]

        results[hessian][opt_name][lr][k] = entry

    return results


# ---------------------------------------------------------------------------
# Quick sanity check
# ---------------------------------------------------------------------------

def sanity_check():
    """Quick validation: AdamW should converge on both hessian types."""
    print("=== Sanity Check ===")
    for hessian in HESSIAN_TYPES:
        res = run_single(hessian, "adamw", lr=0.01, n_iters=2000,
                         seed=0, subsample_k=3)
        losses = res["losses"]
        print(f"  {hessian}: loss {losses[0]:.2f} -> {losses[-1]:.6f} "
              f"(ratio: {losses[-1]/losses[0]:.2e})")
        if losses[-1] < losses[0]:
            print(f"    OK - loss decreased")
        else:
            print(f"    WARNING - loss did not decrease!")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    sanity_check()

    print("=== Running Full Sweep ===")
    results = run_sweep()
    save_results(results)
    print("Done!")
