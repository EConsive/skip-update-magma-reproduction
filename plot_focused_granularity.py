"""
Plots for the focused granularity experiment: block-wise vs element-wise masking.

Generates per base optimizer (SGD+Mom, AdamW):
1. LR sweep: final loss (median + IQR) vs LR, 3 lines (baseline, block, element)
2. Paired-seed scatter at best LR for each method
3. Summary table printed to stdout
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_experiments import load_results, RESULTS_DIR, LOG_EVERY

K = 3
HESSIAN = "heterogeneous"


def get_final_losses(results, opt_name, lr):
    """Return array of final losses per seed (ignoring NaN)."""
    losses = results[HESSIAN][opt_name][lr][K]["losses"]
    finals = []
    for seed_losses in losses:
        valid = seed_losses[np.isfinite(seed_losses)]
        finals.append(valid[-1] if len(valid) > 0 else np.nan)
    return np.array(finals)


def get_best_lr(results, opt_name):
    """Find LR with lowest median final loss."""
    best_lr, best_med = None, np.inf
    for lr in results[HESSIAN][opt_name]:
        finals = get_final_losses(results, opt_name, lr)
        med = np.nanmedian(finals)
        if med < best_med:
            best_med = med
            best_lr = lr
    return best_lr


# ---------------------------------------------------------------------------
# Plot 1: LR sweep
# ---------------------------------------------------------------------------

def plot_lr_sweep(results, base_name, skip_elem, skip_block, lr_grid, filename):
    """Final loss vs LR with IQR bands."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for opt_name, label, color, marker in [
        (base_name, base_name.replace("_", " ").title(), "#1f77b4", "o"),
        (skip_elem, "SkipUpdate(element)", "#ff7f0e", "s"),
        (skip_block, "SkipUpdate(block)", "#2ca02c", "^"),
    ]:
        medians, q25s, q75s, lrs_valid = [], [], [], []
        for lr in lr_grid:
            if lr not in results[HESSIAN].get(opt_name, {}):
                continue
            finals = get_final_losses(results, opt_name, lr)
            if np.all(np.isnan(finals)):
                continue
            medians.append(np.nanmedian(finals))
            q25s.append(np.nanpercentile(finals, 25))
            q75s.append(np.nanpercentile(finals, 75))
            lrs_valid.append(lr)

        if not lrs_valid:
            continue
        lrs_valid = np.array(lrs_valid)
        medians = np.array(medians)
        q25s = np.array(q25s)
        q75s = np.array(q75s)

        ax.semilogy(range(len(lrs_valid)), medians, label=label,
                     color=color, marker=marker, linewidth=2, markersize=6)
        ax.fill_between(range(len(lrs_valid)), q25s, q75s,
                         alpha=0.15, color=color)
        ax.set_xticks(range(len(lrs_valid)))
        ax.set_xticklabels([f"{lr:.5f}" if lr < 0.001 else f"{lr:.3f}"
                            for lr in lrs_valid], rotation=45, fontsize=7)

    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Final Loss (median, 30 seeds)")
    ax.set_title(f"Block vs Element Masking: {base_name.replace('_', ' ').title()}")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2: Paired-seed scatter
# ---------------------------------------------------------------------------

def plot_paired_scatter(results, base_name, skip_elem, skip_block, filename):
    """Scatter: SkipUpdate loss vs baseline loss at best LR for each."""
    best_lr_base = get_best_lr(results, base_name)
    best_lr_elem = get_best_lr(results, skip_elem)
    best_lr_block = get_best_lr(results, skip_block)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, skip_name, skip_label, best_lr_skip, color in [
        (axes[0], skip_elem, "SkipUpdate(element)", best_lr_elem, "#ff7f0e"),
        (axes[1], skip_block, "SkipUpdate(block)", best_lr_block, "#2ca02c"),
    ]:
        base_finals = get_final_losses(results, base_name, best_lr_base)
        skip_finals = get_final_losses(results, skip_name, best_lr_skip)

        valid = np.isfinite(base_finals) & np.isfinite(skip_finals)
        bf, sf = base_finals[valid], skip_finals[valid]

        ax.scatter(bf, sf, alpha=0.6, color=color, edgecolors="black", linewidths=0.5, s=50)

        # Diagonal
        lo = min(bf.min(), sf.min()) * 0.5
        hi = max(bf.max(), sf.max()) * 2
        ax.plot([lo, hi], [lo, hi], "k--", alpha=0.3, linewidth=1)

        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(f"{base_name.replace('_',' ').title()} (lr={best_lr_base})")
        ax.set_ylabel(f"{skip_label} (lr={best_lr_skip})")
        ax.set_title(f"{skip_label}: wins {np.sum(sf < bf)}/{len(bf)} seeds")
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Paired-Seed Comparison: {base_name.replace('_',' ').title()}", fontsize=13)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results):
    """Print summary table for all 6 variants."""
    print(f"\n{'='*85}")
    print(f"Focused Granularity Experiment: Summary (heterogeneous, k=3, 30 seeds)")
    print(f"{'='*85}")
    print(f"{'Optimizer':<35} {'Best LR':>10} {'Median Loss':>14} {'IQR':>30}")
    print("-" * 85)

    for opt_name in ["sgd_momentum", "skipupdate_sgd_momentum",
                     "skipupdate_sgd_momentum_block",
                     "adamw", "skipupdate_adamw_element",
                     "skipupdate_adamw_block"]:
        if opt_name not in results[HESSIAN]:
            continue
        best_lr = get_best_lr(results, opt_name)
        finals = get_final_losses(results, opt_name, best_lr)
        med = np.nanmedian(finals)
        q25 = np.nanpercentile(finals, 25)
        q75 = np.nanpercentile(finals, 75)
        print(f"{opt_name:<35} {best_lr:>10.5f} {med:>14.6f} [{q25:.6f}, {q75:.6f}]")

    print()


# ---------------------------------------------------------------------------
# Paired-seed table at matched LRs
# ---------------------------------------------------------------------------

def print_paired_table(results, base_name, skip_name, lr_grid):
    """Print paired comparison at each LR."""
    print(f"\nPaired comparison: {skip_name} vs {base_name}")
    print(f"{'LR':>10} {'Skip wins':>12} {'Median ratio':>14}")
    print("-" * 40)

    for lr in lr_grid:
        if lr not in results[HESSIAN].get(base_name, {}):
            continue
        if lr not in results[HESSIAN].get(skip_name, {}):
            continue
        bf = get_final_losses(results, base_name, lr)
        sf = get_final_losses(results, skip_name, lr)
        valid = np.isfinite(bf) & np.isfinite(sf)
        bf, sf = bf[valid], sf[valid]
        if len(bf) == 0:
            continue
        wins = np.sum(sf < bf)
        ratio = np.median(sf / bf)
        print(f"{lr:>10.5f} {wins:>5}/{len(bf):<5} {ratio:>14.2f}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = load_results("focused_granularity.npz")

    print_summary(results)

    # SGD+Momentum family
    from run_focused_granularity import LRS_SGD, LRS_ADAMW

    plot_lr_sweep(results, "sgd_momentum", "skipupdate_sgd_momentum",
                  "skipupdate_sgd_momentum_block", LRS_SGD,
                  "granularity_sgdmom_lr_sweep.png")

    plot_paired_scatter(results, "sgd_momentum", "skipupdate_sgd_momentum",
                        "skipupdate_sgd_momentum_block",
                        "granularity_sgdmom_paired.png")

    print_paired_table(results, "sgd_momentum", "skipupdate_sgd_momentum", LRS_SGD)
    print_paired_table(results, "sgd_momentum", "skipupdate_sgd_momentum_block", LRS_SGD)

    # AdamW family
    plot_lr_sweep(results, "adamw", "skipupdate_adamw_element",
                  "skipupdate_adamw_block", LRS_ADAMW,
                  "granularity_adamw_lr_sweep.png")

    plot_paired_scatter(results, "adamw", "skipupdate_adamw_element",
                        "skipupdate_adamw_block",
                        "granularity_adamw_paired.png")

    print_paired_table(results, "adamw", "skipupdate_adamw_element", LRS_ADAMW)
    print_paired_table(results, "adamw", "skipupdate_adamw_block", LRS_ADAMW)

    print("\nAll plots saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
