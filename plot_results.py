"""
Plotting for the quadratic benchmark experiments.

Generates:
1. Figure 4 reproduction: AdamW vs Magma, homo vs hetero
2. Main comparison: all optimizers, best LR per optimizer
3. Granularity comparison: block vs element masking
4. Momentum hypothesis: SGD vs SGD+Mom vs SkipUpdate(SGD)
5. Base optimizer effect: SkipUpdate on RMSProp vs AdamW
6. Subsampling sweep: effect of noise level
7. Alignment scores over time (Magma only)
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_experiments import load_results, RESULTS_DIR, LOG_EVERY

# Consistent colors per optimizer family
COLORS = {
    "sgd": "#999999",
    "sgd_momentum": "#666666",
    "adamw": "#1f77b4",
    "rmsprop": "#ff7f0e",
    "skipupdate_block": "#2ca02c",
    "skipupdate_element": "#98df8a",
    "skipupdate_adamw_block": "#d62728",
    "skipupdate_adamw_element": "#ff9896",
    "skipupdate_sgd": "#bcbd22",
    "skipupdate_sgd_momentum": "#dbdb8d",
    "magma_block": "#9467bd",
    "magma_element": "#c5b0d5",
}

DISPLAY_NAMES = {
    "sgd": "SGD",
    "sgd_momentum": "SGD+Momentum",
    "adamw": "AdamW",
    "rmsprop": "RMSProp",
    "skipupdate_block": "SkipUpdate(RMSProp,block)",
    "skipupdate_element": "SkipUpdate(RMSProp,elem)",
    "skipupdate_adamw_block": "SkipUpdate(AdamW,block)",
    "skipupdate_adamw_element": "SkipUpdate(AdamW,elem)",
    "skipupdate_sgd": "SkipUpdate(SGD)",
    "skipupdate_sgd_momentum": "SkipUpdate(SGD+Mom)",
    "magma_block": "Magma(block)",
    "magma_element": "Magma(elem)",
}


def get_best_lr(results, hessian, opt_name, k):
    """Find the LR that achieves the lowest final median loss."""
    best_lr = None
    best_final_loss = float("inf")
    for lr in results[hessian][opt_name]:
        if k not in results[hessian][opt_name][lr]:
            continue
        losses = results[hessian][opt_name][lr][k]["losses"]
        # Median final loss across seeds (ignoring NaN)
        final_losses = []
        for seed_losses in losses:
            valid = seed_losses[np.isfinite(seed_losses)]
            if len(valid) > 0:
                final_losses.append(valid[-1])
        if final_losses:
            median_final = np.median(final_losses)
            if median_final < best_final_loss:
                best_final_loss = median_final
                best_lr = lr
    return best_lr


def plot_loss_curves(ax, results, hessian, opt_name, lr, k, label=None, color=None):
    """Plot mean ± std loss curve on an axis."""
    if opt_name not in results.get(hessian, {}):
        return
    if lr not in results[hessian][opt_name]:
        return
    if k not in results[hessian][opt_name][lr]:
        return

    losses = results[hessian][opt_name][lr][k]["losses"]
    mean = np.nanmedian(losses, axis=0)
    q25 = np.nanpercentile(losses, 25, axis=0)
    q75 = np.nanpercentile(losses, 75, axis=0)

    steps = np.arange(len(mean)) * LOG_EVERY
    if label is None:
        label = DISPLAY_NAMES.get(opt_name, opt_name)
    if color is None:
        color = COLORS.get(opt_name, None)

    ax.semilogy(steps, mean, label=label, color=color, linewidth=1.5)
    ax.fill_between(steps, q25, q75, alpha=0.15, color=color)


# ---------------------------------------------------------------------------
# Plot 1: Reproduce Figure 4 (AdamW vs Magma)
# ---------------------------------------------------------------------------

def plot_figure4_reproduction(results, k=3):
    """Reproduce Figure 4: AdamW vs Magma across LRs, homo vs hetero."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    lrs = sorted(results["homogeneous"]["adamw"].keys())

    for col, hessian in enumerate(["homogeneous", "heterogeneous"]):
        ax = axes[col]
        for opt_name in ["adamw", "magma_block"]:
            for lr in lrs:
                if k not in results[hessian].get(opt_name, {}).get(lr, {}):
                    continue
                label = f"{DISPLAY_NAMES[opt_name]} lr={lr}"
                color = COLORS[opt_name]
                # Vary alpha by LR
                alpha_map = {lrs[i]: 0.3 + 0.7 * i / (len(lrs) - 1) for i in range(len(lrs))}
                losses = results[hessian][opt_name][lr][k]["losses"]
                mean = np.nanmedian(losses, axis=0)
                steps = np.arange(len(mean)) * LOG_EVERY
                ax.semilogy(steps, mean, label=label, color=color,
                            alpha=alpha_map.get(lr, 1.0), linewidth=1.5)

        ax.set_title(f"{hessian.capitalize()} Hessian")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Figure 4 Reproduction: AdamW vs Magma (k={})".format(k), fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "fig4_reproduction.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2: Main comparison (best LR per optimizer)
# ---------------------------------------------------------------------------

def plot_main_comparison(results, k=3):
    """Compare all optimizers at their best LR, homo vs hetero."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for col, hessian in enumerate(["homogeneous", "heterogeneous"]):
        ax = axes[col]
        for opt_name in results[hessian]:
            best_lr = get_best_lr(results, hessian, opt_name, k)
            if best_lr is None:
                continue
            label = f"{DISPLAY_NAMES.get(opt_name, opt_name)} (lr={best_lr})"
            plot_loss_curves(ax, results, hessian, opt_name, best_lr, k, label=label)

        ax.set_title(f"{hessian.capitalize()} Hessian (k={k})")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=6, ncol=2)
        ax.grid(True, alpha=0.3)

    fig.suptitle("All Optimizers at Best LR", fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "main_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 3: Granularity comparison
# ---------------------------------------------------------------------------

def plot_granularity(results, k=3):
    """Block vs element masking for SkipUpdate and Magma."""
    pairs = [
        ("skipupdate_block", "skipupdate_element"),
        ("magma_block", "magma_element"),
    ]
    fig, axes = plt.subplots(len(pairs), 2, figsize=(14, 5 * len(pairs)))
    if len(pairs) == 1:
        axes = axes[np.newaxis, :]

    for row, (block_opt, elem_opt) in enumerate(pairs):
        for col, hessian in enumerate(["homogeneous", "heterogeneous"]):
            ax = axes[row, col]
            for opt_name in [block_opt, elem_opt]:
                best_lr = get_best_lr(results, hessian, opt_name, k)
                if best_lr is None:
                    continue
                plot_loss_curves(ax, results, hessian, opt_name, best_lr, k)
            # Also add AdamW baseline
            best_lr_adam = get_best_lr(results, hessian, "adamw", k)
            if best_lr_adam:
                plot_loss_curves(ax, results, hessian, "adamw", best_lr_adam, k)

            ax.set_title(f"{hessian.capitalize()}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Masking Granularity: Block vs Element", fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "granularity_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 4: Momentum hypothesis
# ---------------------------------------------------------------------------

def plot_momentum_hypothesis(results, k=3):
    """Does SkipUpdate need momentum? SGD vs SGD+Mom vs SkipUpdate(SGD) vs SkipUpdate(SGD+Mom)."""
    opts = ["sgd", "sgd_momentum", "skipupdate_sgd", "skipupdate_sgd_momentum",
            "skipupdate_block", "rmsprop"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col, hessian in enumerate(["homogeneous", "heterogeneous"]):
        ax = axes[col]
        for opt_name in opts:
            if opt_name not in results.get(hessian, {}):
                continue
            best_lr = get_best_lr(results, hessian, opt_name, k)
            if best_lr is None:
                continue
            plot_loss_curves(ax, results, hessian, opt_name, best_lr, k)

        ax.set_title(f"{hessian.capitalize()} Hessian")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Momentum Hypothesis: Does SkipUpdate Need Momentum?", fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "momentum_hypothesis.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 5: Base optimizer effect
# ---------------------------------------------------------------------------

def plot_base_optimizer(results, k=3):
    """SkipUpdate on RMSProp vs AdamW vs SGD+Mom."""
    opts = ["skipupdate_block", "skipupdate_adamw_block", "skipupdate_sgd_momentum",
            "rmsprop", "adamw", "sgd_momentum"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col, hessian in enumerate(["homogeneous", "heterogeneous"]):
        ax = axes[col]
        for opt_name in opts:
            if opt_name not in results.get(hessian, {}):
                continue
            best_lr = get_best_lr(results, hessian, opt_name, k)
            if best_lr is None:
                continue
            plot_loss_curves(ax, results, hessian, opt_name, best_lr, k)

        ax.set_title(f"{hessian.capitalize()} Hessian")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Base Optimizer Effect: SkipUpdate on Different Optimizers", fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "base_optimizer_effect.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 6: Subsampling sweep
# ---------------------------------------------------------------------------

def plot_subsampling_sweep(results):
    """How does noise level (k) affect relative rankings?"""
    core_opts = ["adamw", "rmsprop", "skipupdate_block", "magma_block"]
    ks = sorted({k for h in results for o in results[h] for lr in results[h][o]
                 for k in results[h][o][lr]})

    fig, axes = plt.subplots(len(ks), 2, figsize=(14, 5 * len(ks)))
    if len(ks) == 1:
        axes = axes[np.newaxis, :]

    for row, k in enumerate(ks):
        for col, hessian in enumerate(["homogeneous", "heterogeneous"]):
            ax = axes[row, col]
            for opt_name in core_opts:
                if opt_name not in results.get(hessian, {}):
                    continue
                best_lr = get_best_lr(results, hessian, opt_name, k)
                if best_lr is None:
                    continue
                plot_loss_curves(ax, results, hessian, opt_name, best_lr, k)

            ax.set_title(f"{hessian.capitalize()}, k={k}")
            ax.set_xlabel("Iteration")
            ax.set_ylabel("Loss")
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)

    fig.suptitle("Effect of Subsampling (Noise Level)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "subsampling_sweep.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 7: Alignment scores (Magma)
# ---------------------------------------------------------------------------

def plot_alignment_scores(results, k=3):
    """Gradient-momentum alignment per block over training (Magma only)."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col, hessian in enumerate(["homogeneous", "heterogeneous"]):
        ax = axes[col]
        for opt_name in ["magma_block", "magma_element"]:
            if opt_name not in results.get(hessian, {}):
                continue
            best_lr = get_best_lr(results, hessian, opt_name, k)
            if best_lr is None:
                continue
            entry = results[hessian][opt_name][best_lr][k]
            if "alignment_scores" not in entry:
                continue

            scores = entry["alignment_scores"]  # (seeds, steps, blocks)
            mean_scores = np.nanmean(scores, axis=0)  # (steps, blocks)
            steps = np.arange(mean_scores.shape[0]) * LOG_EVERY

            for b in range(mean_scores.shape[1]):
                ax.plot(steps, mean_scores[:, b],
                        label=f"{DISPLAY_NAMES[opt_name]} block {b}",
                        linewidth=1.0, alpha=0.8)

        ax.set_title(f"{hessian.capitalize()} Hessian")
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Alignment Score (EMA)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Gradient-Momentum Alignment (Magma)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "alignment_scores.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary_table(results, k=3):
    """Print a table of final loss for each optimizer at best LR."""
    print(f"\n{'='*80}")
    print(f"Summary: Final Median Loss at Best LR (k={k})")
    print(f"{'='*80}")
    print(f"{'Optimizer':<35} {'Homogeneous':>15} {'Heterogeneous':>15} {'Best LR (H)':>12} {'Best LR (Het)':>14}")
    print("-" * 80)

    all_opts = set()
    for h in results:
        all_opts.update(results[h].keys())

    for opt_name in sorted(all_opts):
        row = [DISPLAY_NAMES.get(opt_name, opt_name)]
        for hessian in ["homogeneous", "heterogeneous"]:
            best_lr = get_best_lr(results, hessian, opt_name, k)
            if best_lr is None:
                row.extend(["N/A", "N/A"])
                continue
            losses = results[hessian][opt_name][best_lr][k]["losses"]
            final = []
            for sl in losses:
                valid = sl[np.isfinite(sl)]
                if len(valid) > 0:
                    final.append(valid[-1])
            if final:
                row.append(f"{np.median(final):.6f}")
            else:
                row.append("diverged")
            row.append(f"{best_lr}")

        print(f"{row[0]:<35} {row[1]:>15} {row[3]:>15} {row[2]:>12} {row[4]:>14}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def generate_all_plots(results_file="results.npz"):
    """Load results and generate all plots."""
    results = load_results(results_file)

    os.makedirs(RESULTS_DIR, exist_ok=True)

    print_summary_table(results, k=3)
    plot_figure4_reproduction(results, k=3)
    plot_main_comparison(results, k=3)
    plot_granularity(results, k=3)
    plot_momentum_hypothesis(results, k=3)
    plot_base_optimizer(results, k=3)
    plot_subsampling_sweep(results)
    plot_alignment_scores(results, k=3)

    print("\nAll plots saved to:", RESULTS_DIR)


if __name__ == "__main__":
    generate_all_plots()
