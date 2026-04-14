"""
Plots for Magma reproduction experiment:
1. Tau × LR heatmap of final loss
2. Loss curves: best Magma vs baselines (at 500 and 10000 iters)
3. Alignment score evolution for best tau vs tau=2.0
4. Summary table
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

from run_experiments import load_results, RESULTS_DIR, LOG_EVERY
from run_magma_reproduction import TAUS, LRS, K, HESSIAN

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_final_losses(data):
    """Return array of final losses per seed."""
    losses = data["losses"]
    finals = []
    for sl in losses:
        valid = sl[np.isfinite(sl)]
        finals.append(valid[-1] if len(valid) > 0 else np.nan)
    return np.array(finals)


def get_loss_at_iter(data, target_iter):
    """Return array of losses per seed at a specific iteration."""
    losses = data["losses"]
    idx = target_iter // LOG_EVERY
    if idx >= losses.shape[1]:
        idx = losses.shape[1] - 1
    return losses[:, idx]


# ---------------------------------------------------------------------------
# Plot 1: Tau × LR heatmap
# ---------------------------------------------------------------------------

def plot_heatmap(results, filename, eval_iter=None):
    """2D heatmap: tau (y) × LR (x), color = median final loss."""
    fig, ax = plt.subplots(figsize=(10, 7))

    grid = np.full((len(TAUS), len(LRS)), np.nan)
    for i, tau in enumerate(TAUS):
        key = f"magma_block_tau{tau}"
        if key not in results[HESSIAN]:
            continue
        for j, lr in enumerate(LRS):
            if lr not in results[HESSIAN][key]:
                continue
            data = results[HESSIAN][key][lr][K]
            if eval_iter is not None:
                finals = get_loss_at_iter(data, eval_iter)
            else:
                finals = get_final_losses(data)
            grid[i, j] = np.nanmedian(finals)

    # Replace NaN/inf with max for colormap
    valid = grid[np.isfinite(grid)]
    if len(valid) == 0:
        print("No valid data for heatmap!")
        return
    vmin, vmax = valid.min(), valid.max()
    grid_plot = np.where(np.isfinite(grid), grid, vmax)

    im = ax.imshow(grid_plot, cmap="viridis_r", aspect="auto",
                   norm=LogNorm(vmin=max(vmin, 1e-10), vmax=vmax))
    cbar = fig.colorbar(im, ax=ax, label="Median Final Loss (20 seeds)")

    ax.set_xticks(range(len(LRS)))
    ax.set_xticklabels([f"{lr}" for lr in LRS])
    ax.set_yticks(range(len(TAUS)))
    ax.set_yticklabels([f"{tau}" for tau in TAUS])
    ax.set_xlabel("Learning Rate")
    ax.set_ylabel("Temperature (tau)")

    # Annotate cells
    for i in range(len(TAUS)):
        for j in range(len(LRS)):
            val = grid[i, j]
            if np.isfinite(val):
                color = "white" if val > np.sqrt(vmin * vmax) else "black"
                ax.text(j, i, f"{val:.4f}" if val < 1 else f"{val:.1f}",
                        ha="center", va="center", fontsize=7, color=color)

    iter_label = f"at {eval_iter} iters" if eval_iter else "at 10000 iters"
    ax.set_title(f"Magma(block) Median Final Loss: Tau × LR ({iter_label})")

    # Mark best cell
    best_idx = np.unravel_index(np.nanargmin(grid), grid.shape)
    ax.add_patch(plt.Rectangle((best_idx[1] - 0.5, best_idx[0] - 0.5),
                                1, 1, fill=False, edgecolor="red", linewidth=3))

    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Plot 2: Loss curves — best Magma vs baselines
# ---------------------------------------------------------------------------

def plot_loss_curves(results, filename):
    """Loss curves: best Magma(tau, LR) vs best RMSProp vs best AdamW."""
    # Find best Magma
    best_tau, best_lr_magma, best_loss = None, None, np.inf
    for tau in TAUS:
        key = f"magma_block_tau{tau}"
        if key not in results[HESSIAN]:
            continue
        for lr in LRS:
            if lr not in results[HESSIAN][key]:
                continue
            finals = get_final_losses(results[HESSIAN][key][lr][K])
            med = np.nanmedian(finals)
            if med < best_loss:
                best_loss = med
                best_tau = tau
                best_lr_magma = lr

    # Find best baseline LRs
    def find_best_lr(opt_name):
        best, blr = np.inf, None
        for lr in LRS:
            if lr not in results[HESSIAN].get(opt_name, {}):
                continue
            finals = get_final_losses(results[HESSIAN][opt_name][lr][K])
            med = np.nanmedian(finals)
            if med < best:
                best, blr = med, lr
        return blr

    best_lr_rmsprop = find_best_lr("rmsprop")
    best_lr_adamw = find_best_lr("adamw")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    magma_key = f"magma_block_tau{best_tau}"

    for ax, xlim, title in [
        (axes[0], 500, "First 500 iterations (paper's Figure 4)"),
        (axes[1], 10000, "Full 10000 iterations"),
    ]:
        idx_lim = xlim // LOG_EVERY

        for opt_key, lr, label, color in [
            ("adamw", best_lr_adamw, f"AdamW (lr={best_lr_adamw})", "#1f77b4"),
            ("rmsprop", best_lr_rmsprop, f"RMSProp (lr={best_lr_rmsprop})", "#9467bd"),
            (magma_key, best_lr_magma, f"Magma(tau={best_tau}, lr={best_lr_magma})", "#2ca02c"),
        ]:
            if lr is None or opt_key not in results[HESSIAN]:
                continue
            data = results[HESSIAN][opt_key][lr][K]
            losses = data["losses"][:, :idx_lim]
            iters = np.arange(losses.shape[1]) * LOG_EVERY

            median = np.nanmedian(losses, axis=0)
            q25 = np.nanpercentile(losses, 25, axis=0)
            q75 = np.nanpercentile(losses, 75, axis=0)

            ax.semilogy(iters, median, label=label, color=color, linewidth=2)
            ax.fill_between(iters, q25, q75, alpha=0.15, color=color)

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Loss (median, 20 seeds)")
        ax.set_title(title)
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Magma Reproduction: Best Config vs Baselines", fontsize=13)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")

    return best_tau, best_lr_magma


# ---------------------------------------------------------------------------
# Plot 3: Alignment score evolution
# ---------------------------------------------------------------------------

def plot_alignment_scores(results, best_tau, best_lr, filename):
    """Alignment score evolution for best tau vs tau=2.0."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    for ax, tau, title in [
        (axes[0], 2.0, f"tau=2.0 (default, lr={best_lr})"),
        (axes[1], best_tau, f"tau={best_tau} (best, lr={best_lr})"),
    ]:
        key = f"magma_block_tau{tau}"
        if key not in results[HESSIAN]:
            continue
        if best_lr not in results[HESSIAN][key]:
            continue
        data = results[HESSIAN][key][best_lr][K]

        if "alignment_scores" not in data:
            ax.text(0.5, 0.5, "No alignment data", ha="center", va="center",
                    transform=ax.transAxes)
            continue

        # alignment_scores: (n_seeds, n_steps, n_blocks)
        scores = data["alignment_scores"]
        # Take median across seeds
        median_scores = np.nanmedian(scores, axis=0)  # (n_steps, n_blocks)
        iters = np.arange(median_scores.shape[0]) * LOG_EVERY

        block_labels = ["Block 0 ({1,99,4998})", "Block 1 ({2,100,4999})",
                        "Block 2 ({3,101,5000})"]
        colors = ["#e41a1c", "#377eb8", "#4daf4a"]

        for b in range(median_scores.shape[1]):
            ax.plot(iters, median_scores[:, b], label=block_labels[b],
                    color=colors[b], linewidth=2)
            # Show IQR
            q25 = np.nanpercentile(scores[:, :, b], 25, axis=0)
            q75 = np.nanpercentile(scores[:, :, b], 75, axis=0)
            ax.fill_between(iters, q25, q75, alpha=0.1, color=colors[b])

        ax.set_xlabel("Iteration")
        ax.set_ylabel("Alignment Score s_t")
        ax.set_title(title)
        ax.set_ylim(0, 1)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.3, label="s=0.5")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Alignment Score Evolution: Default vs Best Tau", fontsize=13)
    plt.tight_layout()

    path = os.path.join(RESULTS_DIR, filename)
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Saved: {path}")


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def print_summary(results):
    """Print summary of all Magma configs vs baselines."""
    print(f"\n{'='*90}")
    print(f"Magma Reproduction Summary (heterogeneous, k=3, 20 seeds)")
    print(f"{'='*90}")

    # Baselines
    print(f"\n--- Baselines ---")
    print(f"{'Optimizer':<25} {'Best LR':>8} {'Med@500':>12} {'Med@10k':>12}")
    print("-" * 60)
    for opt in ["rmsprop", "adamw"]:
        if opt not in results[HESSIAN]:
            continue
        best_lr, best_med = None, np.inf
        for lr in LRS:
            if lr not in results[HESSIAN][opt]:
                continue
            finals = get_final_losses(results[HESSIAN][opt][lr][K])
            med = np.nanmedian(finals)
            if med < best_med:
                best_med, best_lr = med, lr

        if best_lr is None:
            continue
        data = results[HESSIAN][opt][best_lr][K]
        med_500 = np.nanmedian(get_loss_at_iter(data, 500))
        med_10k = np.nanmedian(get_final_losses(data))
        print(f"{opt:<25} {best_lr:>8} {med_500:>12.6f} {med_10k:>12.6f}")

    # Magma per tau
    print(f"\n--- Magma(block) per tau ---")
    print(f"{'Tau':>6} {'Best LR':>8} {'Med@500':>12} {'Med@10k':>12} {'Wins vs AdamW @10k':>20}")
    print("-" * 70)

    # Get best AdamW for comparison
    best_adamw_lr, best_adamw_loss = None, np.inf
    for lr in LRS:
        if lr not in results[HESSIAN].get("adamw", {}):
            continue
        finals = get_final_losses(results[HESSIAN]["adamw"][lr][K])
        med = np.nanmedian(finals)
        if med < best_adamw_loss:
            best_adamw_loss, best_adamw_lr = med, lr

    for tau in TAUS:
        key = f"magma_block_tau{tau}"
        if key not in results[HESSIAN]:
            continue
        best_lr, best_med = None, np.inf
        for lr in LRS:
            if lr not in results[HESSIAN][key]:
                continue
            finals = get_final_losses(results[HESSIAN][key][lr][K])
            med = np.nanmedian(finals)
            if med < best_med:
                best_med, best_lr = med, lr

        if best_lr is None:
            continue
        data = results[HESSIAN][key][best_lr][K]
        med_500 = np.nanmedian(get_loss_at_iter(data, 500))
        med_10k = np.nanmedian(get_final_losses(data))

        # Paired comparison with best AdamW at matched seeds
        if best_adamw_lr is not None:
            adamw_finals = get_final_losses(results[HESSIAN]["adamw"][best_adamw_lr][K])
            magma_finals = get_final_losses(data)
            n = min(len(adamw_finals), len(magma_finals))
            wins = np.sum(magma_finals[:n] < adamw_finals[:n])
            wins_str = f"{wins}/{n}"
        else:
            wins_str = "N/A"

        print(f"{tau:>6} {best_lr:>8} {med_500:>12.6f} {med_10k:>12.6f} {wins_str:>20}")

    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    results = load_results("magma_reproduction.npz")

    print_summary(results)

    plot_heatmap(results, "magma_tau_lr_heatmap_10k.png")
    plot_heatmap(results, "magma_tau_lr_heatmap_500.png", eval_iter=500)

    best_tau, best_lr = plot_loss_curves(results, "magma_loss_curves.png")

    plot_alignment_scores(results, best_tau, best_lr, "magma_alignment_scores.png")

    print(f"\nBest Magma config: tau={best_tau}, lr={best_lr}")
    print("All plots saved to:", RESULTS_DIR)


if __name__ == "__main__":
    main()
