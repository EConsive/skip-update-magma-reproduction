"""
Plot Section 4.3 results: in-context linear regression, light vs heavy regime.

Generates:
  1. incontext_curves.png    — best-LR loss trajectory per optimizer per regime
                                (matches Magma paper Fig. 3 top)
  2. incontext_lr_sweep.png  — final loss vs LR per optimizer per regime
  3. incontext_ratio.png     — best-vs-best ratio bar chart per regime

Reads from results/incontext_regression_passA.npz (Pass A) and optionally
results/incontext_regression_passB.npz (if Pass B was run).
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_experiments import RESULTS_DIR  # noqa: E402

# Optimizer display order, labels, and colors
OPT_STYLE = {
    "adamw":                  ("AdamW",                "#1f77b4", "o"),
    "magma_adamw_block":      ("Magma(AdamW, block)",  "#d62728", "s"),
    "skipupdate_adamw_elem":  ("SkipUpdate(AdamW)",    "#2ca02c", "^"),
    "rmsprop":                ("RMSProp",              "#7f7f7f", "v"),
}


def load_passes():
    """Load Pass A npz and (if present) merge Pass B."""
    path_a = os.path.join(RESULTS_DIR, "incontext_regression_passA.npz")
    if not os.path.exists(path_a):
        sys.exit(f"Pass A results not found: {path_a}")
    a = dict(np.load(path_a, allow_pickle=True))
    path_b = os.path.join(RESULTS_DIR, "incontext_regression_passB.npz")
    if os.path.exists(path_b):
        b = dict(np.load(path_b, allow_pickle=True))
        for k, v in b.items():
            if "__" in k:  # only merge per-(regime,opt,lr) arrays
                a[k] = v
    return a


def parse_results(blob):
    """Return: regimes, lrs, log_every, n_iters, dict[(regime,opt)] -> {lr: arr}."""
    regimes = [str(r) for r in blob["regimes"]]
    lrs = list(blob["lrs"])
    log_every = int(blob["log_every"])
    n_iters = int(blob["n_iters"])
    out = {}
    for k in blob:
        if "__" not in k:
            continue
        regime, opt, lr_tag = k.split("__")
        lr_i = int(lr_tag.replace("lr", ""))
        out.setdefault((regime, opt), {})[lrs[lr_i]] = blob[k]
    # Discover which optimizers are present
    opts_present = sorted({o for (_, o) in out})
    return regimes, lrs, log_every, n_iters, out, opts_present


def final_metric(losses_mat):
    tail = losses_mat[:, -5:]
    per_seed = np.nanmean(tail, axis=1)
    return np.nanmedian(per_seed)


def best_lr(per_lr_dict):
    return min(per_lr_dict, key=lambda k: per_lr_dict[k] if np.isfinite(per_lr_dict[k]) else np.inf)


def ema(x, alpha=0.2):
    out = np.empty_like(x, dtype=float)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1] if np.isfinite(x[i]) else out[i - 1]
    return out


def plot_curves(regimes, lrs, log_every, n_iters, data, opts, out_path):
    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 5), sharey=False)
    if len(regimes) == 1:
        axes = [axes]
    for ax, regime in zip(axes, regimes):
        for opt in opts:
            label, color, _ = OPT_STYLE.get(opt, (opt, None, "o"))
            per_lr = {lr: final_metric(data[(regime, opt)][lr]) for lr in lrs}
            blr = best_lr(per_lr)
            losses_mat = data[(regime, opt)][blr]
            iters = np.arange(losses_mat.shape[1]) * log_every
            med = ema(np.nanmedian(losses_mat, axis=0))
            q25 = ema(np.nanpercentile(losses_mat, 25, axis=0))
            q75 = ema(np.nanpercentile(losses_mat, 75, axis=0))
            ax.semilogy(iters, med, color=color, linewidth=2, label=f"{label} (lr={blr:.4f})")
            ax.fill_between(iters, q25, q75, color=color, alpha=0.15)
        ax.set_title(f"regime: {regime}", fontsize=12)
        ax.set_xlabel("Gradient steps")
        ax.set_ylabel("Population loss (median, IQR)")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("§4.3 In-Context Linear Regression — best-LR loss trajectories", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def plot_lr_sweep(regimes, lrs, data, opts, out_path):
    fig, axes = plt.subplots(1, len(regimes), figsize=(6 * len(regimes), 5), sharey=False)
    if len(regimes) == 1:
        axes = [axes]
    for ax, regime in zip(axes, regimes):
        for opt in opts:
            label, color, marker = OPT_STYLE.get(opt, (opt, None, "o"))
            finals = np.array([final_metric(data[(regime, opt)][lr]) for lr in lrs])
            ax.loglog(lrs, finals, marker=marker, color=color, linewidth=2,
                      markersize=6, label=label)
        ax.set_title(f"regime: {regime}", fontsize=12)
        ax.set_xlabel("Learning rate")
        ax.set_ylabel("Median tail loss")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)
    fig.suptitle("§4.3 In-Context Linear Regression — LR sweep", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def plot_ratio(regimes, lrs, data, opts, out_path):
    if "adamw" not in opts:
        return
    others = [o for o in opts if o != "adamw"]
    if not others:
        return
    fig, ax = plt.subplots(figsize=(7, 5))
    width = 0.8 / len(others)
    x_idx = np.arange(len(regimes))
    for i, opt in enumerate(others):
        label, color, _ = OPT_STYLE.get(opt, (opt, None, "o"))
        ratios = []
        for regime in regimes:
            per_lr_a = {lr: final_metric(data[(regime, "adamw")][lr]) for lr in lrs}
            per_lr_o = {lr: final_metric(data[(regime, opt)][lr]) for lr in lrs}
            best_a = per_lr_a[best_lr(per_lr_a)]
            best_o = per_lr_o[best_lr(per_lr_o)]
            ratios.append(best_o / best_a if best_a > 0 else np.nan)
        offsets = (i - (len(others) - 1) / 2) * width
        bars = ax.bar(x_idx + offsets, ratios, width=width, color=color, label=label)
        for x, r in zip(x_idx + offsets, ratios):
            ax.text(x, r, f"{r:.2f}", ha="center", va="bottom", fontsize=9)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.5, label="parity (= AdamW)")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(regimes)
    ax.set_ylabel("best-vs-best loss / AdamW best-vs-best loss")
    ax.set_title("§4.3 — Ratio to AdamW (lower = better than AdamW)")
    ax.set_yscale("log")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    blob = load_passes()
    regimes, lrs, log_every, n_iters, data, opts = parse_results(blob)
    print(f"Loaded {len(opts)} optimizers across {len(regimes)} regimes, {len(lrs)} LRs")
    print(f"Optimizers: {opts}")

    plot_curves(regimes, lrs, log_every, n_iters, data, opts,
                os.path.join(RESULTS_DIR, "incontext_curves.png"))
    plot_lr_sweep(regimes, lrs, data, opts,
                  os.path.join(RESULTS_DIR, "incontext_lr_sweep.png"))
    plot_ratio(regimes, lrs, data, opts,
               os.path.join(RESULTS_DIR, "incontext_ratio.png"))

    # Print summary
    print()
    print("=" * 80)
    print("Best-vs-best summary")
    print("=" * 80)
    for regime in regimes:
        print(f"\n[{regime}]")
        for opt in opts:
            per_lr = {lr: final_metric(data[(regime, opt)][lr]) for lr in lrs}
            blr = best_lr(per_lr)
            label = OPT_STYLE.get(opt, (opt,))[0]
            print(f"  {label:<25} best LR={blr:.5f}  loss={per_lr[blr]:.6f}")
        if "adamw" in opts:
            base = final_metric(data[(regime, "adamw")][best_lr({lr: final_metric(data[(regime, 'adamw')][lr]) for lr in lrs})])
            for opt in opts:
                if opt == "adamw":
                    continue
                per_lr = {lr: final_metric(data[(regime, opt)][lr]) for lr in lrs}
                v = per_lr[best_lr(per_lr)]
                print(f"  ratio {opt}/adamw = {v / base:.3f}")


if __name__ == "__main__":
    main()
