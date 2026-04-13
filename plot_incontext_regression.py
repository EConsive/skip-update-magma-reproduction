"""
Plot Section 4.3 results: in-context linear regression, light vs heavy regime.

Loads the two Pass A files (optimum init + near-optimum init), both with the
fixed 7-block-of-6 BLOCK_SLICES (covering all 42 Ahn et al. params), and
produces 2x2 grids (rows = init, cols = regime).

Generates:
  1. incontext_lr_sweep.png  — final loss vs LR per optimizer (2x2 grid)
  2. incontext_curves.png    — best-LR loss trajectory per optimizer (2x2 grid)
  3. incontext_ratio.png     — Magma/AdamW ratio per (regime, init), 4 bars
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from run_experiments import RESULTS_DIR  # noqa: E402

OPT_STYLE = {
    "adamw":             ("AdamW",                "#1f77b4", "o"),
    "magma_adamw_block": ("Magma(AdamW, block)",  "#d62728", "s"),
}

INIT_FILES = [
    ("optimum",      "incontext_regression_passA.npz"),
    ("near-optimum", "incontext_regression_passA_nearinit.npz"),
]


def load_pass(filename):
    path = os.path.join(RESULTS_DIR, filename)
    if not os.path.exists(path):
        sys.exit(f"Required results file not found: {path}")
    return dict(np.load(path, allow_pickle=True))


def parse_results(blob):
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
    opts = sorted({o for (_, o) in out}, key=lambda o: 0 if o == "adamw" else 1)
    return regimes, lrs, log_every, n_iters, out, opts


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


def plot_lr_sweep_grid(parsed_by_init, out_path):
    inits = list(parsed_by_init.keys())
    regimes = parsed_by_init[inits[0]][0]
    fig, axes = plt.subplots(len(inits), len(regimes),
                             figsize=(5.5 * len(regimes), 4.5 * len(inits)),
                             sharex=True, sharey=False)
    if len(inits) == 1:
        axes = np.array([axes])
    if len(regimes) == 1:
        axes = axes[:, None]
    for r, init in enumerate(inits):
        regimes_i, lrs, _, _, data, opts = parsed_by_init[init]
        for c, regime in enumerate(regimes_i):
            ax = axes[r, c]
            for opt in opts:
                label, color, marker = OPT_STYLE.get(opt, (opt, None, "o"))
                finals = np.array([final_metric(data[(regime, opt)][lr]) for lr in lrs])
                ax.loglog(lrs, finals, marker=marker, color=color, linewidth=2,
                          markersize=6, label=label)
                blr = best_lr({lr: f for lr, f in zip(lrs, finals)})
                bf = finals[lrs.index(blr)]
                ax.scatter([blr], [bf], s=180, facecolor="none",
                           edgecolor=color, linewidth=2, zorder=5)
            ax.set_title(f"init: {init}   regime: {regime}", fontsize=11)
            if r == len(inits) - 1:
                ax.set_xlabel("Learning rate")
            if c == 0:
                ax.set_ylabel("Median tail loss")
            ax.grid(True, alpha=0.3)
            if r == 0 and c == 0:
                ax.legend(fontsize=9, loc="upper left")
    fig.suptitle("§4.3 In-Context Linear Regression — LR sweep "
                 "(circled = best LR per method)", fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def plot_curves_grid(parsed_by_init, out_path):
    inits = list(parsed_by_init.keys())
    regimes = parsed_by_init[inits[0]][0]
    fig, axes = plt.subplots(len(inits), len(regimes),
                             figsize=(5.5 * len(regimes), 4.5 * len(inits)),
                             sharex=True, sharey=False)
    if len(inits) == 1:
        axes = np.array([axes])
    if len(regimes) == 1:
        axes = axes[:, None]
    for r, init in enumerate(inits):
        regimes_i, lrs, log_every, _, data, opts = parsed_by_init[init]
        for c, regime in enumerate(regimes_i):
            ax = axes[r, c]
            for opt in opts:
                label, color, _ = OPT_STYLE.get(opt, (opt, None, "o"))
                per_lr = {lr: final_metric(data[(regime, opt)][lr]) for lr in lrs}
                blr = best_lr(per_lr)
                losses_mat = data[(regime, opt)][blr]
                iters = np.arange(losses_mat.shape[1]) * log_every
                med = ema(np.nanmedian(losses_mat, axis=0))
                q25 = ema(np.nanpercentile(losses_mat, 25, axis=0))
                q75 = ema(np.nanpercentile(losses_mat, 75, axis=0))
                ax.semilogy(iters, med, color=color, linewidth=2,
                            label=f"{label} (lr={blr:.4f})")
                ax.fill_between(iters, q25, q75, color=color, alpha=0.15)
            ax.set_title(f"init: {init}   regime: {regime}", fontsize=11)
            if r == len(inits) - 1:
                ax.set_xlabel("Gradient steps")
            if c == 0:
                ax.set_ylabel("Population loss (median, IQR)")
            ax.grid(True, alpha=0.3)
            ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("§4.3 In-Context Linear Regression — best-LR loss trajectories",
                 fontsize=13)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def plot_ratio(parsed_by_init, out_path):
    inits = list(parsed_by_init.keys())
    labels = []
    ratios = []
    for init in inits:
        regimes_i, lrs, _, _, data, _ = parsed_by_init[init]
        for regime in regimes_i:
            per_lr_a = {lr: final_metric(data[(regime, "adamw")][lr]) for lr in lrs}
            per_lr_m = {lr: final_metric(data[(regime, "magma_adamw_block")][lr])
                        for lr in lrs}
            best_a = per_lr_a[best_lr(per_lr_a)]
            best_m = per_lr_m[best_lr(per_lr_m)]
            ratios.append(best_m / best_a if best_a > 0 else np.nan)
            labels.append(f"{init}\n{regime}")
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(labels))
    colors = ["#d62728" if r < 1 else "#1f77b4" for r in ratios]
    ax.bar(x, ratios, color=colors, alpha=0.85)
    for xi, r in zip(x, ratios):
        ax.text(xi, r + 0.003, f"{r:.3f}", ha="center", va="bottom", fontsize=10)
    ax.axhline(1.0, color="black", linestyle="--", alpha=0.6,
               label="parity (= AdamW)")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Magma(AdamW, block) best-LR loss / AdamW best-LR loss")
    ax.set_title("§4.3 — Magma vs AdamW best-vs-best ratio "
                 "(red = Magma wins, blue = AdamW wins)")
    ax.set_ylim(0.9, max(1.1, max(ratios) * 1.05))
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved: {out_path}")


def main():
    parsed_by_init = {}
    for init_label, fname in INIT_FILES:
        blob = load_pass(fname)
        parsed_by_init[init_label] = parse_results(blob)
        regimes, lrs, _, _, data, opts = parsed_by_init[init_label]
        print(f"[{init_label}] regimes={regimes}, opts={opts}, {len(lrs)} LRs")

    plot_lr_sweep_grid(parsed_by_init,
                       os.path.join(RESULTS_DIR, "incontext_lr_sweep.png"))
    plot_curves_grid(parsed_by_init,
                     os.path.join(RESULTS_DIR, "incontext_curves.png"))
    plot_ratio(parsed_by_init,
               os.path.join(RESULTS_DIR, "incontext_ratio.png"))

    print()
    print("=" * 72)
    print("Best-vs-best summary")
    print("=" * 72)
    for init_label in parsed_by_init:
        regimes, lrs, _, _, data, opts = parsed_by_init[init_label]
        print(f"\n[init={init_label}]")
        for regime in regimes:
            best_a = None
            cells = []
            for opt in opts:
                per_lr = {lr: final_metric(data[(regime, opt)][lr]) for lr in lrs}
                blr = best_lr(per_lr)
                v = per_lr[blr]
                if opt == "adamw":
                    best_a = v
                cells.append(f"{opt}={v:.4f}@{blr:.5f}")
            ratio_m = None
            per_lr_m = {lr: final_metric(data[(regime, 'magma_adamw_block')][lr]) for lr in lrs}
            best_m = per_lr_m[best_lr(per_lr_m)]
            ratio_m = best_m / best_a
            print(f"  {regime:6}: " + "   ".join(cells) + f"   Magma/AdamW={ratio_m:.3f}")


if __name__ == "__main__":
    main()
