"""
2D coupling experiment: does SkipUpdate's advantage scale with off-diagonal
coupling strength?

Setup:
  H = [[1, c], [c, 1]] — identical diagonals, varying off-diagonal c
  Gradient noise: additive Gaussian, sigma=1.0
  Baseline: SGD+Momentum
  Test: SkipUpdate(SGD+Momentum, element-wise)
  Sweep c ∈ {0.0, 0.5, 0.9, 0.95, 0.99}
  Fine LR grid (25 log-spaced), 30 seeds, 10k iters, best-vs-best per c.

Prediction under coupling hypothesis: SkipUpdate advantage grows with c.
Prediction under noise-suppression hypothesis: invariant to c.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Element-wise masking doesn't depend on BLOCK_SLICES, so no patching needed.
from optimizers import SGDMomentum, SkipUpdate
from run_experiments import LOG_EVERY, RESULTS_DIR

DIM = 2
COUPLINGS = [0.0, 0.5, 0.9, 0.95, 0.99]
NOISE_SIGMA = 1.0
NUM_SEEDS = 30
N_ITERS = 10000
LRS = np.logspace(-4, 0.3, 25).tolist()  # 1e-4 to ~2
W_INIT = np.array([1.0, 0.0])  # has components along both eigenvectors


def make_H(c):
    return np.array([[1.0, c], [c, 1.0]])


def loss_fn(w, H):
    return 0.5 * w @ H @ w


def noisy_grad(w, H, rng, sigma):
    return H @ w + sigma * rng.standard_normal(len(w))


def ema_smooth(x, alpha=0.02):
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i-1] if np.isfinite(x[i]) else out[i-1]
    return out


def run_one(optimizer, H, seed, sigma):
    rng = np.random.default_rng(seed * 10000 + 9999)
    w = W_INIT.copy()
    losses = []
    for t in range(N_ITERS):
        if t % LOG_EVERY == 0:
            losses.append(loss_fn(w, H))
        g = noisy_grad(w, H, rng, sigma)
        w = optimizer.step(w, g, rng)
        if not np.isfinite(w).all() or loss_fn(w, H) > 1e20:
            remaining = (N_ITERS - t - 1) // LOG_EVERY
            losses.extend([float("nan")] * remaining)
            break
    return np.array(losses)


def run_seeds(make_opt, H, sigma):
    all_losses = []
    for seed in range(NUM_SEEDS):
        losses = run_one(make_opt(), H, seed, sigma)
        all_losses.append(losses)
    max_len = max(len(l) for l in all_losses)
    padded = [np.concatenate([l, np.full(max_len - len(l), np.nan)])
              if len(l) < max_len else l for l in all_losses]
    return np.stack(padded)


def median_final(losses):
    finals = [l[np.isfinite(l)][-1] if np.any(np.isfinite(l)) else np.nan for l in losses]
    return np.nanmedian(finals)


def main():
    t0 = time.time()
    total = len(COUPLINGS) * len(LRS) * NUM_SEEDS * 2
    print(f"2D coupling experiment: {total} runs")
    print(f"  couplings={COUPLINGS}, sigma={NOISE_SIGMA}, {NUM_SEEDS} seeds, "
          f"{N_ITERS} iters, {len(LRS)} LRs")
    print(f"  w_init={W_INIT.tolist()}")

    # results[c]["sgdmom" or "skipupdate"][lr] = losses array
    results = {c: {"sgdmom": {}, "skipupdate": {}} for c in COUPLINGS}

    for c in COUPLINGS:
        H = make_H(c)
        lam_max = 1 + c
        lam_min = 1 - c
        print(f"  c={c}: lambda=[{lam_min:.3f}, {lam_max:.3f}], "
              f"cond={lam_max/max(lam_min,1e-6):.1f}")
        for lr in LRS:
            sgd_opt = lambda _lr=lr: SGDMomentum(DIM, _lr)
            su_opt = lambda _lr=lr: SkipUpdate(SGDMomentum(DIM, _lr), masking="element")
            results[c]["sgdmom"][lr] = run_seeds(sgd_opt, H, NOISE_SIGMA)
            results[c]["skipupdate"][lr] = run_seeds(su_opt, H, NOISE_SIGMA)
        print(f"    done ({time.time()-t0:.0f}s)")

    # Best-vs-best per coupling
    print(f"\n{'='*78}")
    print(f"Best-vs-best comparison (median final loss, 30 seeds)")
    print(f"{'='*78}")
    print(f"{'c':>6} {'SGD+Mom LR':>12} {'SGD+Mom':>14} "
          f"{'SkipU LR':>12} {'SkipU':>14} {'Ratio':>8}")
    print("-" * 78)

    best = {}
    for c in COUPLINGS:
        sgd_finals = {lr: median_final(results[c]["sgdmom"][lr]) for lr in LRS}
        su_finals = {lr: median_final(results[c]["skipupdate"][lr]) for lr in LRS}
        best_sgd_lr = min(sgd_finals, key=lambda k: sgd_finals[k] if np.isfinite(sgd_finals[k]) else np.inf)
        best_su_lr = min(su_finals, key=lambda k: su_finals[k] if np.isfinite(su_finals[k]) else np.inf)
        sgd_loss = sgd_finals[best_sgd_lr]
        su_loss = su_finals[best_su_lr]
        ratio = su_loss / sgd_loss if sgd_loss > 0 else np.inf
        winner = "SkipU" if ratio < 1.0 else "SGD+M"
        print(f"{c:>6.2f} {best_sgd_lr:>12.5f} {sgd_loss:>14.8f} "
              f"{best_su_lr:>12.5f} {su_loss:>14.8f} {ratio:>8.3f} {winner}")
        best[c] = {
            "sgd_lr": best_sgd_lr, "sgd_loss": sgd_loss,
            "su_lr": best_su_lr, "su_loss": su_loss,
            "ratio": ratio,
            "sgd_finals": sgd_finals, "su_finals": su_finals,
        }

    # Plot 1: LR sweep per coupling
    fig, axes = plt.subplots(1, len(COUPLINGS), figsize=(5*len(COUPLINGS), 5), sharey=True)
    for i, c in enumerate(COUPLINGS):
        ax = axes[i]
        lr_arr = np.array(LRS)
        sgd_arr = np.array([best[c]["sgd_finals"][lr] for lr in LRS])
        su_arr = np.array([best[c]["su_finals"][lr] for lr in LRS])
        ax.loglog(lr_arr, sgd_arr, "o-", color="#1f77b4", label="SGD+Momentum", linewidth=2, markersize=5)
        ax.loglog(lr_arr, su_arr, "s-", color="#d62728", label="SkipUpdate(SGD+Mom,elem)", linewidth=2, markersize=5)
        ax.axvline(best[c]["sgd_lr"], color="#1f77b4", linestyle=":", alpha=0.4)
        ax.axvline(best[c]["su_lr"], color="#d62728", linestyle=":", alpha=0.4)
        ax.set_title(f"c={c} (cond={((1+c)/max(1-c,1e-6)):.0f})", fontsize=11)
        ax.set_xlabel("Learning Rate")
        if i == 0:
            ax.set_ylabel("Median Final Loss")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
    fig.suptitle(f"2D Coupling: LR sweeps ({NUM_SEEDS} seeds, sigma={NOISE_SIGMA})", fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "coupling_2d_lr_sweep.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved: {path}")

    # Plot 2: Best-vs-best ratio as a function of coupling
    fig, ax = plt.subplots(figsize=(8, 5))
    c_arr = np.array(COUPLINGS)
    ratios = np.array([best[c]["ratio"] for c in COUPLINGS])
    ax.semilogy(c_arr, ratios, "o-", color="#8e44ad", linewidth=2, markersize=8)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="equal")
    ax.set_xlabel("Coupling strength c")
    ax.set_ylabel("SkipUpdate / SGD+Momentum (best-vs-best)")
    ax.set_title(f"Best-vs-best ratio vs coupling ({NUM_SEEDS} seeds, sigma={NOISE_SIGMA})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    for i, c in enumerate(COUPLINGS):
        ax.annotate(f"{ratios[i]:.2f}", (c_arr[i], ratios[i]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "coupling_2d_ratio.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")

    # Plot 3: Convergence curves at best LRs
    fig, axes = plt.subplots(1, len(COUPLINGS), figsize=(5*len(COUPLINGS), 5), sharey=True)
    for i, c in enumerate(COUPLINGS):
        ax = axes[i]
        for name, key, lr_key, loss_key, color in [
            ("SGD+Mom", "sgdmom", "sgd_lr", "sgd_loss", "#1f77b4"),
            ("SkipU(elem)", "skipupdate", "su_lr", "su_loss", "#d62728"),
        ]:
            lr = best[c][lr_key]
            losses = results[c][key][lr]
            iters = np.arange(losses.shape[1]) * LOG_EVERY
            med = ema_smooth(np.nanmedian(losses, axis=0))
            q25 = ema_smooth(np.nanpercentile(losses, 25, axis=0))
            q75 = ema_smooth(np.nanpercentile(losses, 75, axis=0))
            ax.semilogy(iters, med, label=f"{name} (lr={lr:.4f})",
                        color=color, linewidth=2)
            ax.fill_between(iters, q25, q75, alpha=0.15, color=color)
        ax.set_title(f"c={c}", fontsize=11)
        ax.set_xlabel("Gradient Steps")
        if i == 0:
            ax.set_ylabel("Loss (EMA-smoothed median)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"2D Coupling: convergence at best LRs ({NUM_SEEDS} seeds)", fontsize=13)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "coupling_2d_curves.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    print(f"Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
