"""
2D diagonal sweep — does the SkipUpdate advantage scale with condition number?

Move 1 (run_coupling_2d_unequal.py) showed that:
  * with H = [[1, c], [c, 100]], SkipUpdate(SGD+Mom, elem) beats SGD+Mom by
    ~10-20% across all c,
  * but the c=0 (zero coupling, axis-aligned) baseline already shows the win,
  * so the off-diagonal coupling isn't doing the work.

This script strips the off-diagonal entirely:
  H = diag(1, b),  b in {10, 100, 1000, 10000, 100000}.

Hypothesis: the SkipUpdate advantage on diagonal quadratics scales with
condition number b/a. If the ratio drops monotonically as b grows, the
mechanism is "damped LR helps on high-cond anisotropic landscapes" — same
asymptotic-noise-floor story we already attributed to Magma.

Reduced N_ITERS = 1000 for speed. At this horizon and the LRs that turn out
to be optimal, the slow direction (lambda=1) is still in its transient
(time scale ~ 1/(eta*1) ~ several thousand iters), so the loss is
transient-dominated, not noise-floor-dominated. If the gap still shows up,
that's evidence that SkipUpdate also helps during the transient on
high-condition-number landscapes, not just at the asymptotic floor.
"""

import os
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from optimizers import SGDMomentum, SkipUpdate
from run_experiments import LOG_EVERY, RESULTS_DIR

DIM = 2
A_DIAG = 1.0
B_VALUES = [10.0, 100.0, 1000.0, 10000.0, 100000.0]
NOISE_SIGMA = 1.0
NUM_SEEDS = 30
N_ITERS = 1000
LRS = np.logspace(-7, -1, 25).tolist()         # 1e-7 to 1e-1
W_INIT = np.array([1.0, 1.0])


def make_H(b):
    return np.diag([A_DIAG, b])


def loss_fn(w, H):
    return 0.5 * w @ H @ w


def noisy_grad(w, H, rng, sigma):
    return H @ w + sigma * rng.standard_normal(len(w))


def ema_smooth(x, alpha=0.05):
    out = np.empty_like(x)
    out[0] = x[0]
    for i in range(1, len(x)):
        out[i] = alpha * x[i] + (1 - alpha) * out[i - 1] if np.isfinite(x[i]) else out[i - 1]
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
    total = len(B_VALUES) * len(LRS) * NUM_SEEDS * 2
    print(f"2D diagonal condition-number sweep: {total} runs")
    print(f"  H = diag({A_DIAG}, b),  b in {B_VALUES}")
    print(f"  sigma={NOISE_SIGMA}, {NUM_SEEDS} seeds, {N_ITERS} iters, {len(LRS)} LRs")
    print(f"  w_init={W_INIT.tolist()}")

    results = {b: {"sgdmom": {}, "skipupdate": {}} for b in B_VALUES}

    for b in B_VALUES:
        H = make_H(b)
        cond = b / A_DIAG
        print(f"  b={b}: cond={cond:.0f}")
        for lr in LRS:
            sgd_opt = lambda _lr=lr: SGDMomentum(DIM, _lr)
            su_opt = lambda _lr=lr: SkipUpdate(SGDMomentum(DIM, _lr), masking="element")
            results[b]["sgdmom"][lr] = run_seeds(sgd_opt, H, NOISE_SIGMA)
            results[b]["skipupdate"][lr] = run_seeds(su_opt, H, NOISE_SIGMA)
        print(f"    done ({time.time() - t0:.0f}s)")

    print(f"\n{'=' * 84}")
    print(f"Best-vs-best comparison (median final loss, {NUM_SEEDS} seeds, {N_ITERS} iters)")
    print(f"{'=' * 84}")
    print(f"{'b':>10} {'cond':>10} {'SGD+Mom LR':>12} {'SGD+Mom':>14} "
          f"{'SkipU LR':>12} {'SkipU':>14} {'Ratio':>8}")
    print("-" * 84)

    best = {}
    for b in B_VALUES:
        sgd_finals = {lr: median_final(results[b]["sgdmom"][lr]) for lr in LRS}
        su_finals = {lr: median_final(results[b]["skipupdate"][lr]) for lr in LRS}
        best_sgd_lr = min(sgd_finals, key=lambda k: sgd_finals[k] if np.isfinite(sgd_finals[k]) else np.inf)
        best_su_lr = min(su_finals, key=lambda k: su_finals[k] if np.isfinite(su_finals[k]) else np.inf)
        sgd_loss = sgd_finals[best_sgd_lr]
        su_loss = su_finals[best_su_lr]
        ratio = su_loss / sgd_loss if sgd_loss > 0 else np.inf
        winner = "SkipU" if ratio < 1.0 else "SGD+M"
        print(f"{b:>10.0f} {b / A_DIAG:>10.0f} "
              f"{best_sgd_lr:>12.6f} {sgd_loss:>14.8f} "
              f"{best_su_lr:>12.6f} {su_loss:>14.8f} {ratio:>8.3f} {winner}")
        best[b] = {
            "sgd_lr": best_sgd_lr, "sgd_loss": sgd_loss,
            "su_lr": best_su_lr, "su_loss": su_loss,
            "ratio": ratio,
            "sgd_finals": sgd_finals, "su_finals": su_finals,
        }

    # Plot 1: LR sweep per b
    fig, axes = plt.subplots(1, len(B_VALUES), figsize=(4 * len(B_VALUES), 5),
                             sharey=False)
    for i, b in enumerate(B_VALUES):
        ax = axes[i]
        lr_arr = np.array(LRS)
        sgd_arr = np.array([best[b]["sgd_finals"][lr] for lr in LRS])
        su_arr = np.array([best[b]["su_finals"][lr] for lr in LRS])
        ax.loglog(lr_arr, sgd_arr, "o-", color="#1f77b4",
                  label="SGD+Momentum", linewidth=2, markersize=5)
        ax.loglog(lr_arr, su_arr, "s-", color="#d62728",
                  label="SkipUpdate(SGD+Mom,elem)", linewidth=2, markersize=5)
        ax.axvline(best[b]["sgd_lr"], color="#1f77b4", linestyle=":", alpha=0.4)
        ax.axvline(best[b]["su_lr"], color="#d62728", linestyle=":", alpha=0.4)
        ax.set_title(f"b={b:.0f} (cond={b/A_DIAG:.0f})", fontsize=10)
        ax.set_xlabel("Learning Rate")
        if i == 0:
            ax.set_ylabel("Median Final Loss")
        ax.grid(True, alpha=0.3)
        if i == 0:
            ax.legend(fontsize=9)
    fig.suptitle(f"2D Diagonal: LR sweeps ({NUM_SEEDS} seeds, sigma={NOISE_SIGMA}, "
                 f"{N_ITERS} iters)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "coupling_2d_diag_lr_sweep.png")
    fig.savefig(path, dpi=150)
    print(f"\nSaved: {path}")

    # Plot 2: Best-vs-best ratio as a function of cond number
    fig, ax = plt.subplots(figsize=(8, 5))
    cond_arr = np.array([b / A_DIAG for b in B_VALUES])
    ratios = np.array([best[b]["ratio"] for b in B_VALUES])
    ax.semilogx(cond_arr, ratios, "o-", color="#8e44ad", linewidth=2, markersize=8)
    ax.axhline(1.0, color="gray", linestyle="--", alpha=0.6, label="parity")
    ax.set_xlabel(f"Condition number b/a  (a={A_DIAG})")
    ax.set_ylabel("SkipUpdate / SGD+Momentum (best-vs-best)")
    ax.set_title(f"Best-vs-best ratio vs condition number, diagonal H "
                 f"({NUM_SEEDS} seeds, {N_ITERS} iters, sigma={NOISE_SIGMA})")
    ax.grid(True, alpha=0.3)
    ax.legend()
    for i, b in enumerate(B_VALUES):
        ax.annotate(f"{ratios[i]:.2f}", (cond_arr[i], ratios[i]),
                    textcoords="offset points", xytext=(8, 4), fontsize=9)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "coupling_2d_diag_ratio.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")

    # Plot 3: Convergence curves at best LRs
    fig, axes = plt.subplots(1, len(B_VALUES), figsize=(4 * len(B_VALUES), 5),
                             sharey=False)
    for i, b in enumerate(B_VALUES):
        ax = axes[i]
        for name, key, lr_key, color in [
            ("SGD+Mom", "sgdmom", "sgd_lr", "#1f77b4"),
            ("SkipU(elem)", "skipupdate", "su_lr", "#d62728"),
        ]:
            lr = best[b][lr_key]
            losses = results[b][key][lr]
            iters = np.arange(losses.shape[1]) * LOG_EVERY
            med = ema_smooth(np.nanmedian(losses, axis=0))
            q25 = ema_smooth(np.nanpercentile(losses, 25, axis=0))
            q75 = ema_smooth(np.nanpercentile(losses, 75, axis=0))
            ax.semilogy(iters, med, label=f"{name} (lr={lr:.6f})",
                        color=color, linewidth=2)
            ax.fill_between(iters, q25, q75, alpha=0.15, color=color)
        ax.set_title(f"b={b:.0f}", fontsize=10)
        ax.set_xlabel("Gradient Steps")
        if i == 0:
            ax.set_ylabel("Loss (EMA-smoothed median)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
    fig.suptitle(f"2D Diagonal: convergence at best LRs ({NUM_SEEDS} seeds, "
                 f"{N_ITERS} iters)", fontsize=12)
    plt.tight_layout()
    path = os.path.join(RESULTS_DIR, "coupling_2d_diag_curves.png")
    fig.savefig(path, dpi=150)
    print(f"Saved: {path}")
    print(f"Total: {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
