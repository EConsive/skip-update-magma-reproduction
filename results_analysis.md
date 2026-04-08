# Results: SkipUpdate on Heterogeneous Quadratics

## Experiment Setup

- **Problem**: 9D quadratic L(w) = 1/2 w^T H w, block-diagonal Hessian
- **Eigenvalues**: {1, 2, 3, 99, 100, 101, 4998, 4999, 5000} (3 orders of magnitude)
- **Homogeneous**: similar eigenvalues per block; **Heterogeneous**: mixed eigenvalues per block
- **Stochasticity**: row-subsampling of X = H^{1/2}, sweep k = {1, 3, 5}
- **12 optimizer variants**, 4-5 LRs each, 5 seeds, 10000 iterations
- **Total**: 1680 individual runs

## Summary Table (Heterogeneous Hessian, k=3, Ranked by Final Loss)

| Rank | Optimizer | Final Loss | Best LR |
|------|-----------|-----------|---------|
| 1 | **SkipUpdate(AdamW, block)** | **0.0031** | 0.03 |
| 2 | SkipUpdate(SGD+Mom) | 0.0045 | 0.0002 |
| 3 | AdamW | 0.0063 | 0.01 |
| 4 | SGD+Momentum | 0.0104 | 0.0002 |
| 5 | RMSProp | 0.0161 | 0.03 |
| 6 | SkipUpdate(AdamW, elem) | 0.0750 | 0.03 |
| 7 | SGD | 0.0854 | 0.0001 |
| 8 | SkipUpdate(RMSProp, block) | 0.1453 | 0.01 |
| 9 | Magma(block) | 0.2586 | 0.03 |
| 10 | Magma(elem) | 0.3848 | 0.03 |
| 11 | SkipUpdate(RMSProp, elem) | 0.4444 | 0.003 |
| 12 | SkipUpdate(SGD) | 0.9723 | 3e-05 |

---

## Key Finding 1: SkipUpdate Needs Momentum in the Update Direction

The most striking pattern in the results:

| Base Optimizer | Without SkipUpdate | With SkipUpdate | Effect |
|----------------|-------------------|-----------------|--------|
| SGD (no momentum) | 0.085 | 0.972 | **11x worse** |
| SGD + Momentum | 0.010 | 0.005 | **2x better** |
| AdamW (has momentum) | 0.006 | 0.003 | **2x better** |
| RMSProp (no momentum in update) | 0.016 | 0.145 | **9x worse** |

**SkipUpdate is catastrophically bad without momentum** (SGD: 11x worse, RMSProp: 9x worse). But with momentum, it consistently helps (SGD+Mom: 2x better, AdamW: 2x better).

### Why?

The mechanism is clear:
- When an update is **skipped**, momentum keeps accumulating gradient information at the same parameter values
- When the update **is applied** (with 2x rescaling), it uses a momentum-smoothed direction
- Without momentum, the 2x rescaling just amplifies raw stochastic gradient noise

**RMSProp is a subtle case**: it maintains a first-moment EMA internally (for Magma's alignment score), but its *update direction* uses the raw gradient g/sqrt(v), not momentum. So SkipUpdate's rescaling amplifies raw gradient noise, making it worse.

**AdamW** uses momentum for its update direction (mu_hat/sqrt(v_hat)), making each applied update more reliable even with 2x rescaling.

---

## Key Finding 2: Block-wise Masking Beats Element-wise

| Optimizer | Block-wise | Element-wise | Ratio |
|-----------|-----------|-------------|-------|
| SkipUpdate(AdamW) | 0.003 | 0.075 | 25x better |
| SkipUpdate(RMSProp) | 0.145 | 0.444 | 3x better |
| Magma | 0.259 | 0.385 | 1.5x better |

Block-wise masking is consistently better. On this benchmark (3 blocks, 3 elements each), element-wise masking has much higher variance (9 independent coin flips vs 3). The paper's LLM ablation showed near-equivalence, likely because with many more elements per block the variance difference is negligible.

---

## Key Finding 3: Magma Doesn't Reproduce the Paper's Advantage

The paper claims Magma outperforms AdamW on the heterogeneous quadratic. We find the opposite:

| Optimizer | Final Loss |
|-----------|-----------|
| AdamW | 0.006 |
| Magma(block) | 0.259 |

The alignment score s_t stays around 0.5 throughout training (see alignment_scores.png), creating a ~4x effective learning rate reduction (s * p = 0.5 * 0.5 = 0.25). Even with our extended LR grid up to 0.3, Magma can't overcome this damping.

Possible explanations for the discrepancy with the paper:
1. Unspecified hyperparameters (subsampling fraction, beta values)
2. Different number of iterations or LR range
3. Magma may need specific tuning (temperature, EMA coefficient) for this problem scale

---

## Key Finding 4: Noise Level Matters

### k=1 (maximum noise, 1/9 rows subsampled)

Everything struggles on heterogeneous. Best is SkipUpdate(AdamW,block) at 0.40 — even AdamW only reaches 1.03. High noise overwhelms all methods.

### k=3 (moderate noise, 1/3 rows)

The sweet spot where differences emerge clearly. SkipUpdate(AdamW,block) at 0.003 beats AdamW at 0.006.

### k=5 (low noise, 5/9 rows)

AdamW at 0.00001 is nearly perfect. SkipUpdate(AdamW) is slightly worse at 0.014. With low noise, the "accumulate better estimates" benefit of SkipUpdate is less needed.

Full table across noise levels:

| Optimizer | k=1 | k=3 | k=5 |
|-----------|-----|-----|-----|
| AdamW | 1.032 | 0.006 | 0.00001 |
| SkipUpdate(AdamW,block) | 0.395 | 0.003 | 0.014 |
| RMSProp | 0.574 | 0.016 | 0.033 |
| Magma(block) | 1.127 | 0.259 | 0.147 |
| SkipUpdate(SGD+Mom) | 0.830 | 0.005 | 0.012 |

**SkipUpdate's advantage is largest at moderate-to-high noise** (k=1 and k=3), and diminishes with low noise (k=5). This is consistent with the momentum-accumulation hypothesis: when gradients are noisy, averaging over more observations before committing helps most.

---

## Plots

All plots are in `results/`:

| Plot | Description |
|------|------------|
| `fig4_reproduction.png` | Figure 4 reproduction: AdamW vs Magma across LRs |
| `main_comparison.png` | All 12 optimizers at their best LR |
| `granularity_comparison.png` | Block vs element masking for SkipUpdate and Magma |
| `momentum_hypothesis.png` | SGD vs SGD+Mom vs SkipUpdate(SGD) vs SkipUpdate(SGD+Mom) |
| `base_optimizer_effect.png` | SkipUpdate on different base optimizers |
| `subsampling_sweep.png` | Effect of noise level (k=1, 3, 5) |
| `alignment_scores.png` | Gradient-momentum alignment over training (Magma) |

---

## Deep Dive: SGD+Momentum vs SkipUpdate(SGD+Momentum)

The initial 5-seed sweep suggested SkipUpdate(SGD+Mom) beats SGD+Mom (0.005 vs 0.010 at lr=0.0002). To test this rigorously, we ran a **focused experiment with 13 LRs and 30 seeds**.

### The initial finding was misleading

The 5-seed result compared both methods at lr=0.0002 — but this is near SkipUpdate's optimum while being far from SGD+Mom's. At their **respective best LRs**:

| Method | Best LR | Median Final Loss | IQR |
|--------|---------|-------------------|-----|
| SGD+Momentum | 0.00035 | **0.000644** | [0.000190, 0.001366] |
| SkipUpdate(SGD+Mom) | 0.00020 | 0.013625 | [0.004662, 0.025387] |

**SGD+Momentum is 21x better** when each method uses its optimal LR.

### Why: the stability boundary is halved

SkipUpdate's 2x rescaling means every applied step is twice as large. This halves the maximum stable learning rate:
- SGD+Mom: stable up to lr ~0.00035 (some divergence at 0.00032)
- SkipUpdate(SGD+Mom): stable up to lr ~0.00020 (diverges starting at 0.00022)

SGD+Mom achieves its best performance at LRs (0.00028-0.00035) that SkipUpdate **cannot access at all**.

### At matched LRs, SkipUpdate is neutral to slightly worse

Paired-seed comparison (same seed, same Hessian realization):

| LR | SkipUpdate wins (out of 30) | Median loss ratio (SU/SM) |
|----|-----------------------------|---------------------------|
| 0.00005 | 13/30 | 1.00 |
| 0.00008 | 11/30 | 1.10 |
| 0.00010 | 12/30 | 1.11 |
| 0.00012 | 13/30 | 1.09 |
| 0.00015 | 12/30 | 1.13 |
| 0.00018 | 13/30 | 1.09 |
| 0.00020 | 11/30 | 1.32 |
| 0.00022 | 7/27 | 2.45 |
| 0.00025 | 3/19 | 5.32 |

At low LRs (0.00005-0.00015), the two methods are **statistically indistinguishable** — SkipUpdate wins ~40-43% of seeds with ratio ~1.1x. At higher LRs, SGD+Mom increasingly dominates because SkipUpdate approaches its stability boundary.

There is **no LR where SkipUpdate reliably beats SGD+Momentum** on this benchmark.

### Plots

| Plot | Description |
|------|-------------|
| `focused_lr_sweep.png` | Final loss vs LR with IQR bands (30 seeds) |
| `focused_paired_comparison.png` | Paired-seed scatter plots at 6 LRs |
| `focused_trajectories.png` | Loss trajectories at lr=0.0001, 0.0002, 0.0003 |

---

## Implications for Understanding Why SkipUpdate Works

### What the initial sweep suggested (with caveats)

The broad 5-seed sweep found:
1. SkipUpdate hurts without momentum (SGD: 11x worse)
2. SkipUpdate on RMSProp is worse (because RMSProp uses raw gradient, not momentum, for updates)
3. SkipUpdate(AdamW,block) beats AdamW at fixed LR

### What the deep dive on SGD+Momentum revealed

**The apparent benefit of SkipUpdate was an artifact of comparing at the wrong LR.** When both methods use their optimal LR, plain SGD+Momentum is 21x better.

The core issue is that SkipUpdate's 2x rescaling **halves the stability boundary**, preventing the optimizer from using the aggressive LRs where the real gains are. The hypothesized "accumulate better momentum" benefit exists at best weakly (ratio ~1.1 at low LRs) and is overwhelmed by the stability cost.

### Open questions

1. **Does this pattern hold for AdamW?** The broad sweep showed SkipUpdate(AdamW,block) at lr=0.03 beating AdamW at lr=0.01 — but this needs the same fine-grained validation with 30+ seeds
2. **Adaptive methods may behave differently** — Adam's per-parameter scaling might absorb the 2x rescaling better than SGD
3. **This is a 9D quadratic** — the dynamics in billion-parameter transformers are fundamentally different (heterogeneous block structure, heavy-tailed noise, etc.)

The Magma-specific alignment mechanism (damping based on gradient-momentum cosine similarity) does not show a clear benefit on this benchmark, at least with the default hyperparameters and our LR grid. The damping appears to be too aggressive for a 9D problem.

---

## Caveats

1. **This is a 9D quadratic** — very different from billion-parameter transformer training
2. The paper's unspecified hyperparameters (subsampling fraction, iteration count) may account for the Magma discrepancy
3. Block structure (3 blocks × 3 elements) is far smaller than real neural network blocks
4. The multiplicative noise structure (vanishing at optimum) differs from real training noise
5. **The LR stability boundary issue may not apply to adaptive methods** — Adam/RMSProp auto-scale steps, so the 2x rescaling may be absorbed differently
