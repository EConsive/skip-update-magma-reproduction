# SkipUpdate / Magma Reproduction: Heterogeneous Quadratic Benchmark

Reproduction and extension of the Section 4.4 experiment from ["On Surprising Effectiveness of Masking Updates in Adaptive Optimizers"](https://arxiv.org/abs/2602.15322) (the Magma paper).

## Motivation

The Magma paper introduces **SkipUpdate** (randomly masking parameter updates with 1/p rescaling) and **Magma** (SkipUpdate + momentum-gradient alignment). Both surprisingly outperform standard optimizers like Adam and Muon in LLM pretraining. But *why* does randomly skipping updates help?

This repository reproduces the paper's synthetic quadratic benchmark (Section 4.4) and extends it with experiments the paper doesn't include, specifically to test two hypotheses:

1. **Momentum interaction**: skipping updates while keeping dense momentum accumulates better gradient estimates before committing
2. **Curvature heterogeneity**: SkipUpdate helps specifically when wildly different curvatures are mixed within parameter blocks

## The Benchmark

A 9-dimensional quadratic `L(w) = 1/2 w^T H w` with block-diagonal Hessian (3 blocks of 3x3). Eigenvalues span 3 orders of magnitude: {1, 2, 3, 99, 100, 101, 4998, 4999, 5000}.

- **Homogeneous**: similar eigenvalues grouped per block ({1,2,3}, {99,100,101}, {4998,4999,5000})
- **Heterogeneous**: mixed eigenvalues per block ({1,99,4998}, {2,100,4999}, {3,101,5000})

Stochastic gradients via row-subsampling of X = H^{1/2}.

## What We Test (Beyond the Paper)

The paper only compares AdamW vs Magma on this benchmark. We add:

| Optimizer | What it tests |
|-----------|--------------|
| SGD | Baseline without momentum or adaptivity |
| SGD + Momentum | Isolates momentum contribution |
| AdamW | Paper's baseline |
| RMSProp | Base optimizer for SkipUpdate/Magma |
| SkipUpdate(RMSProp, block/element) | **Key missing experiment** from the paper |
| SkipUpdate(AdamW, block/element) | Does base optimizer matter? |
| SkipUpdate(SGD) | Does SkipUpdate work without momentum? |
| SkipUpdate(SGD+Momentum) | Does it need adaptive LR or just momentum? |
| Magma(block/element) | Paper's method, for validation |

All with LR sweeps, 5 seeds, and subsampling sweep k={1,3,5}.

## Files

- `benchmark.py` - Quadratic benchmark (Hessian construction, stochastic gradients)
- `optimizers.py` - All optimizer implementations
- `run_experiments.py` - Experiment runner, config, result serialization
- `plot_results.py` - Plotting and summary tables
- `results/` - Saved data (.npz) and plots (.png)
- `results_analysis.md` - Detailed findings

## Running

```bash
pip install numpy scipy matplotlib

# Run full sweep (~8 minutes, 1680 configurations)
python run_experiments.py

# Generate all plots
python plot_results.py
```

## Key Findings

See [results_analysis.md](results_analysis.md) for the full analysis. The headline result:

**SkipUpdate works when and only when the base optimizer uses momentum for the update direction.** SkipUpdate(AdamW) beats AdamW; SkipUpdate(SGD) is catastrophically worse than SGD. The mechanism is "accumulate momentum over skipped steps, then commit a better-averaged update."
