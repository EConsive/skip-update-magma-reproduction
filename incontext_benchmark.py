"""
In-context linear regression benchmark (Section 4.3 of Magma paper, from Ahn et al. 2024).

Task
----
Each training step samples a fresh prompt containing n=20 in-context pairs
(x_i, y_i) with y_i = <w*, x_i>, plus a query x_query. The model predicts
y_query; loss is squared error.

Model
-----
Single-layer linear self-attention, reduced to the canonical *preconditioner*
parameterization from Ahn et al. 2024 ("Transformers learn to implement
preconditioned gradient descent for in-context linear regression"):

    y_hat = (1/n) * y^T X W x_query

where W in R^{d x d} is the learned preconditioner.

Ahn et al. show this is exactly equivalent to one step of preconditioned GD
from W_init = 0 on the in-context empirical risk, and that population-optimal
W for isotropic covariates is a scalar multiple of the identity.

    --- Deviation from initial plan ---
The original plan sketched a single merged matrix P in R^{(d+1) x (d+1)} with
prediction e^T P M v (extract last entry). That formulation has a *rank-1*
gradient in which only the last ROW of P ever receives signal, because the
scalar output e^T(.) collapses all upstream rows. The preconditioner form
above has non-degenerate gradients (every entry of W sees signal at every
step) and is the canonical object Ahn et al. actually analyze. For d=5 this
gives 25 parameters.

Gradient (analytic, hand-derived)
---------------------------------
Let a = (1/n) * X^T y in R^d, then y_hat = a^T W x_query is linear in W, so

    dL/dW = (y_hat - y_query) * a * x_query^T      (shape d x d)

Data regimes (Magma Appendix B.4)
---------------------------------
  light:  x_i ~ N(0, I_d),   w* ~ N(0, I_d)
  heavy:  x_i = u_i * sqrt(gamma_i),
          u_i ~ Uniform(S^{d-1}),
          gamma_i ~ Gamma(shape=0.1, scale=10),
          w* ~ N(0, I_d)

Both regimes have E[x x^T] = I_d (heavy-tailed regime: E[gamma]=1 and
E[u u^T]=I_d/d, so E[x x^T]=I_d/d * E[gamma] * d = I_d). The heavy-tailed
regime produces rare prompts with enormous ||X||, which drives heavy-tailed
gradient noise -- the setting where the Magma paper claims its
curvature-aware masking should beat Adam.
"""

import numpy as np

DIM_D = 5
CONTEXT_N = 20
PARAM_DIM = DIM_D * DIM_D  # 25


def sample_x_light(n, d, rng):
    """x_i ~ N(0, I_d)."""
    return rng.standard_normal((n, d))


def sample_x_heavy(n, d, rng):
    """x_i = u_i * sqrt(gamma_i), u_i ~ Uniform(S^{d-1}), gamma_i ~ Gamma(0.1, 10)."""
    u = rng.standard_normal((n, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    gamma = rng.gamma(shape=0.1, scale=10.0, size=(n, 1))
    return u * np.sqrt(gamma)


class LinearAttentionRegression:
    """In-context linear regression benchmark (light- vs heavy-tailed covariates)."""

    def __init__(self, regime, d=DIM_D, n=CONTEXT_N):
        assert regime in ("light", "heavy")
        self.regime = regime
        self.d = d
        self.n = n
        self.param_dim = d * d
        self._sample_x = sample_x_light if regime == "light" else sample_x_heavy

    def sample_prompt(self, rng):
        """Sample (X, y, x_query, y_query) for a fresh prompt."""
        w_star = rng.standard_normal(self.d)
        X = self._sample_x(self.n, self.d, rng)
        y = X @ w_star
        x_q = self._sample_x(1, self.d, rng).ravel()
        y_q = float(w_star @ x_q)
        return X, y, x_q, y_q

    def predict(self, W_flat, X, y, x_q):
        W = W_flat.reshape(self.d, self.d)
        a = X.T @ y / self.n
        return float(a @ W @ x_q)

    def loss_on_prompt(self, W_flat, X, y, x_q, y_q):
        y_hat = self.predict(W_flat, X, y, x_q)
        return 0.5 * (y_hat - y_q) ** 2

    def stochastic_gradient(self, W_flat, rng):
        """One fresh prompt; returns flattened gradient of length d*d."""
        X, y, x_q, y_q = self.sample_prompt(rng)
        W = W_flat.reshape(self.d, self.d)
        a = X.T @ y / self.n
        y_hat = float(a @ W @ x_q)
        grad = (y_hat - y_q) * np.outer(a, x_q)
        return grad.ravel()

    def loss(self, W_flat, rng, n_eval=1024):
        """Population-loss estimate (for logging only)."""
        total = 0.0
        for _ in range(n_eval):
            X, y, x_q, y_q = self.sample_prompt(rng)
            total += self.loss_on_prompt(W_flat, X, y, x_q, y_q)
        return total / n_eval

    def initial_point(self, rng):
        return np.zeros(self.param_dim)


# ---------------------------------------------------------------------------
# Gradient correctness check via finite differences
# ---------------------------------------------------------------------------

def _gradient_check(regime, n_trials=10, eps=1e-5, tol=1e-5):
    benchmark = LinearAttentionRegression(regime)
    rng = np.random.default_rng(42)
    max_rel_err = 0.0
    for trial in range(n_trials):
        X, y, x_q, y_q = benchmark.sample_prompt(rng)
        W_flat = rng.standard_normal(benchmark.param_dim) * 0.3
        a = X.T @ y / benchmark.n
        y_hat = a @ W_flat.reshape(benchmark.d, benchmark.d) @ x_q
        g_ana = ((y_hat - y_q) * np.outer(a, x_q)).ravel()
        g_fd = np.zeros_like(W_flat)
        for i in range(len(W_flat)):
            Wp = W_flat.copy(); Wp[i] += eps
            Wm = W_flat.copy(); Wm[i] -= eps
            lp = benchmark.loss_on_prompt(Wp, X, y, x_q, y_q)
            lm = benchmark.loss_on_prompt(Wm, X, y, x_q, y_q)
            g_fd[i] = (lp - lm) / (2 * eps)
        num = np.linalg.norm(g_ana - g_fd)
        den = np.linalg.norm(g_fd) + 1e-12
        max_rel_err = max(max_rel_err, num / den)
    return max_rel_err


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("Gradient correctness check (analytic vs finite differences)")
    print("=" * 60)
    for regime in ("light", "heavy"):
        err = _gradient_check(regime)
        status = "PASS" if err < 1e-5 else "FAIL"
        print(f"  [{regime:6}] max rel err = {err:.3e}  [{status}]")
        assert err < 1e-5

    print()
    print("=" * 60)
    print("Smoke run: AdamW 500 steps, both regimes")
    print("=" * 60)
    from optimizers import AdamW
    for regime in ("light", "heavy"):
        benchmark = LinearAttentionRegression(regime)
        rng = np.random.default_rng(0)
        W = benchmark.initial_point(rng)
        opt = AdamW(benchmark.param_dim, lr=0.03)
        eval_rng_seed = 99
        loss0 = benchmark.loss(W, np.random.default_rng(eval_rng_seed), n_eval=512)
        print(f"  [{regime:6}] iter    0: loss = {loss0:.5f}")
        for t in range(1, 501):
            g = benchmark.stochastic_gradient(W, rng)
            W = opt.step(W, g, rng)
            if t % 100 == 0:
                lt = benchmark.loss(W, np.random.default_rng(eval_rng_seed), n_eval=512)
                print(f"  [{regime:6}] iter {t:4d}: loss = {lt:.5f}")
    print()
    print("Optimal W is approximately (1/d) * I_d; optimal loss approaches 0.")
