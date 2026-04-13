"""
In-context linear regression benchmark (§4.3 Magma paper, from Ahn et al. 2024).

Architecture (Ahn et al. 2024, Eq. 1 and §3.1)
----------------------------------------------
Single-layer linear self-attention.

    Attn_{P,Q}(Z) = P · Z · M · (Z^T Q Z),     M = diag(I_n, 0) ∈ R^{(n+1)×(n+1)}
    Z_1 = Z_0 + (1/n) · Attn_{P,Q}(Z_0)
    TF(Z_0) = -[Z_1]_{(d+1), (n+1)}            (with a sign flip, per Ahn et al.)

Since Z_0[d+1, n+1] = 0 (the query y slot), we have:

    y_hat = -(1/n) · e_{d+1}^T · P · (Z_c Z_c^T) · Q · z_query

where z_k = [x_k; y_k] ∈ R^{d+1}, Z_c = [z_1 ... z_n] ∈ R^{(d+1)×n},
z_query = [x_query; 0], and e_{d+1} extracts the last (y) coordinate.

Identifiability reduction. Only the LAST ROW of P enters the loss, because
e_{d+1}^T P selects one row. The other d rows of P are dead params (zero
gradient forever). We therefore store only the last row as `row_p ∈ R^{d+1}`
(6 entries for d=5) and the full Q ∈ R^{(d+1)×(d+1)} (36 entries for d=5).

    total params = (d+1) + (d+1)^2 = 42 for d=5.

Compact forward pass:

    z_q = [x_q; 0]                        (d+1,)
    G   = Z_c Z_c^T                       (d+1, d+1)
    u   = Q z_q                           (d+1,)
    v   = G u                             (d+1,)
    y_hat = -(1/n) · row_p · v

Gradient (hand-derived, linear in row_p and Q)
----------------------------------------------
    dL/d y_hat = (y_hat - y_q)

    d y_hat / d row_p = -(1/n) · v           → d L / d row_p = (y_hat - y_q) · (-1/n) · v
    d y_hat / d Q     = -(1/n) · outer(b, z_q)  where b = G · row_p
                                             → d L / d Q = (y_hat - y_q) · (-1/n) · b ⊗ z_q

Dead-init note
--------------
(row_p, Q) = (0, 0) is a dead saddle: all gradients are zero. We therefore
initialize `row_p = e_{d+1}` (last coordinate = 1) and Q = 0. At this init
y_hat = 0 and dL/dQ = (y_q/n) · outer(G row_p, z_q) ≠ 0, so Q trains
immediately. row_p then starts getting gradient as soon as Q becomes nonzero.

This init is also very close to the optimal for this task: at the population
optimum, row_p = e_{d+1} and Q_xx = -I_d (see Ahn et al.'s preconditioned-GD
interpretation).

Data regimes (Magma Appendix B.4)
---------------------------------
    light:  x_i ~ N(0, I_d),    w* ~ N(0, I_d)
    heavy:  x_i = u_i · sqrt(γ_i),   u_i ~ Unif(S^{d-1}),
            γ_i ~ Gamma(shape=0.1, scale=10),    w* ~ N(0, I_d)

Both regimes have E[x x^T] = I_d; heavy has heavy-tailed radii.
"""

import numpy as np

DIM_D = 5
CONTEXT_N = 20
ROW_DIM = DIM_D + 1           # 6
Q_DIM = (DIM_D + 1) ** 2      # 36
PARAM_DIM = ROW_DIM + Q_DIM   # 42


def sample_x_light(n, d, rng):
    return rng.standard_normal((n, d))


def sample_x_heavy(n, d, rng):
    u = rng.standard_normal((n, d))
    u /= np.linalg.norm(u, axis=1, keepdims=True) + 1e-12
    gamma = rng.gamma(shape=0.1, scale=10.0, size=(n, 1))
    return u * np.sqrt(gamma)


def _unpack(params):
    row_p = params[:ROW_DIM]
    Q = params[ROW_DIM:].reshape(DIM_D + 1, DIM_D + 1)
    return row_p, Q


def _pack(row_p, Q):
    return np.concatenate([row_p, Q.ravel()])


class LinearAttentionRegression:
    """In-context linear regression benchmark (Ahn et al. 2024 architecture)."""

    def __init__(self, regime, d=DIM_D, n=CONTEXT_N):
        assert regime in ("light", "heavy")
        self.regime = regime
        self.d = d
        self.n = n
        self.dp1 = d + 1
        self.param_dim = (d + 1) + (d + 1) ** 2
        self._sample_x = sample_x_light if regime == "light" else sample_x_heavy

    def sample_prompt(self, rng):
        """Sample (Z_context, x_query, y_query). Z_context is (d+1, n)."""
        w_star = rng.standard_normal(self.d)
        X = self._sample_x(self.n, self.d, rng)         # (n, d)
        y = X @ w_star                                    # (n,)
        Z_c = np.vstack([X.T, y[np.newaxis, :]])          # (d+1, n)
        x_q = self._sample_x(1, self.d, rng).ravel()
        y_q = float(w_star @ x_q)
        return Z_c, x_q, y_q

    def _forward(self, row_p, Q, Z_c, x_q):
        z_q = np.concatenate([x_q, [0.0]])        # (d+1,)
        G = Z_c @ Z_c.T                            # (d+1, d+1)
        u = Q @ z_q                                # (d+1,)
        v = G @ u                                  # (d+1,)
        y_hat = -(1.0 / self.n) * (row_p @ v)
        return y_hat, G, u, v, z_q

    def predict(self, params, Z_c, x_q):
        row_p, Q = _unpack(params)
        y_hat, *_ = self._forward(row_p, Q, Z_c, x_q)
        return y_hat

    def loss_on_prompt(self, params, Z_c, x_q, y_q):
        y_hat = self.predict(params, Z_c, x_q)
        return 0.5 * (y_hat - y_q) ** 2

    def stochastic_gradient(self, params, rng):
        """One fresh prompt; returns flat gradient of length PARAM_DIM."""
        Z_c, x_q, y_q = self.sample_prompt(rng)
        row_p, Q = _unpack(params)
        y_hat, G, u, v, z_q = self._forward(row_p, Q, Z_c, x_q)
        err = y_hat - y_q
        # d y_hat / d row_p  = -(1/n) · v
        g_row = err * (-1.0 / self.n) * v
        # d y_hat / d Q      = -(1/n) · outer(b, z_q),  b = G @ row_p
        b = G @ row_p
        g_Q = err * (-1.0 / self.n) * np.outer(b, z_q)
        return _pack(g_row, g_Q)

    def loss(self, params, rng, n_eval=1024):
        total = 0.0
        for _ in range(n_eval):
            Z_c, x_q, y_q = self.sample_prompt(rng)
            total += self.loss_on_prompt(params, Z_c, x_q, y_q)
        return total / n_eval

    def initial_point(self, rng=None):
        """row_p = e_{d+1} (last coord 1), Q = 0. Near-optimal starting point."""
        params = np.zeros(self.param_dim)
        params[self.dp1 - 1] = 1.0   # row_p[d] = 1
        return params

    def optimum_point(self, rng=None):
        """row_p = e_{d+1}, Q_xx = -I_d, rest zero. Exact theoretical optimum
        (Ahn et al. 2024 §4: optimal single-layer linear attention implements
        one step of preconditioned GD from w=0). Use this to test noise-floor
        behaviour: gradient at the optimum is pure noise, so the optimizer's
        drift under heavy-tailed gradients is the only thing left to compete
        on.
        """
        row_p = np.zeros(self.dp1)
        row_p[-1] = 1.0
        Q = np.zeros((self.dp1, self.dp1))
        Q[:self.d, :self.d] = -np.eye(self.d)
        return _pack(row_p, Q)


# ---------------------------------------------------------------------------
# Gradient correctness check via finite differences
# ---------------------------------------------------------------------------

def _gradient_check(regime, n_trials=8, eps=1e-5):
    benchmark = LinearAttentionRegression(regime)
    rng = np.random.default_rng(42)
    max_rel = 0.0
    for _ in range(n_trials):
        Z_c, x_q, y_q = benchmark.sample_prompt(rng)
        # Random non-degenerate params so all gradients are nonzero
        params = rng.standard_normal(benchmark.param_dim) * 0.3
        # Analytic gradient via the benchmark (need to wrap to pass prompt)
        row_p, Q = _unpack(params)
        y_hat, G, u, v, z_q = benchmark._forward(row_p, Q, Z_c, x_q)
        err = y_hat - y_q
        g_row = err * (-1.0 / benchmark.n) * v
        b = G @ row_p
        g_Q = err * (-1.0 / benchmark.n) * np.outer(b, z_q)
        g_ana = _pack(g_row, g_Q)
        # Finite differences
        g_fd = np.zeros_like(params)
        for i in range(benchmark.param_dim):
            pp = params.copy(); pp[i] += eps
            pm = params.copy(); pm[i] -= eps
            lp = benchmark.loss_on_prompt(pp, Z_c, x_q, y_q)
            lm = benchmark.loss_on_prompt(pm, Z_c, x_q, y_q)
            g_fd[i] = (lp - lm) / (2 * eps)
        num = np.linalg.norm(g_ana - g_fd)
        den = np.linalg.norm(g_fd) + 1e-12
        max_rel = max(max_rel, num / den)
    return max_rel


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

    print("=" * 60)
    print("Gradient correctness check (FD vs analytic)")
    print("=" * 60)
    for regime in ("light", "heavy"):
        err = _gradient_check(regime)
        status = "PASS" if err < 1e-5 else "FAIL"
        print(f"  [{regime:6}] max rel err = {err:.3e}  [{status}]")
        assert err < 1e-5

    print()
    print("=" * 60)
    print("Smoke run: AdamW, 500 steps, both regimes")
    print(f"  PARAM_DIM = {PARAM_DIM}")
    print("=" * 60)
    from optimizers import AdamW
    for regime in ("light", "heavy"):
        benchmark = LinearAttentionRegression(regime)
        rng = np.random.default_rng(0)
        W = benchmark.initial_point(rng)
        opt = AdamW(benchmark.param_dim, lr=0.03)
        print(f"\n  [{regime}] init: row_p={W[:ROW_DIM]}")
        print(f"  [{regime}] init loss = "
              f"{benchmark.loss(W, np.random.default_rng(99), n_eval=256):.5f}")
        for t in range(1, 1001):
            g = benchmark.stochastic_gradient(W, rng)
            W = opt.step(W, g, rng)
            if t % 200 == 0:
                lt = benchmark.loss(W, np.random.default_rng(99), n_eval=256)
                row_p, Q = _unpack(W)
                Qxx = Q[:DIM_D, :DIM_D]
                qxx_diag = np.mean(np.diag(Qxx))
                qxx_off = (Qxx.sum() - np.trace(Qxx)) / (DIM_D * (DIM_D - 1))
                print(f"  [{regime}] iter {t:4d}: loss={lt:.5f}  "
                      f"|row_p-e|={np.linalg.norm(W[:ROW_DIM] - benchmark.initial_point()[:ROW_DIM]):.3f}  "
                      f"Qxx_diag_mean={qxx_diag:.3f}  Qxx_off_mean={qxx_off:.4f}")
    print()
    print("Expected optimum: row_p -> (0,0,0,0,0,1), Qxx -> -I_d (diag ≈ -1, off ≈ 0)")
