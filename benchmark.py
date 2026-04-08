"""
Quadratic benchmark from Section 4.4 / Appendix B.3 of the Magma paper.

L(w) = 1/2 * w^T H w  in R^9, with block-diagonal Hessian.
Stochastic gradients via row-subsampling of X = H^{1/2}.
"""

import numpy as np
from scipy.linalg import block_diag


# Eigenvalue assignments
EIGENVALUES = [1, 2, 3, 99, 100, 101, 4998, 4999, 5000]

HOMOGENEOUS_BLOCKS = [
    [1, 2, 3],
    [99, 100, 101],
    [4998, 4999, 5000],
]

HETEROGENEOUS_BLOCKS = [
    [1, 99, 4998],
    [2, 100, 4999],
    [3, 101, 5000],
]

BLOCK_SIZE = 3
NUM_BLOCKS = 3
DIM = BLOCK_SIZE * NUM_BLOCKS  # 9


def random_orthogonal(n, rng):
    """Generate a random orthogonal matrix via QR decomposition."""
    A = rng.standard_normal((n, n))
    Q, R = np.linalg.qr(A)
    # Ensure uniform distribution over O(n) by fixing sign ambiguity
    Q = Q @ np.diag(np.sign(np.diag(R)))
    return Q


class QuadraticBenchmark:
    """
    9D quadratic benchmark: L(w) = 1/2 w^T H w.

    H is block-diagonal with 3 blocks of 3x3, each randomly rotated.
    Stochastic gradients from subsampling rows of X = H^{1/2}.
    """

    def __init__(self, hessian_type, seed=0):
        assert hessian_type in ("homogeneous", "heterogeneous")
        self.hessian_type = hessian_type
        self.seed = seed

        rng = np.random.default_rng(seed)

        if hessian_type == "homogeneous":
            block_eigs = HOMOGENEOUS_BLOCKS
        else:
            block_eigs = HETEROGENEOUS_BLOCKS

        # Build each block: Q @ diag(eigs) @ Q^T
        blocks = []
        for eigs in block_eigs:
            Q = random_orthogonal(BLOCK_SIZE, rng)
            block = Q @ np.diag(eigs) @ Q.T
            blocks.append(block)

        self.H = block_diag(*blocks)

        # X = H^{1/2}, computed per-block for numerical stability
        sqrt_blocks = []
        rng2 = np.random.default_rng(seed)  # same seed to get same Q
        for eigs in block_eigs:
            Q = random_orthogonal(BLOCK_SIZE, rng2)
            sqrt_block = Q @ np.diag(np.sqrt(eigs)) @ Q.T
            sqrt_blocks.append(sqrt_block)

        self.X = block_diag(*sqrt_blocks)  # (9, 9)

    def loss(self, w):
        """True loss: L(w) = 1/2 w^T H w."""
        return 0.5 * w @ self.H @ w

    def full_gradient(self, w):
        """True gradient: H @ w."""
        return self.H @ w

    def stochastic_gradient(self, w, subsample_k, rng):
        """
        Stochastic gradient by subsampling k rows of X = H^{1/2}.

        g = (n/k) * X_S^T @ X_S @ w, where n=9, so E[g] = H @ w.
        """
        n = self.X.shape[0]
        indices = rng.choice(n, size=subsample_k, replace=False)
        X_S = self.X[indices]  # (k, 9)
        return (n / subsample_k) * (X_S.T @ X_S @ w)

    def initial_point(self, rng):
        """Standard initial point: all ones."""
        return np.ones(DIM)
