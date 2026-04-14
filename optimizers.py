"""
Optimizer implementations for the quadratic benchmark experiment.

Implements: SGD, SGD+Momentum, AdamW, RMSProp, SkipUpdate, Magma.
SkipUpdate/Magma support both block-wise and element-wise masking.
SkipUpdate can wrap any base optimizer (SGD, SGD+Momentum, RMSProp, AdamW).
"""

import numpy as np

BLOCK_SIZE = 3
NUM_BLOCKS = 3
BLOCK_SLICES = [slice(i * BLOCK_SIZE, (i + 1) * BLOCK_SIZE) for i in range(NUM_BLOCKS)]


# ---------------------------------------------------------------------------
# Base optimizers
# ---------------------------------------------------------------------------

class SGD:
    """Vanilla SGD: w -= lr * g."""

    def __init__(self, dim, lr):
        self.lr = lr
        self.name = "SGD"

    def step(self, w, grad, rng):
        return w - self.lr * grad


class SGDMomentum:
    """SGD with EMA momentum: mu = beta1*mu + (1-beta1)*g; w -= lr*mu."""

    def __init__(self, dim, lr, beta1=0.9):
        self.lr = lr
        self.beta1 = beta1
        self.mu = np.zeros(dim)
        self.name = "SGD+Momentum"

    def step(self, w, grad, rng):
        self.mu = self.beta1 * self.mu + (1 - self.beta1) * grad
        return w - self.lr * self.mu


class AdamW:
    """AdamW with bias correction. Weight decay lambda=0 by default."""

    def __init__(self, dim, lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.mu = np.zeros(dim)
        self.v = np.zeros(dim)
        self.t = 0
        self.name = "AdamW"

    def step(self, w, grad, rng):
        self.t += 1
        self.mu = self.beta1 * self.mu + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2

        mu_hat = self.mu / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)

        update = self.lr * mu_hat / (np.sqrt(v_hat) + self.eps)
        if self.weight_decay > 0:
            update += self.lr * self.weight_decay * w
        return w - update

    def compute_update(self, grad):
        """Compute the update vector without applying it. Used by Magma."""
        mu_hat = self.mu / (1 - self.beta1 ** self.t)
        v_hat = self.v / (1 - self.beta2 ** self.t)
        return self.lr * mu_hat / (np.sqrt(v_hat) + self.eps)


class RMSProp:
    """
    RMSProp: v = beta2*v + (1-beta2)*g^2; delta = lr*g/sqrt(v+eps).

    Also maintains a first-moment EMA (mu) for use by Magma's alignment score.
    """

    def __init__(self, dim, lr, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.mu = np.zeros(dim)
        self.v = np.zeros(dim)
        self.name = "RMSProp"

    def step(self, w, grad, rng):
        self.mu = self.beta1 * self.mu + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad ** 2
        update = self.lr * grad / (np.sqrt(self.v) + self.eps)
        return w - update

    def compute_update(self, grad):
        """Compute the update vector without applying it. Used by SkipUpdate/Magma."""
        return self.lr * grad / (np.sqrt(self.v) + self.eps)


# ---------------------------------------------------------------------------
# SkipUpdate wrapper (works with any base optimizer)
# ---------------------------------------------------------------------------

class SkipUpdate:
    """
    SkipUpdate: Bernoulli masking of updates with 1/p rescaling.

    Wraps a base optimizer. Dense momentum updates are maintained
    (base optimizer's internal state is always updated).

    masking: 'block' (mask entire 3D blocks) or 'element' (mask individual elements)
    """

    def __init__(self, base_optimizer, masking="block", p=0.5):
        self.base = base_optimizer
        self.masking = masking
        self.p = p
        self.scale = 1.0 / p  # = 2 for p=0.5
        self.name = f"SkipUpdate({self.base.name},{masking})"

    def step(self, w, grad, rng):
        # Always update base optimizer's internal state (dense momentum/v)
        w_full = self.base.step(w, grad, rng)
        update = w - w_full  # positive update vector (base wanted to subtract this)

        if self.masking == "block":
            mask = np.zeros(len(w))
            for s in BLOCK_SLICES:
                if rng.random() < self.p:
                    mask[s] = 1.0
        else:  # element-wise
            mask = (rng.random(len(w)) < self.p).astype(float)

        return w - self.scale * mask * update


# ---------------------------------------------------------------------------
# Magma wrapper (on top of RMSProp)
# ---------------------------------------------------------------------------

def cosine_similarity(a, b, eps=1e-8):
    """Cosine similarity between vectors a and b."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a < eps or norm_b < eps:
        return 0.0
    return np.dot(a, b) / (norm_a * norm_b)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -500, 500)))


class Magma:
    """
    Magma: momentum-aligned gradient masking on top of any adaptive optimizer.

    For each block:
      s_tilde = sigmoid(cossim(mu, g) / tau)
      s = ema_beta * s_prev + (1 - ema_beta) * s_tilde
      mask ~ Bernoulli(0.5)
      w_new = w - s * mask * delta

    masking: 'block' or 'element' (alignment always computed per block)
    Base can be RMSProp or AdamW.
    """

    def __init__(self, base_optimizer, masking="block", tau=2.0, ema_beta=0.9, p=0.5):
        self.base = base_optimizer
        self.masking = masking
        self.tau = tau
        self.ema_beta = ema_beta
        self.p = p
        self.s = np.ones(NUM_BLOCKS)  # alignment score EMA per block
        self.name = f"Magma({masking})"

    def step(self, w, grad, rng):
        # 1. Compute per-block alignment scores using OLD momentum (before update)
        for i, sl in enumerate(BLOCK_SLICES):
            cossim = cosine_similarity(self.base.mu[sl], grad[sl])
            s_tilde = sigmoid(cossim / self.tau)
            self.s[i] = self.ema_beta * self.s[i] + (1 - self.ema_beta) * s_tilde

        # 2. Update base optimizer state (dense momentum + v)
        if hasattr(self.base, 't'):
            self.base.t += 1  # needed for AdamW bias correction
        self.base.v = self.base.beta2 * self.base.v + (1 - self.base.beta2) * grad ** 2
        self.base.mu = self.base.beta1 * self.base.mu + (1 - self.base.beta1) * grad

        # 3. Compute base optimizer update
        update = self.base.compute_update(grad)

        # 4. Apply masked, alignment-scaled update
        w_new = w.copy()
        if self.masking == "block":
            for i, sl in enumerate(BLOCK_SLICES):
                if rng.random() < self.p:
                    w_new[sl] = w[sl] - self.s[i] * update[sl]
        else:  # element-wise masking, block-wise alignment
            elem_mask = (rng.random(len(w)) < self.p).astype(float)
            for i, sl in enumerate(BLOCK_SLICES):
                w_new[sl] = w[sl] - self.s[i] * elem_mask[sl] * update[sl]

        return w_new

    def get_alignment_scores(self):
        """Return current alignment scores for diagnostics."""
        return self.s.copy()


# ---------------------------------------------------------------------------
# Factory function
# ---------------------------------------------------------------------------

def create_optimizer(name, dim=9, lr=0.01, **kwargs):
    """
    Create an optimizer by name.

    Names:
      'sgd', 'sgd_momentum', 'adamw', 'rmsprop',
      'skipupdate_block', 'skipupdate_element',
      'skipupdate_adamw_block', 'skipupdate_adamw_element',
      'skipupdate_sgd', 'skipupdate_sgd_momentum',
      'magma_block', 'magma_element'
    """
    if name == "sgd":
        return SGD(dim, lr)
    elif name == "sgd_momentum":
        return SGDMomentum(dim, lr)
    elif name == "adamw":
        return AdamW(dim, lr)
    elif name == "rmsprop":
        return RMSProp(dim, lr)
    elif name == "skipupdate_block":
        return SkipUpdate(RMSProp(dim, lr), masking="block")
    elif name == "skipupdate_element":
        return SkipUpdate(RMSProp(dim, lr), masking="element")
    elif name == "skipupdate_adamw_block":
        return SkipUpdate(AdamW(dim, lr), masking="block")
    elif name == "skipupdate_adamw_element":
        return SkipUpdate(AdamW(dim, lr), masking="element")
    elif name == "skipupdate_sgd":
        return SkipUpdate(SGD(dim, lr), masking="element")
    elif name == "skipupdate_sgd_block":
        return SkipUpdate(SGD(dim, lr), masking="block")
    elif name == "skipupdate_sgd_momentum":
        return SkipUpdate(SGDMomentum(dim, lr), masking="element")
    elif name == "skipupdate_sgd_momentum_block":
        return SkipUpdate(SGDMomentum(dim, lr), masking="block")
    elif name == "magma_block":
        return Magma(RMSProp(dim, lr), masking="block", **kwargs)
    elif name == "magma_element":
        return Magma(RMSProp(dim, lr), masking="element", **kwargs)
    elif name == "magma_adamw_block":
        return Magma(AdamW(dim, lr), masking="block", **kwargs)
    elif name == "magma_adamw_element":
        return Magma(AdamW(dim, lr), masking="element", **kwargs)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


ALL_OPTIMIZERS = [
    "sgd",
    "sgd_momentum",
    "adamw",
    "rmsprop",
    "skipupdate_block",
    "skipupdate_element",
    "skipupdate_adamw_block",
    "skipupdate_adamw_element",
    "skipupdate_sgd",
    "skipupdate_sgd_momentum",
    "magma_block",
    "magma_element",
]
