"""Matrix factorisation for implicit feedback: alternating least squares (Hu, Koren & Volinsky, 2008).

Every shopper u and product i get a vector (x_u, y_i). The model wants x_u . y_i close to 1
for products the shopper interacted with and close to 0 otherwise, trusting stronger signals
more: confidence c_ui = 1 + alpha * log(1 + w_ui), where w_ui sums event weights
(purchase 1, cart 0.3, view 0.05). It alternates between solving for all shoppers with
products fixed and for all products with shoppers fixed.

Each half-step is solved with a few conjugate-gradient iterations for all shoppers (or products)
at once, the same trick production libraries use. Only matrix products and sparse ops are
needed, so it is fast in plain numpy.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..eval.split import EVENT_WEIGHTS, TrainData

_BLOCK_ELEMENTS = 20_000_000  # memory cap for dense temporary blocks


class ALS:
    name = "Matrix factorisation (ALS)"

    def __init__(self, factors: int = 64, reg: float = 0.1, alpha: float = 10.0, iterations: int = 12,
                 cg_steps: int = 3, use_views: bool = True, seed: int = 0):
        self.factors = factors
        self.reg = reg
        self.alpha = alpha
        self.iterations = iterations
        self.cg_steps = cg_steps
        self.use_views = use_views
        self.seed = seed

    def fit(self, data: TrainData) -> "ALS":
        weights = EVENT_WEIGHTS if self.use_views else {"purchase": 1.0}
        w = data.interaction_matrix(weights).astype(np.float32)
        conf = w.copy()
        conf.data = 1.0 + self.alpha * np.log1p(conf.data)
        conf_t = conf.T.tocsr()

        rng = np.random.default_rng(self.seed)
        scale = 0.01
        self.user_factors = (rng.normal(0, scale, (data.n_users, self.factors))).astype(np.float32)
        self.item_factors = (rng.normal(0, scale, (data.n_items, self.factors))).astype(np.float32)
        self.loss_history = []
        for _ in range(self.iterations):
            self.user_factors = self._solve(conf, self.user_factors, self.item_factors)
            self.item_factors = self._solve(conf_t, self.item_factors, self.user_factors)
            self.loss_history.append(self.loss(conf))
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.user_factors[users] @ self.item_factors.T

    # ---- internals
    def _solve(self, conf: sp.csr_matrix, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """A few CG steps on (Y'Y + Y'(C_u - I)Y + reg I) x_u = Y' C_u p_u, for every row u at once."""
        yty = y.T @ y
        conf_minus_1 = conf.copy()
        conf_minus_1.data = conf_minus_1.data - 1.0

        def apply_a(v: np.ndarray) -> np.ndarray:
            inner = _values_at_nonzeros(conf, v, y)  # v_u . y_i for every observed (u, i)
            weighted = sp.csr_matrix((conf_minus_1.data * inner, conf.indices, conf.indptr), shape=conf.shape)
            return v @ yty + weighted @ y + self.reg * v

        b = conf @ y  # preferences are 1 on observed entries
        x = x.copy()
        r = b - apply_a(x)
        p = r.copy()
        rs = np.einsum("ij,ij->i", r, r)
        for _ in range(self.cg_steps):
            ap = apply_a(p)
            step = rs / np.maximum(np.einsum("ij,ij->i", p, ap), 1e-12)
            x += step[:, None] * p
            r -= step[:, None] * ap
            rs_new = np.einsum("ij,ij->i", r, r)
            p = r + (rs_new / np.maximum(rs, 1e-12))[:, None] * p
            rs = rs_new
        return x.astype(np.float32)

    def loss(self, conf: sp.csr_matrix) -> float:
        """Weighted squared error over all (user, item) pairs plus regularisation, per user."""
        x, y = self.user_factors, self.item_factors
        # sum over all pairs of pred^2 (confidence 1, preference 0) ...
        total = float(np.einsum("ij,ij->", x @ (y.T @ y), x))
        # ... corrected on observed pairs: c (1 - pred)^2 instead of pred^2
        pred = _values_at_nonzeros(conf, x, y)
        total += float(np.sum(conf.data * (1 - pred) ** 2 - pred ** 2))
        total += self.reg * float((x ** 2).sum() + (y ** 2).sum())
        return total / conf.shape[0]


def _values_at_nonzeros(m: sp.csr_matrix, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """x_u . y_i for every stored entry (u, i) of m, computed in dense row blocks (fast BLAS)."""
    out = np.empty(m.nnz, dtype=np.float32)
    n_rows, n_cols = m.shape
    block = max(1, _BLOCK_ELEMENTS // max(n_cols, 1))
    for a in range(0, n_rows, block):
        b = min(a + block, n_rows)
        lo, hi = m.indptr[a], m.indptr[b]
        if lo == hi:
            continue
        dense = x[a:b] @ y.T
        local_rows = np.repeat(np.arange(b - a), np.diff(m.indptr[a:b + 1]))
        out[lo:hi] = dense[local_rows, m.indices[lo:hi]]
    return out
