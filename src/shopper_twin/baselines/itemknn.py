"""Item-to-item collaborative filtering ("customers who bought this also bought")."""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from ..eval.harness import top_k_rows
from ..eval.split import TrainData


class ItemKNN:
    """Similar products are ones bought by the same shoppers (cosine similarity, shrunk for rare items).

    A shopper's score for a product = sum over what they bought (recency-weighted) of how
    similar it is to that product. Only each product's `neighbours` most similar products are kept.
    """

    name = "Item-to-item"

    def __init__(self, neighbours: int = 50, shrink: float = 10.0, half_life_days: float = 60.0):
        self.neighbours = neighbours
        self.shrink = shrink
        self.half_life_days = half_life_days

    def fit(self, data: TrainData) -> "ItemKNN":
        bought = (data.purchase_counts > 0).astype(np.float32)
        co = (bought.T @ bought).toarray()
        norms = np.sqrt(np.diag(co))
        sim = co / (np.outer(norms, norms) + self.shrink + 1e-9)
        np.fill_diagonal(sim, 0.0)
        k = min(self.neighbours, data.n_items - 1)
        # keep the top-k neighbours of each item (per row), drop the rest; ties go to the lower product id
        keep = np.vstack([top_k_rows(sim[a:a + 1000], k) for a in range(0, data.n_items, 1000)])
        rows = np.repeat(np.arange(data.n_items), k)
        vals = np.take_along_axis(sim, keep, axis=1).ravel()
        self.similarity = sp.csr_matrix((vals, (rows, keep.ravel())), shape=sim.shape, dtype=np.float32)

        p = data.purchases
        weight = np.exp(-(data.cutoff_day - p.day.to_numpy()) * np.log(2) / self.half_life_days)
        profile = sp.coo_matrix((weight, (p.shopper_id, p.product_id)), shape=(data.n_users, data.n_items)).tocsr()
        self.profile = profile
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return (self.profile[users] @ self.similarity).toarray()
