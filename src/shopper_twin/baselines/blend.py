"""The usual industry recipe: "buy it again" for habits, collaborative filtering for everything else."""

from __future__ import annotations

import numpy as np

from ..eval.split import TrainData
from .als import ALS
from .simple import RepurchaseCycle


def percentile_ranks(scores: np.ndarray) -> np.ndarray:
    """Per row, each item's rank as a fraction in [0, 1] (1 = best; ties keep product-id order)."""
    n = scores.shape[1]
    order = np.argsort(scores, axis=1, kind="stable")
    ranks = np.empty(scores.shape, dtype=np.float64)
    np.put_along_axis(ranks, order, np.broadcast_to(np.arange(n) / max(n - 1, 1), scores.shape), axis=1)
    return ranks


class Blend:
    """Mix of the repurchase-cycle model and ALS by rank: (1 - weight) * rank_cycle + weight * rank_als.

    weight = 0 is the cycle model alone, weight = 1 is ALS alone.
    """

    name = "Blend (cycle + ALS)"

    def __init__(self, weight: float = 0.3, cycle: dict | None = None, als: dict | None = None):
        if not 0 <= weight <= 1:
            raise ValueError("weight must be between 0 and 1")
        self.weight = weight
        self.cycle_params = cycle or {}
        self.als_params = als or {}

    def fit(self, data: TrainData) -> "Blend":
        self.cycle = RepurchaseCycle(**self.cycle_params).fit(data)
        self.als = ALS(**self.als_params).fit(data)
        return self

    @classmethod
    def from_fitted(cls, cycle: RepurchaseCycle, als: ALS, weight: float) -> "Blend":
        """Reuse already-fitted components (e.g. when only the weight is being tuned)."""
        blend = cls(weight=weight)
        blend.cycle, blend.als = cycle, als
        return blend

    def score(self, users: np.ndarray) -> np.ndarray:
        return ((1 - self.weight) * percentile_ranks(self.cycle.score(users))
                + self.weight * percentile_ranks(self.als.score(users)))
