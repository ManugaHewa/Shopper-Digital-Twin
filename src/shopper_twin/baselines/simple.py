"""Simple, strong heuristics that every serious recommender has to beat."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.sparse import coo_matrix, csr_matrix

from ..eval.split import TrainData


class Random:
    """Sanity floor: random products."""

    name = "Random"

    def __init__(self, seed: int = 0):
        self.seed = seed

    def fit(self, data: TrainData) -> "Random":
        self.n_items = data.n_items
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        # seeded per user so the same user always gets the same list
        return np.stack([np.random.default_rng([self.seed, int(u)]).random(self.n_items) for u in users])


class Popularity:
    """Everyone gets the store's best sellers. With `window_days`, only recent sales count."""

    def __init__(self, window_days: int | None = None):
        self.window_days = window_days
        self.name = "Popularity" if window_days is None else f"Popularity (last {window_days} days)"

    def fit(self, data: TrainData) -> "Popularity":
        p = data.purchases
        if self.window_days is not None:
            p = p[p.day >= data.cutoff_day - self.window_days]
        # count buyers, not units, so one bulk buyer can't dominate
        buyers = p.drop_duplicates(["shopper_id", "product_id"]).product_id
        self.counts = np.bincount(buyers, minlength=data.n_items).astype(np.float64)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return np.broadcast_to(self.counts, (len(users), len(self.counts)))


class BuyAgain:
    """"Buy it again": the shopper's own past purchases, weighted by how often and how recently.

    score(u, i) = sum over u's purchases of i of exp(-(days ago) / half_life * ln 2),
    plus a tiny popularity term so unbought products are still ranked sensibly.
    """

    name = "Buy again"

    def __init__(self, half_life_days: float = 30.0):
        self.half_life_days = half_life_days

    def fit(self, data: TrainData) -> "BuyAgain":
        p = data.purchases
        weight = np.exp(-(data.cutoff_day - p.day.to_numpy()) * np.log(2) / self.half_life_days)
        self.matrix = coo_matrix((weight, (p.shopper_id, p.product_id)),
                                 shape=(data.n_users, data.n_items)).tocsr()
        self.fallback = Popularity().fit(data).counts
        self.fallback = 1e-6 * self.fallback / max(self.fallback.max(), 1)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.matrix[users].toarray() + self.fallback


class RepurchaseCycle:
    """"Buy it again", timed: is the shopper about to run out, and which product will they pick?

    Staples are regular at the category level (milk every week) even when the exact product
    changes, so this works in two parts:
      1. Category due-ness: the shopper's typical gap between purchases in the category
         (their own gaps, shrunk towards the category-wide typical gap when they have few),
         and how far through that gap they will be by the end of the prediction window.
      2. Product share: which products they pick within the category, recency-weighted.
    Score = due-ness x habit strength x product share.
    """

    name = "Repurchase cycle"

    def __init__(self, half_life_days: float = 30.0, prior_strength: float = 2.0, one_off_weight: float = 0.1):
        if prior_strength <= 0:
            raise ValueError("prior_strength must be positive")
        self.half_life_days = half_life_days
        self.prior_strength = prior_strength  # how many "virtual" gaps the category-wide prior is worth
        self.one_off_weight = one_off_weight  # habit score for a category bought only once

    def fit(self, data: TrainData) -> "RepurchaseCycle":
        p = data.purchases[["shopper_id", "product_id", "day"]].copy()
        p["category"] = data.item_category[p.product_id.to_numpy()]

        # 1. category due-ness, from distinct purchase days per (shopper, category)
        days = p.drop_duplicates(["shopper_id", "category", "day"])
        g = days.groupby(["shopper_id", "category"])["day"].agg(["min", "max", "size"]).reset_index()
        n_gaps = (g["size"] - 1).to_numpy()
        has_gap = n_gaps > 0
        own_gap = np.where(has_gap, (g["max"] - g["min"]).to_numpy() / np.maximum(n_gaps, 1), 0.0)
        prior = np.full(int(data.item_category.max()) + 1, 90.0)  # no repeat buyers at all: assume rare
        typical = pd.Series(own_gap[has_gap]).groupby(g["category"].to_numpy()[has_gap]).median()
        prior[typical.index.to_numpy()] = typical.to_numpy()
        category_prior = prior[g["category"].to_numpy()]
        gap = (own_gap * n_gaps + category_prior * self.prior_strength) / (n_gaps + self.prior_strength)
        days_since = data.cutoff_day - g["max"].to_numpy()
        due = np.minimum((days_since + data.horizon) / np.maximum(gap, 1.0), 1.0)
        habit = n_gaps / (n_gaps + 1.0) + self.one_off_weight
        g["category_score"] = due * habit

        # 2. product share within the category, recency-weighted
        p["w"] = np.exp(-(data.cutoff_day - p["day"].to_numpy()) * np.log(2) / self.half_life_days)
        s = p.groupby(["shopper_id", "category", "product_id"])["w"].sum().reset_index()
        s["share"] = s["w"] / s.groupby(["shopper_id", "category"])["w"].transform("sum")
        s = s.merge(g[["shopper_id", "category", "category_score"]], on=["shopper_id", "category"])

        self.matrix = csr_matrix((s["share"] * s["category_score"], (s["shopper_id"], s["product_id"])),
                                 shape=(data.n_users, data.n_items))
        pop = Popularity().fit(data).counts
        self.fallback = 1e-6 * pop / max(pop.max(), 1)
        return self

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.matrix[users].toarray() + self.fallback
