"""Time-based train/test splits.

The question every model answers: "given everything up to day T, which products
will this shopper buy in the next H days?" Training data stops at T, so no model
can peek at the future.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import cached_property

import numpy as np
import pandas as pd
import scipy.sparse as sp

# how much each kind of event counts as a signal of interest, for models that use them all
EVENT_WEIGHTS = {"view": 0.05, "cart": 0.3, "purchase": 1.0}


@dataclass
class TrainData:
    """Everything a model may learn from: events strictly before `cutoff_day`."""

    n_users: int
    n_items: int
    cutoff_day: int  # first day of the prediction window
    horizon: int  # length of the prediction window in days
    events: pd.DataFrame  # shopper_id, product_id, event_type, day (+ any extra columns loaded)
    item_category: np.ndarray  # category id of each product

    @cached_property
    def purchases(self) -> pd.DataFrame:
        return self.events[self.events.event_type == "purchase"]

    @cached_property
    def purchase_counts(self) -> sp.csr_matrix:
        """[users, items] number of purchase events."""
        p = self.purchases
        return _csr(p.shopper_id, p.product_id, np.ones(len(p), np.float32), (self.n_users, self.n_items))

    def interaction_matrix(self, weights: dict[str, float] | None = None) -> sp.csr_matrix:
        """[users, items] summed event weights (views, carts and purchases)."""
        weights = weights or EVENT_WEIGHTS
        w = np.zeros(len(self.events), dtype=np.float32)
        for name, value in weights.items():
            w[(self.events.event_type == name).to_numpy()] = value
        keep = w > 0
        return _csr(self.events.shopper_id.to_numpy()[keep], self.events.product_id.to_numpy()[keep], w[keep],
                    (self.n_users, self.n_items))


@dataclass
class EvalTask:
    """Who we score and what they actually bought in the prediction window."""

    users: np.ndarray  # shoppers with purchases both before and inside the window
    truth: sp.csr_matrix  # [len(users), items] 1 where the product was bought in the window
    truth_new: sp.csr_matrix  # same, but only products the shopper had never bought before
    seen: sp.csr_matrix  # [len(users), items] products bought before the window (to exclude in "new" mode)


@dataclass
class Split:
    train: TrainData
    task: EvalTask


def _csr(rows, cols, vals, shape, sum_duplicates: bool = True) -> sp.csr_matrix:
    m = sp.coo_matrix((vals, (np.asarray(rows), np.asarray(cols))), shape=shape).tocsr()
    if sum_duplicates:
        m.sum_duplicates()
    return m


def make_split(events: pd.DataFrame, n_users: int, n_items: int, item_category: np.ndarray,
               cutoff_day: int, horizon: int = 28) -> Split:
    """Train on days [0, cutoff_day), evaluate on purchases in [cutoff_day, cutoff_day + horizon)."""
    if cutoff_day < 1:
        raise ValueError(f"cutoff_day must leave some training history (got {cutoff_day})")
    day = events.day.to_numpy()
    before = events.loc[day < cutoff_day].reset_index(drop=True)
    window = events.loc[(day >= cutoff_day) & (day < cutoff_day + horizon)
                        & (events.event_type == "purchase").to_numpy(), ["shopper_id", "product_id"]]
    train = TrainData(n_users=n_users, n_items=n_items, cutoff_day=cutoff_day, horizon=horizon,
                      events=before, item_category=item_category)

    bought_before = train.purchase_counts > 0
    has_history = np.asarray(bought_before.sum(axis=1)).ravel() > 0
    in_window = np.zeros(n_users, dtype=bool)
    in_window[window.shopper_id.unique()] = True
    users = np.flatnonzero(has_history & in_window)

    truth_full = _csr(window.shopper_id, window.product_id, np.ones(len(window), np.float32), (n_users, n_items))
    truth_full.data[:] = 1.0
    truth = truth_full[users]
    seen = bought_before[users].astype(np.float32).tocsr()
    truth_new = (truth - truth.multiply(seen)).tocsr()
    truth_new.eliminate_zeros()
    return Split(train=train, task=EvalTask(users=users, truth=truth, truth_new=truth_new, seen=seen))
