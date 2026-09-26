"""Score any recommender the same way: fit on the past, rank every product, compare with what was bought."""

from __future__ import annotations

import itertools
import time
from typing import Callable, Protocol

import numpy as np
import scipy.sparse as sp

from .metrics import summarise
from .split import Split, TrainData

MODES = ("all", "new")  # all purchases, or only products the shopper had never bought before


class Recommender(Protocol):
    name: str

    def fit(self, data: TrainData) -> "Recommender": ...

    def score(self, users: np.ndarray) -> np.ndarray:
        """[len(users), n_items] scores, higher = more likely to be bought."""
        ...


def top_k(model: Recommender, users: np.ndarray, k: int, exclude: sp.csr_matrix | None = None,
          batch_size: int = 2048) -> np.ndarray:
    """[len(users), k] product ids, best first; -1 where fewer than k products are allowed.

    `exclude` rows line up with `users`. Ties are broken by lower product id, so results are
    identical on every machine.
    """
    out = np.empty((len(users), k), dtype=np.int64)
    for s in range(0, len(users), batch_size):
        scores = np.array(model.score(users[s:s + batch_size]), dtype=np.float64, copy=True)
        if np.isnan(scores).any():
            raise ValueError(f"{model.name} produced NaN scores")
        if exclude is not None:
            ex = exclude[s:s + batch_size].tocoo()
            scores[ex.row, ex.col] = -np.inf
        out[s:s + batch_size] = top_k_rows(scores, k)
    return out


def top_k_rows(scores: np.ndarray, k: int) -> np.ndarray:
    """Column ids of the k highest values in each row, best first; ties go to the lower id."""
    n = scores.shape[0]
    if k > scores.shape[1]:
        raise ValueError(f"cannot pick {k} products from {scores.shape[1]}")
    # the k-th best score in each row; argpartition's own tie order varies by CPU, so it is not used directly
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    kth = np.take_along_axis(scores, part, axis=1).min(axis=1, keepdims=True)
    above = scores > kth
    tied = scores == kth
    need = k - above.sum(axis=1, keepdims=True)
    chosen = above | (tied & (np.cumsum(tied, axis=1) <= need))  # lowest product ids win ties
    idx = np.nonzero(chosen)[1].reshape(n, k)
    vals = np.take_along_axis(scores, idx, axis=1)
    order = np.argsort(-vals, axis=1, kind="stable")
    ranked = np.take_along_axis(idx, order, axis=1)
    ranked[np.take_along_axis(vals, order, axis=1) == -np.inf] = -1
    return ranked


def evaluate(model: Recommender, split: Split, mode: str = "all", ks: tuple[int, ...] = (10, 20),
             batch_size: int = 2048) -> dict:
    """Metrics for an already-fitted model on the split's prediction window."""
    task = split.task
    if max(ks) > split.train.n_items:
        raise ValueError(f"cannot recommend {max(ks)} products from a catalogue of {split.train.n_items}")
    if mode == "all":
        users, truth, exclude = task.users, task.truth, None
    elif mode == "new":
        keep = task.truth_new.getnnz(axis=1) > 0
        users, truth, exclude = task.users[keep], task.truth_new[keep], task.seen[keep]
    else:
        raise ValueError(f"mode must be one of {MODES}")

    k_max = max(ks)
    t0 = time.perf_counter()
    recs = top_k(model, users, k_max, exclude, batch_size)
    seconds = time.perf_counter() - t0

    hits = np.zeros(recs.shape, dtype=bool)
    valid = recs >= 0
    for s in range(0, len(users), batch_size):
        dense = truth[s:s + batch_size].toarray() > 0
        hits[s:s + batch_size] = np.take_along_axis(dense, np.maximum(recs[s:s + batch_size], 0), axis=1)
    hits &= valid
    n_relevant = truth.getnnz(axis=1)

    result = {"model": model.name, "mode": mode, "users": int(len(users)),
              "ms_per_user": round(1000 * seconds / max(len(users), 1), 4)}
    for k in ks:
        result.update(summarise(hits, n_relevant, k))
        shown = recs[:, :k][valid[:, :k]]
        result[f"coverage@{k}"] = float(len(np.unique(shown)) / split.train.n_items)
    return result


def fit_and_evaluate(make_model: Callable[[], Recommender], split: Split, modes=MODES, ks=(10, 20)) -> list[dict]:
    model = make_model()
    t0 = time.perf_counter()
    model.fit(split.train)
    fit_seconds = time.perf_counter() - t0
    return [{**evaluate(model, split, mode, ks), "fit_seconds": round(fit_seconds, 2)} for mode in modes]


def grid(**options) -> list[dict]:
    """All combinations: grid(a=[1, 2], b=[3]) -> [{'a': 1, 'b': 3}, {'a': 2, 'b': 3}]."""
    keys = list(options)
    return [dict(zip(keys, values)) for values in itertools.product(*options.values())]


def tune(model_class, params_grid: list[dict], val_split: Split, modes=MODES,
         metric: str = "ndcg@10") -> tuple[dict[str, dict], list[dict]]:
    """Fit each parameter set once on the validation split; return the best parameters per mode."""
    k = int(metric.rsplit("@", 1)[1])
    ks = tuple(sorted({k} | {x for x in (10, 20) if x <= val_split.train.n_items}))
    trials = []
    for params in params_grid:
        model = model_class(**params).fit(val_split.train)
        for mode in modes:
            trials.append({"params": params, **evaluate(model, val_split, mode, ks)})
    best = {}
    for mode in modes:
        scored = [t for t in trials if t["mode"] == mode and np.isfinite(t[metric])]
        if not scored:
            raise ValueError(f"no validation shoppers to tune on in mode '{mode}'")
        best[mode] = max(scored, key=lambda t: t[metric])["params"]
    return best, trials
