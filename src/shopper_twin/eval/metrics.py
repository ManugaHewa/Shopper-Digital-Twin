"""Ranking metrics for top-K recommendation lists.

All functions take `hits`, a [n_users, K] boolean array where hits[u, r] is True
when the item at rank r (0 = top) was actually bought, and `n_relevant`, the
number of products each user actually bought in the window.
"""

from __future__ import annotations

import numpy as np


def hit_rate(hits: np.ndarray) -> np.ndarray:
    """1 if at least one recommended product was bought."""
    return hits.any(axis=1).astype(float)


def precision(hits: np.ndarray) -> np.ndarray:
    """Share of the K recommendations that were bought."""
    return hits.mean(axis=1)


def recall(hits: np.ndarray, n_relevant: np.ndarray) -> np.ndarray:
    """Share of the products bought that were in the top K."""
    return hits.sum(axis=1) / np.maximum(n_relevant, 1)


def ndcg(hits: np.ndarray, n_relevant: np.ndarray) -> np.ndarray:
    """Normalised discounted cumulative gain: rewards hits near the top of the list.

    DCG = sum over ranks r of hit / log2(r + 2); divided by the best possible DCG,
    which puts min(K, n_relevant) hits at the top.
    """
    k = hits.shape[1]
    discounts = 1.0 / np.log2(np.arange(k) + 2)
    dcg = (hits * discounts).sum(axis=1)
    ideal = np.cumsum(discounts)[np.clip(n_relevant, 1, k) - 1]
    return dcg / ideal


def mrr(hits: np.ndarray) -> np.ndarray:
    """1 / rank of the first hit (0 if none)."""
    first = hits.argmax(axis=1)
    return np.where(hits.any(axis=1), 1.0 / (first + 1), 0.0)


def summarise(hits: np.ndarray, n_relevant: np.ndarray, k: int) -> dict[str, float]:
    """Mean metrics at cut-off k (hits may have more columns than k)."""
    h = hits[:, :k]
    return {
        f"hit_rate@{k}": float(hit_rate(h).mean()),
        f"precision@{k}": float(precision(h).mean()),
        f"recall@{k}": float(recall(h, n_relevant).mean()),
        f"ndcg@{k}": float(ndcg(h, n_relevant).mean()),
        f"mrr@{k}": float(mrr(h).mean()),
    }
