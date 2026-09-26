"""Did the twin learn the right things about each shopper? Checks against the answer key.

  Traits       each shopper's learned price sensitivity, store-brand liking, brand loyalty and habit vs the true
               values the store was simulated with (rank correlation: 1 = same order, 0 = no relation).
  Taste        which products in a category each shopper likes most, learned vs true.
  Calibration  when the twin says "30% chance", does the shopper buy about 30% of the time?

Only evaluation code may use the answer key, and nothing here feeds back into the twin.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch
from scipy import stats

# learned trait, true trait, label, whether the levels can be compared. Price sensitivity and habit enter the true
# store and the twin the same way. Store-brand liking's average is shared with the store-brand products' own appeal
# (only the differences between shoppers are learned), and the twin's loyalty uses brand share instead of a
# favourite brand, so only their order is compared.
TRAIT_PAIRS = (
    ("price_sens", "price_sensitivity", "price sensitivity", True),
    ("store", "store_brand_affinity", "store-brand liking", False),
    ("loyalty", "brand_loyalty", "brand loyalty", False),
    ("habit", "habit_strength", "habit", True),
)

# edges of the reliability bins (predicted chance of buying a product in the window)
CALIBRATION_EDGES = np.array([0, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.4, 0.6, 1.0])


def rank_correlation(x: np.ndarray, y: np.ndarray) -> dict:
    """Spearman rank correlation with a 90% range (Fisher transform)."""
    r = float(stats.spearmanr(x, y).statistic)
    half = 1.645 * 1.03 / np.sqrt(max(len(x) - 3, 1))
    z = np.arctanh(np.clip(r, -0.999999, 0.999999))
    return {"r": r, "range": [float(np.tanh(z - half)), float(np.tanh(z + half))], "n": int(len(x))}


def trait_recovery(twin, truth: pd.DataFrame, history: np.ndarray) -> dict:
    """Learned vs true traits for every shopper. `truth` is shoppers_truth (one row per shopper, in id order);
    `history` is each shopper's number of purchases before the cutoff. Also shows how much the public profile
    alone (the twin's group averages) gets right, and how the match improves with more history."""
    learned = twin.shopper_traits()
    with torch.no_grad():
        group_only = pd.DataFrame(twin.choice.trait_group[twin.choice.group].numpy(), columns=learned.columns)
    thirds = pd.qcut(pd.Series(history).rank(method="first"), 3, labels=["least", "middle", "most"]).to_numpy()
    out = {}
    for mine, true_col, label, same_scale in TRAIT_PAIRS:
        x, y = learned[mine].to_numpy(), truth[true_col].to_numpy()
        by_history = {}
        for third in ("least", "middle", "most"):
            m = thirds == third
            by_history[third] = {"purchases_median": float(np.median(history[m])),
                                 **rank_correlation(x[m], y[m])}
        out[label] = {
            "learned": mine, "true": true_col, "same_scale": same_scale,
            "twin": rank_correlation(x, y),
            "group_average_only": rank_correlation(group_only[mine].to_numpy(), y),
            "by_history": by_history,
            "median": {"learned": float(np.median(x)), "true": float(np.median(y))},
        }
    return out


def taste_recovery(twin, true_taste: np.ndarray, style: np.ndarray, quality: np.ndarray, quality_weight: np.ndarray,
                   taste_scale: float, n_shoppers: int = 1000, seed: int = 0) -> dict:
    """For a sample of shoppers and every category, how well the twin orders the category's products by appeal
    (everything but price, brand and habit), compared with the truth:

        true appeal    taste_scale x (taste . style) + quality_weight x 2 (quality - 0.5)
        twin appeal    product bias + taste . embedding
        popularity     product bias alone (the same order for everyone)

    Returns the average within-category rank correlation of each with the true appeal."""
    rng = np.random.default_rng(seed)
    users = np.sort(rng.choice(len(true_taste), size=min(n_shoppers, len(true_taste)), replace=False))
    cat, m = twin.cat, twin.choice
    with torch.no_grad():
        taste = m.taste(torch.as_tensor(users)).numpy()
        emb, bias = m.emb.numpy(), m.bias.numpy()
    sums = {"twin": 0.0, "popularity": 0.0}
    n = 0
    for c in range(cat.n_categories):
        mem = cat.members[c][cat.members[c] >= 0]
        if len(mem) < 5:
            continue
        truth = (taste_scale * true_taste[users] @ style[mem].T
                 + quality_weight[users, None] * 2 * (quality[mem][None, :] - 0.5))
        twin_appeal = bias[mem][None, :] + taste @ emb[mem].T
        popularity = np.broadcast_to(bias[mem][None, :], truth.shape)
        for key, pred in (("twin", twin_appeal), ("popularity", popularity)):
            sums[key] += _row_rank_correlation(pred, truth).sum()
        n += len(users)
    return {"shoppers": int(len(users)), "twin": sums["twin"] / n, "popularity": sums["popularity"] / n}


def _row_rank_correlation(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Spearman correlation of each row of a with the same row of b (no ties expected)."""
    ra = a.argsort(axis=1).argsort(axis=1).astype(np.float64)
    rb = b.argsort(axis=1).argsort(axis=1).astype(np.float64)
    ra -= ra.mean(axis=1, keepdims=True)
    rb -= rb.mean(axis=1, keepdims=True)
    return (ra * rb).sum(axis=1) / np.sqrt((ra * ra).sum(axis=1) * (rb * rb).sum(axis=1))


def calibration(twin, purchases: pd.DataFrame, n_shoppers: int, n_products: int, cutoff: int, horizon: int,
                batch_size: int = 1000) -> dict:
    """Chance of buying each product in the window (cutoff .. cutoff+horizon-1), predicted for every shopper x
    product, against what happened. Two simple forecasts to compare with, both learned on the previous window:

        popularity     the share of shoppers who bought the product in the last `horizon` days
        own history    the same, but separately for shoppers who had bought the product before and those who
                       had not (the strongest "no model" forecast)

    Brier score = average squared error (lower is better); log loss punishes confident mistakes harder."""
    day = purchases.day.to_numpy()
    shape = (n_shoppers, n_products)

    def bought(mask: np.ndarray) -> sp.csr_matrix:
        m = sp.csr_matrix((np.ones(mask.sum(), np.float32), (purchases.shopper_id.to_numpy()[mask],
                                                            purchases.product_id.to_numpy()[mask])), shape=shape)
        m.data[:] = 1.0
        return m

    window = bought((day >= cutoff) & (day < cutoff + horizon))
    last = bought((day >= cutoff - horizon) & (day < cutoff))
    before_last = bought(day < cutoff - horizon)
    history = bought(day < cutoff)
    popularity = np.asarray(last.sum(axis=0)).ravel() / n_shoppers
    # rates in the last window among shoppers who had / had not bought the product before it (lightly smoothed)
    had = np.asarray(before_last.sum(axis=0)).ravel()
    had_and_bought = np.asarray(last.multiply(before_last).sum(axis=0)).ravel()
    new_bought = np.asarray(last.sum(axis=0)).ravel() - had_and_bought
    repeat_rate = (had_and_bought + 10 * popularity) / (had + 10)
    new_rate = (new_bought + 10 * popularity) / (n_shoppers - had + 10)

    methods = ("twin", "popularity", "own history")
    k = len(CALIBRATION_EDGES) - 1
    acc = {name: {"brier": 0.0, "logloss": 0.0, "n": np.zeros(k), "pred": np.zeros(k), "hit": np.zeros(k)}
           for name in methods}
    total_pred = dict.fromkeys(methods, 0.0)
    for s in range(0, n_shoppers, batch_size):
        users = np.arange(s, min(s + batch_size, n_shoppers))
        y = window[users].toarray()
        seen = history[users].toarray() > 0
        preds = {"twin": twin.purchase_probability(users),
                 "popularity": np.broadcast_to(popularity[None, :], y.shape),
                 "own history": np.where(seen, repeat_rate[None, :], new_rate[None, :])}
        for name, p in preds.items():
            p = np.clip(p, 1e-6, 1 - 1e-6)
            a = acc[name]
            a["brier"] += float(((p - y) ** 2).sum())
            a["logloss"] += float(-(y * np.log(p) + (1 - y) * np.log1p(-p)).sum())
            b = np.clip(np.searchsorted(CALIBRATION_EDGES, p, side="right") - 1, 0, k - 1).ravel()
            a["n"] += np.bincount(b, minlength=k)
            a["pred"] += np.bincount(b, weights=p.ravel(), minlength=k)
            a["hit"] += np.bincount(b, weights=y.ravel(), minlength=k)
            total_pred[name] += float(p.sum())
    n_pairs = n_shoppers * n_products
    out = {"pairs": n_pairs, "pairs_bought": int(window.nnz), "methods": {}}
    for name in methods:
        a = acc[name]
        used = a["n"] > 0
        out["methods"][name] = {
            "brier": a["brier"] / n_pairs,
            "logloss": a["logloss"] / n_pairs,
            "pairs_bought_predicted": total_pred[name],
            "bins": [{"from": float(CALIBRATION_EDGES[i]), "to": float(CALIBRATION_EDGES[i + 1]),
                      "pairs": int(a["n"][i]), "predicted": float(a["pred"][i] / a["n"][i]),
                      "actual": float(a["hit"][i] / a["n"][i])} for i in np.flatnonzero(used)],
        }
    return out
