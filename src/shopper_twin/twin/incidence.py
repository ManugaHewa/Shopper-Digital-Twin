"""When will a shopper next look at each category? Expected number of trips per (shopper, category).

Staples come round on a cycle (milk every week), other categories get browsed now and then.
The model learns this from each shopper's own trip history with a small neural network that
predicts a count (Poisson regression). It is trained "time-travel" style: features are computed
as if it were an earlier date, and the target is what then happened in the following weeks, all
inside the training period, so nothing from the prediction window is ever used.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from torch import nn

from .trips import Trips

FEATURES = (
    "trips_7", "trips_28", "trips_56", "trips_per_28", "buys_28", "buys_per_28",
    "days_since_trip", "days_since_buy", "never_tripped", "never_bought",
    "gap_buy", "gap_trip", "due_buy", "due_trip", "buy_rate",
    "shopper_visits_28", "shopper_visits_per_28", "shopper_active_share", "dept_share", "dept_share_28",
    "cat_trip_rate_28", "cat_trip_rate_all", "cat_views", "cat_trend", "cat_buyer_share", "cat_median_gap",
)
# Every feature means the same thing however much history there is (rates instead of running totals,
# "days since" capped), because the model is trained at earlier dates with less history than it is used at.
LONG_AGO = 120.0  # "days since" is capped here, and "never" counts as this long ago



def snapshot(trips: Trips, n_shoppers: int, n_categories: int, cat_department: np.ndarray, cutoff: int,
             horizon: int, with_target: bool = True) -> tuple[np.ndarray, np.ndarray | None]:
    """Features for every (shopper, category) pair using trips before `cutoff`, and (with `with_target`) the
    number of trips in [cutoff, cutoff + horizon) as target, else None.

    Rows are shopper-major: row = shopper * n_categories + category.
    """
    before = trips.day < cutoff
    u, c, d = trips.shopper[before], trips.category[before], trips.day[before]
    b = trips.bought[before]
    n_pairs = n_shoppers * n_categories
    pair = u * n_categories + c
    age = cutoff - d  # days before the cutoff (1 = the day before)

    def count(mask):
        return np.bincount(pair[mask], minlength=n_pairs).astype(np.float32)

    f = {}
    f["trips_7"], f["trips_28"], f["trips_56"] = count(age <= 7), count(age <= 28), count(age <= 56)
    span = float(max(cutoff - int(trips.day.min()), 1))  # days of history
    trips_all, buys_all = count(np.ones_like(b)), count(b)
    f["trips_per_28"] = trips_all / span * 28
    f["buys_28"], f["buys_per_28"] = count(b & (age <= 28)), buys_all / span * 28
    last_trip = _last(pair, d, n_pairs)
    last_buy = _last(pair[b], d[b], n_pairs)
    f["days_since_trip"] = np.where(last_trip >= 0, np.minimum(cutoff - last_trip, LONG_AGO), LONG_AGO)
    f["days_since_buy"] = np.where(last_buy >= 0, np.minimum(cutoff - last_buy, LONG_AGO), LONG_AGO)
    f["never_tripped"] = (last_trip < 0).astype(np.float32)
    f["never_bought"] = (last_buy < 0).astype(np.float32)
    gap_buy = _mean_gap(pair[b], d[b], n_pairs)
    gap_trip = _mean_gap(pair, d, n_pairs)
    # a pair's own gaps, falling back to the category's typical gap when it has none
    cat_gap = pd.Series(gap_buy).groupby(np.tile(np.arange(n_categories), n_shoppers)).median()
    cat_gap = cat_gap.reindex(range(n_categories)).fillna(LONG_AGO).to_numpy()
    cat_of_pair = np.tile(np.arange(n_categories), n_shoppers)
    f["gap_buy"] = np.minimum(np.where(np.isnan(gap_buy), cat_gap[cat_of_pair], gap_buy), LONG_AGO)
    f["gap_trip"] = np.minimum(np.where(np.isnan(gap_trip), f["gap_buy"], gap_trip), LONG_AGO)
    f["due_buy"] = ((f["days_since_buy"] + horizon / 2) / np.maximum(f["gap_buy"], 1)).astype(np.float32)
    f["due_trip"] = ((f["days_since_trip"] + horizon / 2) / np.maximum(f["gap_trip"], 1)).astype(np.float32)
    f["buy_rate"] = ((buys_all + 0.5) / (trips_all + 1)).astype(np.float32)

    # shopper-level activity
    shopper_days = pd.DataFrame({"u": u, "d": d}).drop_duplicates()
    visits_all = np.bincount(shopper_days.u, minlength=n_shoppers).astype(np.float32)
    visits_28 = np.bincount(shopper_days.u[cutoff - shopper_days.d <= 28], minlength=n_shoppers).astype(np.float32)
    first = np.full(n_shoppers, cutoff, dtype=np.int64)
    np.minimum.at(first, u, d)
    active = (cutoff - first).astype(np.float32)
    f["shopper_visits_28"] = np.repeat(visits_28, n_categories)
    f["shopper_visits_per_28"] = np.repeat(visits_all / np.maximum(active, 1) * 28, n_categories)
    f["shopper_active_share"] = np.repeat(active / span, n_categories)
    # how much of the shopper's attention goes to this category's department
    dept = cat_department[c]
    n_dept = int(cat_department.max()) + 1
    for name, mask in (("dept_share", np.ones_like(b)), ("dept_share_28", age <= 28)):
        dcount = np.bincount(u[mask] * n_dept + dept[mask], minlength=n_shoppers * n_dept).reshape(n_shoppers, n_dept)
        share = (dcount + 0.1) / (dcount.sum(axis=1, keepdims=True) + 0.1 * n_dept)
        f[name] = share[:, cat_department].reshape(-1).astype(np.float32)

    # category-level context
    active_shoppers = max(int((visits_all > 0).sum()), 1)
    cat_28 = np.bincount(c[age <= 28], minlength=n_categories) / active_shoppers
    cat_prev = np.bincount(c[(age > 28) & (age <= 56)], minlength=n_categories) / active_shoppers
    cat_all = np.bincount(c, minlength=n_categories) / active_shoppers / max(span, 1) * 28
    views = np.bincount(c, weights=(trips.viewed[before] >= 0).sum(axis=1), minlength=n_categories)
    views = views / np.maximum(np.bincount(c, minlength=n_categories), 1)
    buyers = np.bincount(np.unique(pair[b]) % n_categories, minlength=n_categories) / active_shoppers
    for name, value in (("cat_trip_rate_28", cat_28), ("cat_trip_rate_all", cat_all), ("cat_views", views),
                        ("cat_trend", np.log((cat_28 + 1e-3) / (cat_prev + 1e-3))), ("cat_buyer_share", buyers),
                        ("cat_median_gap", cat_gap)):
        f[name] = np.tile(value.astype(np.float32), n_shoppers)

    x = np.stack([f[k] for k in FEATURES], axis=1).astype(np.float32)
    target = None
    if with_target:
        window = (trips.day >= cutoff) & (trips.day < cutoff + horizon)
        target = np.bincount(trips.shopper[window] * n_categories + trips.category[window],
                             minlength=n_pairs).astype(np.float32)
    return x, target


def _last(pair: np.ndarray, day: np.ndarray, n: int) -> np.ndarray:
    out = np.full(n, -1, dtype=np.int64)
    np.maximum.at(out, pair, day)
    return out


def _mean_gap(pair: np.ndarray, day: np.ndarray, n: int) -> np.ndarray:
    """Average days between distinct trip days of each pair (NaN with fewer than two)."""
    df = pd.DataFrame({"p": pair, "d": day}).drop_duplicates()
    g = df.groupby("p")["d"].agg(["min", "max", "size"])
    out = np.full(n, np.nan)
    ok = g["size"].to_numpy() > 1
    out[g.index.to_numpy()[ok]] = ((g["max"] - g["min"]) / (g["size"] - 1)).to_numpy()[ok]
    return out


def _transform(x: np.ndarray) -> np.ndarray:
    """Counts and day spans on a log scale; everything else as is."""
    out = x.copy()
    log_cols = [i for i, k in enumerate(FEATURES)
                if not k.startswith(("never", "cat_trend", "buy_rate", "shopper_active_share"))]
    out[:, log_cols] = np.log1p(np.maximum(out[:, log_cols], 0))
    return out


class TripRateModel(nn.Module):
    """Small MLP: features -> log of the expected number of trips in the window, times a per-category
    correction that makes the training totals of each category come out right."""

    def __init__(self, n_features: int, n_categories: int, hidden: int = 64, seed: int = 0):
        super().__init__()
        torch.manual_seed(seed)
        self.net = nn.Sequential(nn.Linear(n_features, hidden), nn.ReLU(), nn.Linear(hidden, hidden), nn.ReLU(),
                                 nn.Linear(hidden, 1))
        self.register_buffer("mean", torch.zeros(n_features))
        self.register_buffer("scale", torch.ones(n_features))
        self.register_buffer("log_correction", torch.zeros(n_categories))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net((x - self.mean) / self.scale).squeeze(-1)

    def rate(self, x: np.ndarray, category: np.ndarray, batch: int = 200_000) -> np.ndarray:
        """Expected trips for feature rows `x` (from `snapshot`) of the given categories."""
        return self._raw_rate(torch.as_tensor(_transform(x)), batch) * self.log_correction.exp().numpy()[category]

    def _raw_rate(self, xt: torch.Tensor, batch: int = 200_000) -> np.ndarray:
        with torch.no_grad():
            return np.concatenate([self(xt[s:s + batch]).exp().numpy() for s in range(0, len(xt), batch)])


def fit_trip_rates(trips: Trips, n_shoppers: int, n_categories: int, cat_department: np.ndarray, cutoff: int,
                   horizon: int, n_snapshots: int = 3, epochs: int = 4, seed: int = 0,
                   verbose: bool = False) -> TripRateModel:
    """Train on snapshots at cutoff - horizon, cutoff - 2 * horizon, ... (each needs 4 weeks of history)."""
    xs, ys = [], []
    for k in range(1, n_snapshots + 1):
        t = cutoff - k * horizon
        if t - int(trips.day.min()) < 28:
            break
        sub = trips.subset(trips.day < t + horizon)
        x, y = snapshot(sub, n_shoppers, n_categories, cat_department, t, horizon)
        xs.append(x)
        ys.append(y)
    if not xs:
        raise ValueError(f"need at least {horizon + 28} days of history before day {cutoff} to learn trip rates")
    x = torch.as_tensor(_transform(np.concatenate(xs)))
    y = torch.as_tensor(np.concatenate(ys))
    model = TripRateModel(x.shape[1], n_categories, seed=seed)
    model.mean.copy_(x.mean(0))
    model.scale.copy_(x.std(0).clamp(min=0.05))  # a (nearly) constant feature must not blow up later
    opt = torch.optim.Adam(model.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(seed)
    loss_fn = nn.PoissonNLLLoss(log_input=True, full=False)
    for epoch in range(epochs):
        total = 0.0
        for idx in torch.randperm(len(x), generator=g).split(8192):
            loss = loss_fn(model(x[idx]), y[idx])
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        if verbose:
            print(f"    trip-rate epoch {epoch + 1}/{epochs}: Poisson loss {total / len(x):.4f}")
    # Stochastic training leaves each category's overall level a few % off (it moves with the random seed).
    # Rescale so every category's predicted trips on the training snapshots add up to the real number.
    model.eval()
    category = np.tile(np.arange(n_categories), n_shoppers * len(xs))
    observed = np.bincount(category, weights=y.numpy(), minlength=n_categories)
    predicted = np.bincount(category, weights=model._raw_rate(x), minlength=n_categories)
    model.log_correction.copy_(torch.as_tensor(np.log((observed + 1) / (predicted + 1)), dtype=torch.float32))
    return model.requires_grad_(False)

