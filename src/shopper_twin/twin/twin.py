"""The shopper twin: when each shopper will shop each category x what they will pick there.

    P(shopper u buys product i in the window) = 1 - exp(-sum over window weeks of
        expected trips to i's category that week x P(i is viewed) x P(i is chosen | viewed))

Trips come from the trip-rate model, views and choices from the choice model, evaluated at the
store's planned prices and promotions for each week of the window (the store sets its own prices,
so they are known in advance). With `planned_prices=False` the twin knows nothing about the window:
every product keeps its last regular price and there are no promotions. That is the fair comparison
with the baselines, which never look at prices.
"""

from __future__ import annotations

import contextlib
import copy
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from ..eval.split import TrainData
from .choice import SHOPPER_TRAITS, ChoiceModel, fit_choice_model
from .incidence import FEATURES, TripRateModel, fit_trip_rates, snapshot
from .trips import Catalogue, Trips, build_trips


@contextlib.contextmanager
def torch_threads(n: int | None):
    """Small tensor ops run fastest on one or two threads and give the same numbers every run."""
    n = n or int(os.environ.get("SHOPPER_TWIN_THREADS", 1))
    old = torch.get_num_threads()
    torch.set_num_threads(n)
    try:
        yield
    finally:
        torch.set_num_threads(old)


def shopper_groups(shoppers: pd.DataFrame) -> np.ndarray:
    """Pooling groups from the public profile: age band x household size."""
    s = shoppers.sort_values("shopper_id")
    return pd.factorize(s.age_band.astype(str) + "|" + s.household_band.astype(str), sort=True)[0]


class ShopperTwin:
    """Implements the evaluation harness's recommender interface: fit(TrainData) then score(users)."""

    SETTINGS = ("dim", "epochs", "view_weight", "taste_reg", "trait_reg", "lr", "batch_size", "seed")
    # what the twin remembers at the cutoff
    SAVED_ARRAYS = ("trip_rates", "last_bought", "brand_n", "dept_n", "cat_n", "cat_brand_n", "views_per_trip",
                    "quantity", "purchase_scale")

    def __init__(self, world, dim: int = 16, epochs: int = 8, view_weight: float = 1.0, taste_reg: float = 1.0,
                 trait_reg: float = 2.0, lr: float = 0.03, batch_size: int = 2048, planned_prices: bool = True,
                 seed: int = 0, threads: int | None = None, verbose: bool = False):
        self.cat = Catalogue.from_world(world)
        self.groups = shopper_groups(world.shoppers)
        self.world_name = world.path.name
        self.dim, self.epochs, self.view_weight = dim, epochs, view_weight
        self.taste_reg, self.trait_reg, self.lr, self.batch_size = taste_reg, trait_reg, lr, batch_size
        self.planned_prices, self.seed, self.threads, self.verbose = planned_prices, seed, threads, verbose
        self.cutoff: int | None = None
        n_cat = self.cat.n_categories
        self.cat_department = pd.Series(self.cat.department).groupby(self.cat.category).first().reindex(
            range(n_cat)).to_numpy()

    @property
    def name(self) -> str:
        return "Shopper twin (knows price plan)" if self.planned_prices else "Shopper twin"

    @property
    def settings(self) -> dict:
        return {k: getattr(self, k) for k in self.SETTINGS}

    @property
    def cutoff_day(self) -> int:
        """First day the twin knows nothing about (it was trained on days before this)."""
        if self.cutoff is None:
            raise RuntimeError("the twin has not been fitted yet")
        return self.cutoff

    def with_prices(self, planned: bool) -> "ShopperTwin":
        """The same fitted twin, predicting with (True) or without (False) the store's planned prices."""
        twin = copy.copy(self)
        twin.planned_prices = planned
        return twin

    # ---- training
    def fit(self, data: TrainData, trips: Trips | None = None) -> "ShopperTwin":
        """Learn from `data` (events before its cutoff). `trips` can be passed to save rebuilding them: trips
        built from a longer event log are cut at the cutoff, which gives exactly the same table."""
        if len(data.events) and int(data.events.day.max()) >= data.cutoff_day:
            raise ValueError(f"training events must all be before day {data.cutoff_day}")
        with torch_threads(self.threads):
            t0 = time.perf_counter()
            if trips is None:
                trips = build_trips(data.events, self.cat)
            trips = trips.subset(trips.day < data.cutoff_day)
            self.n_shoppers, self.cutoff, self.horizon = data.n_users, data.cutoff_day, data.horizon
            self.n_trips = len(trips)
            self._log(f"  {len(trips):,} trips built ({time.perf_counter() - t0:.0f}s)")
            self.views_per_trip = _views_per_trip(trips, self.cat.n_categories)
            self.choice: ChoiceModel = fit_choice_model(
                trips, self.cat, self.groups, data.n_users, dim=self.dim, epochs=self.epochs,
                batch_size=self.batch_size, lr=self.lr, view_weight=self.view_weight, taste_reg=self.taste_reg,
                trait_reg=self.trait_reg, seed=self.seed, verbose=self.verbose)
            self._log(f"  choice model fitted ({time.perf_counter() - t0:.0f}s)")
            self.rate_model: TripRateModel = fit_trip_rates(
                trips, data.n_users, self.cat.n_categories, self.cat_department, data.cutoff_day, data.horizon,
                seed=self.seed, verbose=self.verbose)
            x, _ = snapshot(trips, data.n_users, self.cat.n_categories, self.cat_department, data.cutoff_day,
                            data.horizon, with_target=False)
            self.trip_rates = self.rate_model.rate(x, np.tile(np.arange(self.cat.n_categories), data.n_users)
                                                   ).reshape(data.n_users, self.cat.n_categories)
            self._log(f"  trip-rate model fitted ({time.perf_counter() - t0:.0f}s)")
            self._memory_at_cutoff(trips)
            self.purchase_scale = self._calibrate_purchases(trips, max(data.cutoff_day - data.horizon, 1))
            self._log(f"  purchases per trip calibrated ({time.perf_counter() - t0:.0f}s)")
            self.quantity = _mean_quantity(data.events, self.cat, data.n_users)
        return self

    # ---- saving
    def save(self, path: str | Path) -> Path:
        """Save everything needed to predict (not the trip table) to a .pt file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        # plain Python numbers only: torch.load(weights_only=True) refuses numpy scalars
        settings = {k: (float(v) if isinstance(v, (float, np.floating)) else int(v)) for k, v in self.settings.items()}
        blob = {
            "format": 1, "world": self.world_name, "n_products": int(self.cat.n_products), "settings": settings,
            "cutoff_day": int(self.cutoff_day), "horizon": int(self.horizon), "n_shoppers": int(self.n_shoppers),
            "choice": self.choice.state_dict(), "rate_model": self.rate_model.state_dict(),
            "arrays": {k: torch.as_tensor(getattr(self, k)) for k in self.SAVED_ARRAYS},
        }
        # write to a temporary name first, so an interrupted save never leaves a broken file behind
        tmp = path.with_name(path.name + ".tmp")
        torch.save(blob, tmp)
        os.replace(tmp, path)
        return path

    @classmethod
    def load(cls, path: str | Path, world, planned_prices: bool = True) -> "ShopperTwin":
        blob = torch.load(Path(path), weights_only=True)
        if blob["world"] != world.path.name or blob["n_products"] != world.n_products:
            raise ValueError(f"{path} was fitted on world '{blob['world']}', not '{world.path.name}'")
        twin = cls(world, planned_prices=planned_prices, **blob["settings"])
        twin.cutoff, twin.horizon, twin.n_shoppers = blob["cutoff_day"], blob["horizon"], blob["n_shoppers"]
        twin.choice = ChoiceModel(twin.n_shoppers, twin.cat.n_products, twin.cat.n_categories, twin.groups,
                                  dim=twin.dim)
        twin.choice.load_state_dict(blob["choice"])
        twin.choice.eval().requires_grad_(False)
        twin.rate_model = TripRateModel(len(FEATURES), twin.cat.n_categories)
        twin.rate_model.load_state_dict(blob["rate_model"])
        twin.rate_model.eval().requires_grad_(False)
        for k in cls.SAVED_ARRAYS:
            setattr(twin, k, blob["arrays"][k].numpy())
        return twin

    # ---- what the twin learned
    def shopper_traits(self) -> pd.DataFrame:
        """One row per shopper with the learned traits (see choice.SHOPPER_TRAITS)."""
        with torch.no_grad():
            tr = self.choice.traits(torch.arange(self.n_shoppers)).numpy()
        return pd.DataFrame(tr, columns=list(SHOPPER_TRAITS)).rename_axis("shopper_id")

    def _memory_at_cutoff(self, t: Trips) -> None:
        """What the shopper remembers at the cutoff: last product per category, and purchase counts by brand,
        department, category and (category, brand) for the brand-loyalty feature."""
        cat = self.cat
        n_u, n_c = self.n_shoppers, cat.n_categories
        self.last_bought = -np.ones((n_u, n_c), dtype=np.int64)
        b = t.bought
        chosen = t.viewed[np.flatnonzero(b), t.chosen[b]]
        last = pd.DataFrame({"u": t.shopper[b], "c": t.category[b], "p": chosen}).groupby(["u", "c"]).p.last()
        self.last_bought[last.index.get_level_values(0), last.index.get_level_values(1)] = last.to_numpy()

        def counts(key: np.ndarray) -> np.ndarray:
            out = np.zeros((n_u, int(key.max()) + 1), dtype=np.float32)
            np.add.at(out, (t.shopper[b], key[chosen]), 1)
            return out

        self.brand_n, self.dept_n = counts(cat.brand), counts(cat.department)
        self.cat_n, self.cat_brand_n = counts(cat.category), counts(cat.cat_brand)

    def brand_share(self, users: np.ndarray, products: np.ndarray, category: int) -> np.ndarray:
        """[len(users), len(products)] the loyalty feature at the cutoff (same formula as in trips.py):
        the share of purchases in the department's other categories that were each product's brand."""
        cat = self.cat
        own = self.brand_n[users][:, cat.brand[products]] - self.cat_brand_n[users][:, cat.cat_brand[products]]
        others = self.dept_n[users, cat.department[products[0]]] - self.cat_n[users, category]
        return own / (others[:, None] + 1.0)

    # ---- prediction
    def window_weeks(self, start: int | None = None, horizon: int | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Price weeks covering the window and the share of window days in each."""
        start = self.cutoff_day if start is None else start
        horizon = self.horizon if horizon is None else horizon
        days = np.arange(start, start + horizon)
        weeks = np.minimum(days // 7, self.cat.n_weeks - 1)
        uniq, counts = np.unique(weeks, return_counts=True)
        return uniq, counts / counts.sum()

    def score(self, users: np.ndarray) -> np.ndarray:
        return self.purchase_probability(np.asarray(users))

    def purchase_probability(self, users: np.ndarray, price: np.ndarray | None = None, promo: np.ndarray | None = None,
                             available: np.ndarray | None = None) -> np.ndarray:
        """[len(users), n_products] P(buys each product at least once in the window).

        `price`, `promo` and `available` ([products, weeks]) replace the store's plan, which is how
        what-if scenarios are scored (see shopper_twin.sim).
        """
        return -np.expm1(-self.expected_purchases(users, price, promo, available))  # 1 - exp(-x), exact for tiny x

    def expected_purchases(self, users: np.ndarray, price: np.ndarray | None = None, promo: np.ndarray | None = None,
                           available: np.ndarray | None = None) -> np.ndarray:
        """[len(users), n_products] expected number of times each product is bought in the window."""
        users = np.asarray(users)
        price, promo = self._prices(price, promo)
        out = np.zeros((len(users), self.cat.n_products), dtype=np.float64)
        for c in range(self.cat.n_categories):
            products, purchases, _ = self._expectations(users, c, price, promo, available, spend=False)
            out[:, products] = purchases
        return out

    def category_expectations(self, users: np.ndarray, category: int, price: np.ndarray | None = None,
                              promo: np.ndarray | None = None, available: np.ndarray | None = None
                              ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """One category's products [M], and for each shopper x product [len(users), M]: the expected number of
        purchases in the window and the expected spend on them (purchases x price that week), per unit bought."""
        price, promo = self._prices(price, promo)
        return self._expectations(np.asarray(users), category, price, promo, available, spend=True)

    def _expectations(self, users: np.ndarray, category: int, price: np.ndarray, promo: np.ndarray,
                      available: np.ndarray | None, spend: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        weeks, wfrac = self.window_weeks()
        mem, outside, per_week = self._trip_terms(users, category, price, promo, available, weeks)
        purchases = np.zeros((len(users), len(mem)))
        spent = np.zeros((len(users), len(mem))) if spend else None
        for (w, ok, v, incl, k), frac in zip(per_week, wfrac):
            if ok is None:
                continue  # everything in the category is out of stock this week
            cols = slice(None) if ok.all() else ok
            q = frac * choice_given_trip(v, outside, incl, k)
            purchases[:, cols] += q
            if spend:
                spent[:, cols] += q * price[mem[cols], w][None, :]
        rate = self.trip_rates[users, category][:, None] * self.purchase_scale[category]
        return mem, rate * purchases, (rate * spent if spend else None)

    def _trip_terms(self, users: np.ndarray, category: int, price: np.ndarray, promo: np.ndarray,
                    available: np.ndarray | None, weeks: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
        """What a trip to `category` looks like to each shopper in each of `weeks`.

        Returns the category's products [M], each shopper's appeal of buying nothing [len(users)], and per week
        (week, in-stock products (bool [M], None if none), appeal v and chance of being looked at incl of every
        shopper x in-stock product, number of products looked at k). Uses the twin's memory (last purchase,
        brand counts), so a copy with an earlier memory replays an earlier date.
        """
        cat, m = self.cat, self.choice
        mem = cat.members[category][cat.members[category] >= 0]
        with torch.no_grad(), torch_threads(self.threads):
            ut = torch.as_tensor(users)
            taste = m.taste(ut)
            traits = m.traits(ut).numpy().astype(np.float64)
            memt = torch.as_tensor(mem)
            taste_fit = (taste @ m.emb[memt].T).numpy().astype(np.float64)  # [B, M]
            bias, view_bias = m.bias[memt].numpy(), m.view_bias[memt].numpy()
        habit = (mem[None, :] == self.last_bought[users, category][:, None]).astype(np.float64)
        share = self.brand_share(users, mem, category).astype(np.float64)
        base_v = (bias[None, :] + taste_fit + traits[:, 1:2] * cat.is_store[mem][None, :]
                  + traits[:, 2:3] * share + traits[:, 3:4] * habit)
        base_s = view_bias[None, :] + float(m.view_taste) * taste_fit + float(m.view_habit) * habit
        outside = float(m.outside_cat[category]) + traits[:, 4]
        log_price = np.log(price[mem] / cat.median[category]).astype(np.float32)  # [M, W]
        per_week = []
        for w in weeks:
            ok = np.ones(len(mem), dtype=bool) if available is None else available[mem, w]
            if not ok.any():
                per_week.append((w, None, None, None, 0))
                continue
            k = int(min(self.views_per_trip[category], ok.sum()))
            pr = promo[mem[ok], w][None, :]
            s = base_s[:, ok] + float(m.view_promo) * pr
            v = base_v[:, ok] - traits[:, 0:1] * log_price[ok, w][None, :] + float(m.promo) * pr
            per_week.append((w, ok, v, inclusion_probabilities(s, k), k))
        return mem, outside, per_week

    def _calibrate_purchases(self, trips: Trips, start: int) -> np.ndarray:
        """[categories] factor that makes each category's predicted purchases per trip add up.

        The twin replays the days from `start` to its cutoff: starting from what it knew on day `start`, it
        predicts purchases for the trips that really happened then, at the prices of the time, and compares
        them with the purchases that really happened. The factor corrects the overall level only (every product
        in the category and every scenario is scaled the same way), so it never changes how strongly shoppers
        react to prices. Kept between 0.5 and 2: a larger gap means something the twin can't model.
        """
        past = copy.copy(self)
        past._memory_at_cutoff(trips.subset(trips.day < start))
        window = trips.subset(trips.day >= start)
        scale = np.ones(self.cat.n_categories)
        for c in range(self.cat.n_categories):
            sel = window.category == c
            if not sel.any():
                continue
            users, row = np.unique(window.shopper[sel], return_inverse=True)
            weeks, col = np.unique(window.week[sel], return_inverse=True)
            n_trips = np.zeros((len(users), len(weeks)))
            np.add.at(n_trips, (row, col), 1)
            _, outside, per_week = past._trip_terms(users, c, self.cat.price, self.cat.promo, None, weeks)
            predicted = sum(float(n_trips[:, j] @ choice_given_trip(v, outside, incl, k).sum(axis=1))
                            for j, (_, _, v, incl, k) in enumerate(per_week))
            scale[c] = np.clip((window.bought[sel].sum() + 1) / (predicted + 1), 0.5, 2.0)
        return scale

    def _prices(self, price: np.ndarray | None, promo: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """The prices to predict with: given ones, else the store's plan (or the frozen prices)."""
        if price is not None and promo is not None:
            return price, promo
        plan_price, plan_promo = (self.cat.price, self.cat.promo) if self.planned_prices else self.cat.frozen(
            (self.cutoff_day - 1) // 7)
        return (plan_price if price is None else price), (plan_promo if promo is None else promo)

    def _log(self, msg: str) -> None:
        if self.verbose:
            print(msg)


def _mean_quantity(events: pd.DataFrame, cat: Catalogue, n_shoppers: int) -> np.ndarray:
    """[shoppers, categories] average units per purchase (a shopper's own average, else the category's, else 1)."""
    out = np.ones((n_shoppers, cat.n_categories), dtype=np.float32)
    if "quantity" not in events.columns:
        return out
    buys = events[events.event_type == "purchase"]
    df = pd.DataFrame({"u": buys.shopper_id.to_numpy(), "c": cat.category[buys.product_id.to_numpy()],
                       "q": buys.quantity.to_numpy().astype(np.float32)})
    out *= df.groupby("c").q.mean().reindex(range(cat.n_categories)).fillna(1.0).to_numpy(np.float32)[None, :]
    own = df.groupby(["u", "c"]).q.mean()
    out[own.index.get_level_values(0), own.index.get_level_values(1)] = own.to_numpy()
    return out


def _views_per_trip(trips: Trips, n_categories: int) -> np.ndarray:
    """Typical number of products viewed per trip in each category (4 for everyday items, 8 when browsing)."""
    views = (trips.viewed >= 0).sum(axis=1)
    typical = pd.Series(views).groupby(trips.category).median().reindex(range(n_categories)).fillna(4)
    return np.maximum(np.round(typical.to_numpy()), 1).astype(np.int64)


def inclusion_probabilities(scores: np.ndarray, k: int, max_iterations: int = 100, tol: float = 1e-9
                            ) -> np.ndarray:
    """P(each product is among the k viewed) when k products are drawn one by one without replacement
    with probability proportional to exp(score). Uses the standard Poisson-sampling approximation:
    incl_i = 1 - exp(-t * p_i) with t chosen so the inclusion probabilities add up to k."""
    if k >= scores.shape[1]:
        return np.ones(scores.shape)  # every product gets looked at
    p = np.exp(scores - scores.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    t = np.full((p.shape[0], 1), float(k))
    for _ in range(max_iterations):  # Newton's method; a few steps are enough unless k is close to the count
        e = np.exp(-t * p)
        f = (1 - e).sum(axis=1, keepdims=True) - k
        if np.abs(f).max() < tol:
            break
        slope = (p * e).sum(axis=1, keepdims=True)
        t = t - f / np.maximum(slope, 1e-12)
    return np.clip(1 - np.exp(-t * p), 0, 1)


def choice_given_trip(v: np.ndarray, outside: np.ndarray, incl: np.ndarray, k: int) -> np.ndarray:
    """P(product i is viewed and then chosen on one trip): the logit choice among the viewed products and
    'buy nothing', averaged over which k - 1 other products are viewed alongside i.

    With S = the total appeal exp(v_j) of the other viewed products (random: it depends on which ones are
    viewed), P(i chosen | i viewed) = E[e_i / (e_i + e_0 + S)]. Plugging in the average of S would give too
    little (the curve bends upwards), so a second-order term uses the spread of S as well:
        E[f(S)] ~ f(mean) + f''(mean) Var(S) / 2 = e_i / (c + mean) x (1 + Var(S) / (c + mean)^2), c = e_i + e_0.
    Given that i is viewed, each other product j is viewed with probability p_j = incl_j (k - 1) / (k - incl_i),
    and Var(S) uses Hajek's approximation for a fixed number of draws: sum_j d_j (e_j - ebar)^2 with
    d_j = p_j (1 - p_j) and ebar the d-weighted mean. Checked against simulating the views: within 2% on the
    chance of buying and within 0.2 percentage points on the effect of a 20% price rise.
    """
    top = np.maximum(v.max(axis=1, keepdims=True), outside[:, None])
    ev = np.exp(v - top)
    e0 = np.exp(outside[:, None] - top)
    r = (k - 1) / np.maximum(k - incl, 1e-9)  # p_j = r_i * incl_j for the products viewed alongside i

    def total(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """sum over j != i of x_j * y_j, for every i."""
        return (x * y).sum(axis=1, keepdims=True) - x * y

    mean = r * total(incl, ev)
    # sums over j != i of d_j, d_j e_j and d_j e_j^2 with d_j = r_i incl_j - r_i^2 incl_j^2
    d0 = r * total(incl, 1.0) - r * r * total(incl * incl, 1.0)
    d1 = r * total(incl, ev) - r * r * total(incl * incl, ev)
    d2 = r * total(incl, ev * ev) - r * r * total(incl * incl, ev * ev)
    var = np.maximum(d2 - d1 * d1 / np.maximum(d0, 1e-12), 0.0)
    c = ev + e0
    p = incl * ev / (c + mean) * (1 + var / (c + mean) ** 2)
    # the correction can't push a shopper's total chance of buying past 1 (it never gets close in practice)
    return p / np.maximum(p.sum(axis=1, keepdims=True), 1.0)
