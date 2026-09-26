"""Ask the twins what would happen under a scenario, compared with the store's own plan.

For every shopper and every product in the categories a scenario touches, the twin gives the expected number of
purchases in the window (expected shopping trips to the category x chance of picking that product on a trip).
Multiplying by the shopper's usual quantity gives units, and by the price that week gives revenue. The twin has no
link between categories (a dearer coffee doesn't change how much milk anyone buys), so only the touched categories
are recomputed.

The twin's answer is an expected value, so running it twice gives the same numbers. The 90% bands come from
resampling shoppers (a bootstrap): they show how much the result depends on which shoppers happen to be in the
store, not how wrong the twin itself might be (milestone 5 measures that against the true store).
"""

from __future__ import annotations

from collections import OrderedDict

import numpy as np
import pandas as pd

from .scenario import PriceChange, PricePlan, Promotion, Scenario, StockOut, realised_price_change

HOUSEHOLD_ORDER = ["1", "2", "3-4", "5+"]
SENSITIVITY_ORDER = ["lowest", "low", "high", "highest"]


class WhatIf:
    """What-if questions for a fitted twin. `users` limits the store to some shoppers (all of them by default)."""

    def __init__(self, twin, world, users: np.ndarray | None = None, batch_size: int = 2500, cache_size: int = 16):
        self.twin = twin
        self.products = world.products.sort_values("product_id").reset_index(drop=True)
        self.shoppers = world.shoppers.sort_values("shopper_id").reset_index(drop=True)
        self.users = np.arange(twin.n_shoppers) if users is None else np.asarray(users)
        self.base_plan = PricePlan.from_world(world)
        self.weeks, _ = twin.window_weeks()
        self.batch_size = batch_size
        self.cache_size = cache_size
        self._base: OrderedDict[int, dict] = OrderedDict()

    # ---- expected purchases
    def category_outcome(self, plan: PricePlan, category: int) -> dict:
        """Expected purchases, units and revenue of every shopper x product in one category over the window."""
        parts = [self.twin.category_expectations(self.users[s:s + self.batch_size], category, plan.price,
                                                 plan.promo, plan.available)
                 for s in range(0, len(self.users), self.batch_size)]
        products = parts[0][0]
        purchases = np.vstack([p[1] for p in parts])
        spend = np.vstack([p[2] for p in parts])
        qty = self.twin.quantity[self.users, category][:, None]
        return {"products": products, "purchases": purchases, "units": purchases * qty, "revenue": spend * qty}

    def base_outcome(self, category: int) -> dict:
        """The store's own plan for one category (kept for the most recently used categories)."""
        if category in self._base:
            self._base.move_to_end(category)
        else:
            self._base[category] = self.category_outcome(self.base_plan, category)
            if len(self._base) > self.cache_size:
                self._base.popitem(last=False)
        return self._base[category]

    def category_units(self, plan: PricePlan | None = None) -> pd.DataFrame:
        """Expected units and revenue per category over the window (the store's own plan by default)."""
        plan = plan or self.base_plan
        rows = []
        for c in range(self.twin.cat.n_categories):
            out = self.category_outcome(plan, c)
            rows.append({"category_id": c, "units": float(out["units"].sum()), "revenue": float(out["revenue"].sum())})
        return pd.DataFrame(rows)

    # ---- one scenario
    def run(self, scenario: Scenario, n_boot: int = 200, seed: int = 0, shopper_detail: bool = False) -> dict:
        """Everything the report shows about one scenario, as plain numbers (plus a per-product table).
        `shopper_detail` also returns each shopper's expected units and revenue (used by the evaluation)."""
        cat = self.twin.cat
        window = (int(self.weeks.min()), int(self.weeks.max()))
        plan = scenario.apply(self.base_plan, self.products, window)
        target = scenario.targets(self.products)
        categories = np.unique(cat.category[target])
        scen_weeks = np.arange(scenario.weeks[0], scenario.weeks[1] + 1) if scenario.weeks else self.weeks
        in_window = np.intersect1d(scen_weeks, self.weeks)

        n_u = len(self.users)
        per_shopper = {k: np.zeros(n_u) for k in ("t_base", "t_new", "c_base", "c_new", "r_base", "r_new")}
        buys_base, buys_new = np.zeros(n_u), np.zeros(n_u)
        product_rows = []
        for c in categories:
            base, new = self.base_outcome(c), self.category_outcome(plan, c)
            is_t = target[base["products"]]
            per_shopper["t_base"] += base["units"][:, is_t].sum(axis=1)
            per_shopper["t_new"] += new["units"][:, is_t].sum(axis=1)
            per_shopper["c_base"] += base["units"].sum(axis=1)
            per_shopper["c_new"] += new["units"].sum(axis=1)
            per_shopper["r_base"] += base["revenue"].sum(axis=1)
            per_shopper["r_new"] += new["revenue"].sum(axis=1)
            buys_base += base["purchases"][:, is_t].sum(axis=1)
            buys_new += new["purchases"][:, is_t].sum(axis=1)
            product_rows.append(pd.DataFrame({
                "product_id": base["products"], "category_id": c, "target": is_t,
                "base_units": base["units"].sum(axis=0), "new_units": new["units"].sum(axis=0),
                "base_revenue": base["revenue"].sum(axis=0), "new_revenue": new["revenue"].sum(axis=0)}))
        products = pd.concat(product_rows, ignore_index=True).merge(
            self.products[["product_id", "name", "brand", "category"]], on="product_id")
        products["change"] = products.new_units - products.base_units

        tot = {k: float(v.sum()) for k, v in per_shopper.items()}
        bands = _bootstrap_bands(per_shopper, n_boot, seed)
        others = products[~products.target]
        moved = others.reindex(others.change.abs().sort_values(ascending=False).index).head(5)
        prices_changed = any(isinstance(ch, (PriceChange, Promotion)) for ch in scenario.changes)
        # for a stock-out the changed products always lose everything, so segments show the category instead
        segment_on = "category" if all(isinstance(ch, StockOut) for ch in scenario.changes) else "target"
        out = {
            "name": scenario.name,
            "description": scenario.describe(),
            "weeks": [int(in_window.min()), int(in_window.max())] if len(in_window) else None,
            "n_target_products": int(target.sum()),
            "categories": [str(cat_name) for cat_name in
                           self.products.drop_duplicates("category_id").set_index("category_id")
                           .loc[categories, "category"]],
            "price_change_pct": (realised_price_change(self.base_plan, plan, np.flatnonzero(target), in_window)
                                 if prices_changed else None),
            "target_units": _summary(tot["t_base"], tot["t_new"], bands["t"]),
            "category_units": _summary(tot["c_base"], tot["c_new"], bands["c"]),
            "category_revenue": _summary(tot["r_base"], tot["r_new"], bands["r"]),
            # expected number of shoppers buying a targeted product at least once
            "target_buyers": {"base": float((-np.expm1(-buys_base)).sum()),
                              "new": float((-np.expm1(-buys_new)).sum())},
            "substitution": {
                "target_change": tot["t_new"] - tot["t_base"],
                "other_products_change": float(others.change.sum()),
                "category_change": tot["c_new"] - tot["c_base"],
                "most_affected_other_products": moved[["name", "brand", "change"]].to_dict("records"),
            },
            "segments_measure": segment_on,
            "by_segment": self._segments(per_shopper, "c" if segment_on == "category" else "t"),
            "products": products,  # full table (not saved to JSON)
        }
        if shopper_detail:
            out["per_shopper"] = {"users": self.users, **per_shopper}  # arrays (not saved to JSON)
        return out

    def _segments(self, per_shopper: dict, key: str) -> dict:
        """% change in units (key "t": the changed products, "c": their categories) by household size and by
        learned price sensitivity."""
        sh = self.shoppers.iloc[self.users]
        traits = self.twin.shopper_traits().iloc[self.users]
        # ranked first, so equal values (e.g. a twin without personal traits) still split into quarters
        quartile = pd.qcut(traits.price_sens.rank(method="first").to_numpy(), 4, labels=SENSITIVITY_ORDER)
        out = {}
        for name, labels, order in (("household size", sh.household_band.astype(str).to_numpy(), HOUSEHOLD_ORDER),
                                    ("price sensitivity (learned)", np.asarray(quartile).astype(str),
                                     SENSITIVITY_ORDER)):
            rows = {}
            for label in [x for x in order if x in set(labels)] + sorted(set(labels) - set(order)):
                m = labels == label
                rows[label] = _pct(per_shopper[f"{key}_base"][m].sum(), per_shopper[f"{key}_new"][m].sum())
            out[name] = rows
        return out


def _pct(base: float, new: float) -> float:
    return float(100 * (new / base - 1)) if base > 0 else float("nan")


def _summary(base: float, new: float, band: list | None) -> dict:
    return {"base": base, "new": new, "pct": _pct(base, new), "band_pct": band}


def _bootstrap_bands(per_shopper: dict, n_boot: int, seed: int) -> dict:
    """5th-95th percentile of the % change when the shoppers are resampled (Poisson weights)."""
    rng = np.random.default_rng(seed)
    w = rng.poisson(1.0, size=(n_boot, len(per_shopper["t_base"]))).astype(np.float64)
    bands = {}
    for key in ("t", "c", "r"):
        base, new = w @ per_shopper[f"{key}_base"], w @ per_shopper[f"{key}_new"]
        with np.errstate(divide="ignore", invalid="ignore"):
            pct = 100 * (new / base - 1)
        pct = pct[np.isfinite(pct)]
        bands[key] = [float(np.percentile(pct, 5)), float(np.percentile(pct, 95))] if len(pct) else None
    return bands
