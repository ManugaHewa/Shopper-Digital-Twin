"""Simpler ways to answer a what-if question, to compare the twin with.

  No reaction          Shoppers keep buying exactly what they bought before; a product that is out of stock
                       simply loses its sales. What you get if you ignore customer behaviour.
  Store-wide elasticity  The classic pricing-team model, estimated from the store's own weekly sales: one
                       elasticity for switching between products in a category, one for how much a whole
                       category sells when its average price moves, and a promotion lift for each. The same
                       for every product and every shopper.
  Twin without personal traits  The twin's choice model fitted again with group values only: every shopper
                       gets the taste and traits of their group (age band x household size). Trip rates and
                       memory are the twin's own. Shows how much knowing each shopper matters.

All of them only use data from before the prediction window.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd

from ..sim.scenario import PricePlan
from ..twin.choice import fit_choice_model
from ..twin.twin import torch_threads


def weekly_units(purchases: pd.DataFrame, n_products: int, n_weeks: int) -> np.ndarray:
    """[products, weeks] units sold (`purchases` needs product_id, day, quantity)."""
    week = purchases.day.to_numpy() // 7
    keep = week < n_weeks
    out = np.zeros((n_products, n_weeks))
    np.add.at(out, (purchases.product_id.to_numpy()[keep], week[keep]), purchases.quantity.to_numpy()[keep])
    return out


class StoreElasticity:
    """The classic pricing-team model, fitted to the store's weekly sales per product. Two levels:

      Switching   a product's share of its category: share x exp(elasticity x log(price change)
                  + promo_lift x [promotion started]), re-normalised over the category's products in stock
      Category    the category's total units: x exp(category_elasticity x change in the category's average log
                  price + category_promo_lift x change in the share of it on promotion)

    Both are log-linear models of weekly units fitted by Poisson maximum likelihood (the standard way to fit
    counts with zeros). Switching: product effects + category-week effects, so the elasticity is learned from each
    product's price moving against the rest of its category. Category: category effects + week effects, learned
    from each category's average price moving over time. Everyone is treated the same: no personal traits."""

    def fit(self, purchases: pd.DataFrame, plan: PricePlan, category: np.ndarray, first_week: int
            ) -> "StoreElasticity":
        """Weeks before `first_week` (the prediction window's first week) only."""
        n_products = len(category)
        self.category = np.asarray(category)
        self.n_categories = int(self.category.max()) + 1
        y = weekly_units(purchases, n_products, first_week)
        log_price, promo = np.log(plan.price[:, :first_week]), plan.promo[:, :first_week]
        sold = y.sum(axis=1) > 0
        x = np.stack([log_price, promo], axis=-1)
        self.elasticity, self.promo_lift = _poisson_fit(y[sold], x[sold], self.category[sold])

        # each category's average log price (relative to each product's usual price) and share on promotion,
        # weighted by the products' share of the category's units over the same weeks
        weight = y.sum(axis=1)
        weight = weight / np.maximum(self._by_category(weight)[self.category], 1e-12)
        centred = log_price - log_price.mean(axis=1, keepdims=True)
        y_cat = self._by_category(y)
        x_cat = np.stack([self._by_category(weight[:, None] * centred), self._by_category(weight[:, None] * promo)],
                         axis=-1)
        sold = y_cat.sum(axis=1) > 0
        self.category_elasticity, self.category_promo_lift = _poisson_fit(
            y_cat[sold], x_cat[sold], np.zeros(int(sold.sum()), dtype=np.int64))

        # the normal weekly level each product is forecast from: its average over the last 4 weeks
        self.level = y[:, max(first_week - 4, 0):first_week].mean(axis=1)
        level_cat = self._by_category(self.level)[self.category]
        self.share = np.divide(self.level, level_cat, out=np.zeros_like(self.level), where=level_cat > 0)
        return self

    def _by_category(self, x: np.ndarray) -> np.ndarray:
        """Sum the rows of x (one per product) within each category."""
        out = np.zeros((self.n_categories, *x.shape[1:]))
        np.add.at(out, self.category, x)
        return out

    def predict(self, base: PricePlan, new: PricePlan, target: np.ndarray, in_categories: np.ndarray,
                weeks: np.ndarray, week_days: np.ndarray) -> dict:
        """% change in units and revenue for the changed products and their categories, over the window's
        `weeks` (with `week_days` of the window in each)."""
        b_units = self.level[:, None] * week_days[None, :] / 7
        change = np.log(new.price[:, weeks] / base.price[:, weeks])
        promo = new.promo[:, weeks] - base.promo[:, weeks]
        s = self.share[:, None]
        total = np.exp(self.category_elasticity * self._by_category(s * change)
                       + self.category_promo_lift * self._by_category(s * promo))
        pull = s * np.exp(self.elasticity * change + self.promo_lift * promo) * new.available[:, weeks]
        norm = self._by_category(pull)[self.category]
        n_units = np.divide(self._by_category(b_units)[self.category] * total[self.category] * pull, norm,
                            out=np.zeros_like(b_units), where=norm > 0)
        return _summarise(b_units, n_units, base.price[:, weeks], new.price[:, weeks], target, in_categories)


def _poisson_fit(y: np.ndarray, x: np.ndarray, group: np.ndarray, iterations: int = 200) -> tuple[float, float]:
    """log E[y[r, w]] = row effect + (group of r, week) effect + x[r, w] . beta, by Poisson maximum likelihood.
    y [rows, weeks], x [rows, weeks, 2]. Returns beta."""
    beta = np.zeros(x.shape[-1])
    delta = np.zeros((int(group.max()) + 1, y.shape[1]))
    for _ in range(iterations):
        lin = x @ beta
        # row effects, then group-week effects, each in closed form given the rest
        alpha = np.log(np.maximum(y.sum(axis=1), 1e-12) / np.exp(delta[group] + lin).sum(axis=1))
        num = np.zeros_like(delta)
        den = np.zeros_like(delta)
        np.add.at(num, group, y)
        np.add.at(den, group, np.exp(alpha[:, None] + lin))
        delta = np.log(np.maximum(num, 1e-12) / np.maximum(den, 1e-300))
        # one Newton step for the slopes, with x centred within each row (the row effects already take up each
        # row's average, so only changes over time are informative)
        mu = np.exp(alpha[:, None] + delta[group] + lin)
        xc = x - (mu[..., None] * x).sum(axis=1, keepdims=True) / mu.sum(axis=1)[:, None, None]
        grad = ((y - mu)[..., None] * xc).reshape(-1, x.shape[-1]).sum(axis=0)
        hess = (mu[..., None, None] * xc[..., :, None] * xc[..., None, :]).reshape(
            -1, x.shape[-1], x.shape[-1]).sum(axis=0)
        step = np.linalg.solve(hess, grad)
        beta = beta + step
        if np.abs(step).max() < 1e-8:
            break
    return float(beta[0]), float(beta[1])


def no_reaction(base: PricePlan, new: PricePlan, level: np.ndarray, target: np.ndarray, in_categories: np.ndarray,
                weeks: np.ndarray, week_days: np.ndarray) -> dict:
    """Everyone keeps buying the same things; products out of stock lose their sales and nobody switches.
    `level` is each product's normal weekly units."""
    b_units = level[:, None] * week_days[None, :] / 7
    n_units = b_units * new.available[:, weeks]
    return _summarise(b_units, n_units, base.price[:, weeks], new.price[:, weeks], target, in_categories)


def _summarise(b_units, n_units, b_price, n_price, target, in_categories) -> dict:
    def pct(new, old):
        return float(100 * (new / old - 1)) if old > 0 else float("nan")

    out = {}
    for key, mask, b, n in (("target_units", target, b_units, n_units),
                            ("category_units", in_categories, b_units, n_units),
                            ("category_revenue", in_categories, b_units * b_price, n_units * n_price)):
        out[key] = {"base": float(b[mask].sum()), "new": float(n[mask].sum()),
                    "pct": pct(n[mask].sum(), b[mask].sum())}
    return out


def without_personal_traits(twin, trips):
    """A copy of the twin whose choice model is fitted again, on the same `trips` (before the twin's cutoff) and
    with the same settings, but with group values only. Its purchases per trip are calibrated the same way as
    the twin's, so both start from the same normal sales."""
    pooled = copy.copy(twin)
    trips = trips.subset(trips.day < twin.cutoff_day)
    with torch_threads(twin.threads):
        pooled.choice = fit_choice_model(
            trips, twin.cat, twin.groups, twin.n_shoppers, dim=twin.dim, epochs=twin.epochs,
            batch_size=twin.batch_size, lr=twin.lr, view_weight=twin.view_weight, taste_reg=twin.taste_reg,
            trait_reg=twin.trait_reg, seed=twin.seed, personal=False)
        pooled.purchase_scale = pooled._calibrate_purchases(trips, max(twin.cutoff_day - twin.horizon, 1))
    return pooled
