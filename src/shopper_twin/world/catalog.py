"""Build the store's product catalogue and its week-by-week price schedule."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import spec
from .config import WorldConfig


@dataclass
class Catalog:
    # Categories (index c)
    cat_name: np.ndarray
    cat_dept: np.ndarray  # department index
    cat_staple: np.ndarray  # bool
    cat_cycle: np.ndarray  # typical repurchase cycle in days (0 for discretionary)
    cat_median: np.ndarray  # median price
    cat_peak: np.ndarray
    cat_amp: np.ndarray
    cat_requires: np.ndarray  # "", "dog", "cat", "baby", "car"
    cat_center: np.ndarray  # [C, D] hidden position of the category in taste space
    cat_products: np.ndarray  # [C, max_products] product ids, padded with -1
    # Brands (index b)
    brand_name: np.ndarray
    brand_dept: np.ndarray
    brand_is_store: np.ndarray
    brand_quality: np.ndarray
    brand_premium: np.ndarray  # log price premium over the category median
    # Products (index i, equal to product_id)
    name: np.ndarray
    category: np.ndarray
    department: np.ndarray
    brand: np.ndarray
    is_store: np.ndarray
    quality: np.ndarray  # hidden, 0..1
    style: np.ndarray  # [P, D] hidden, what the product is like
    base_price: np.ndarray
    popularity: np.ndarray  # hidden log-popularity (shelf placement, awareness)
    rating: np.ndarray  # public, noisy view of quality
    n_reviews: np.ndarray

    @property
    def n_products(self) -> int:
        return len(self.category)

    @property
    def n_categories(self) -> int:
        return len(self.cat_name)

    @property
    def embedding(self) -> np.ndarray:
        """Full hidden product vector: category position plus the product's own style."""
        return self.cat_center[self.category] + self.style


def shelf_price(p: np.ndarray) -> np.ndarray:
    """Round to retail-looking prices: $x.49 / $x.99 below $20, whole dollars minus a cent above."""
    p = np.maximum(p, 0.5)
    small = np.round(p * 2) / 2 - 0.01
    big = np.round(p) - 0.01
    return np.where(p < 20, small, big).astype(np.float32)


def _allocate(total: int, weights: np.ndarray, minimum: int = 4) -> np.ndarray:
    raw = total * weights / weights.sum()
    counts = np.maximum(minimum, np.floor(raw)).astype(int)
    # hand out the rounding remainder to the largest fractional parts
    short = total - counts.sum()
    if short > 0:
        order = np.argsort(-(raw - np.floor(raw)))
        counts[order[:short]] += 1
    return counts


def build_catalog(cfg: WorldConfig, rng: np.random.Generator) -> Catalog:
    cats = spec.CATEGORIES
    depts = spec.DEPARTMENTS
    dim = cfg.embed_dim
    n_cat = len(cats)

    cat_dept = np.array([depts.index(c.department) for c in cats])
    cat_center = rng.normal(size=(n_cat, dim)) / np.sqrt(dim)
    # categories in the same department sit near each other
    dept_center = rng.normal(size=(len(depts), dim)) / np.sqrt(dim)
    cat_center = 0.6 * cat_center + dept_center[cat_dept]

    # ---- brands: a few per department plus the store brand everywhere
    combos = [p + s for p in spec.BRAND_PREFIXES for s in spec.BRAND_SUFFIXES]
    picks = rng.choice(len(combos), size=len(depts) * cfg.brands_per_department, replace=False)
    b_name, b_dept, b_store, b_q, b_prem, b_pop = [], [], [], [], [], []
    k = 0
    for d in range(len(depts)):
        for _ in range(cfg.brands_per_department):
            q = rng.beta(4, 3)
            b_name.append(combos[picks[k]])
            b_dept.append(d)
            b_store.append(False)
            b_q.append(q)
            b_prem.append(0.9 * (q - 0.55) + rng.normal(0, 0.1))
            b_pop.append(rng.normal(0, 0.6))
            k += 1
        b_name.append(spec.STORE_BRAND)
        b_dept.append(d)
        b_store.append(True)
        b_q.append(rng.beta(3, 4))
        b_prem.append(-0.35 + rng.normal(0, 0.05))
        b_pop.append(0.5)
    brand_dept = np.array(b_dept)
    brand_is_store = np.array(b_store)
    brand_quality = np.array(b_q)
    brand_premium = np.array(b_prem)
    brand_pop = np.array(b_pop)

    # ---- products
    counts = _allocate(cfg.n_products, np.array([c.breadth for c in cats]))
    category = np.repeat(np.arange(n_cat), counts)
    n = len(category)
    department = cat_dept[category]

    brand = np.empty(n, dtype=np.int64)
    for d in range(len(depts)):
        members = np.flatnonzero(department == d)
        options = np.flatnonzero(brand_dept == d)
        w = np.exp(brand_pop[options])
        w[brand_is_store[options]] *= 1.5  # the store stocks plenty of its own brand
        brand[members] = rng.choice(options, size=len(members), p=w / w.sum())

    is_store = brand_is_store[brand]
    quality = np.clip(brand_quality[brand] + rng.normal(0, 0.1, n), 0.02, 0.98)
    style = rng.normal(size=(n, dim)) * (0.8 / np.sqrt(dim))
    cat_median = np.array([c.median_price for c in cats])
    log_price = (np.log(cat_median[category]) + brand_premium[brand]
                 + 0.6 * (quality - brand_quality[brand]) + rng.normal(0, 0.15, n))
    base_price = shelf_price(np.exp(log_price))
    popularity = 0.7 * brand_pop[brand] + rng.normal(0, 0.5, n)
    rating = np.round(np.clip(2.6 + 2.4 * quality + rng.normal(0, 0.35, n), 1, 5), 1)
    n_reviews = rng.poisson(40 * np.exp(popularity)) + 1

    variants = np.array(spec.PRODUCT_VARIANTS)
    b_name_arr = np.array(b_name)
    cat_name = np.array([c.name for c in cats])
    names = np.array([f"{b_name_arr[b]} {cat_name[c]} {variants[v]}"
                      for b, c, v in zip(brand, category, rng.integers(0, len(variants), n))])

    max_p = counts.max()
    cat_products = -np.ones((n_cat, max_p), dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]])
    for c in range(n_cat):
        cat_products[c, : counts[c]] = np.arange(starts[c], starts[c] + counts[c])

    return Catalog(
        cat_name=cat_name,
        cat_dept=cat_dept,
        cat_staple=np.array([c.staple for c in cats]),
        cat_cycle=np.array([c.cycle_days for c in cats], dtype=float),
        cat_median=cat_median,
        cat_peak=np.array([c.peak_day for c in cats], dtype=float),
        cat_amp=np.array([c.season_amp for c in cats], dtype=float),
        cat_requires=np.array([c.requires or "" for c in cats]),
        cat_center=cat_center.astype(np.float32),
        cat_products=cat_products,
        brand_name=b_name_arr,
        brand_dept=brand_dept,
        brand_is_store=brand_is_store,
        brand_quality=brand_quality,
        brand_premium=brand_premium,
        name=names,
        category=category,
        department=department,
        brand=brand,
        is_store=is_store,
        quality=quality.astype(np.float32),
        style=style.astype(np.float32),
        base_price=base_price,
        popularity=popularity.astype(np.float32),
        rating=rating.astype(np.float32),
        n_reviews=n_reviews,
    )


@dataclass
class PriceSchedule:
    """Prices change weekly: occasional regular price changes plus one-week promotions."""

    regular: np.ndarray  # [P, W] regular shelf price
    discount: np.ndarray  # [P, W] promo discount, 0 = not on promotion

    @property
    def price(self) -> np.ndarray:
        return shelf_price(self.regular * (1 - self.discount))

    @property
    def n_weeks(self) -> int:
        return self.regular.shape[1]


def build_price_schedule(cfg: WorldConfig, catalog: Catalog, rng: np.random.Generator) -> PriceSchedule:
    n_weeks = -(-cfg.n_days // 7)
    n = catalog.n_products
    level = np.zeros((n, n_weeks))
    changes = rng.random((n, n_weeks)) < cfg.price_change_prob
    changes[:, 0] = False
    steps = rng.normal(0, cfg.price_change_sd, (n, n_weeks))
    current = np.zeros(n)
    for w in range(n_weeks):
        # a change moves the price to a new level near the base price (no runaway drift)
        current = np.where(changes[:, w], 0.5 * current + steps[:, w], current)
        level[:, w] = current
    regular = shelf_price(catalog.base_price[:, None] * np.exp(level))

    on_promo = rng.random((n, n_weeks)) < cfg.promo_share
    discount = np.where(on_promo, rng.choice([0.10, 0.15, 0.20, 0.25, 0.30], (n, n_weeks)), 0.0)
    return PriceSchedule(regular=regular.astype(np.float32), discount=discount.astype(np.float32))
