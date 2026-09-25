"""Build the shopper population: hidden personal traits plus the public profile a store would see."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import spec
from .catalog import Catalog
from .config import WorldConfig


@dataclass
class Population:
    # hidden traits (answer key)
    segment: np.ndarray
    price_sensitivity: np.ndarray
    quality_weight: np.ndarray
    store_brand_affinity: np.ndarray
    brand_loyalty: np.ndarray
    habit_strength: np.ndarray
    visits_per_week: np.ndarray
    discretionary_per_visit: np.ndarray
    household_size: np.ndarray
    has_dog: np.ndarray
    has_cat: np.ndarray
    has_baby: np.ndarray  # at the start of the run
    has_car: np.ndarray
    shop_hour: np.ndarray
    taste: np.ndarray  # [U, D] at the start of the run
    fav_brand: np.ndarray  # [U, n_departments] brand index
    staple_uses: np.ndarray  # [U, C] bool, only meaningful for staple categories
    staple_cycle: np.ndarray  # [U, C] personal repurchase cycle in days
    disc_weight: np.ndarray  # [U, C] interest in each discretionary category
    # life events: one row per event
    event_shopper: np.ndarray
    event_type: np.ndarray  # "new_baby" or "moved_house"
    event_day: np.ndarray
    # public profile
    region: np.ndarray
    age_band: np.ndarray
    signup_date: np.ndarray

    @property
    def n(self) -> int:
        return len(self.segment)


def eligibility(catalog: Catalog, has: dict[str, np.ndarray]) -> np.ndarray:
    """[U, C] bool: can this shopper buy from this category at all (no dog, no dog food)."""
    n = len(next(iter(has.values())))
    elig = np.ones((n, catalog.n_categories), dtype=bool)
    for c, req in enumerate(catalog.cat_requires):
        if req:
            elig[:, c] = has[req]
    return elig


def staple_use_prob(catalog: Catalog, dept_boost: np.ndarray, base: float) -> np.ndarray:
    """[U, C] chance a shopper uses each staple category, given their department interests."""
    p = np.clip(base * dept_boost[:, catalog.cat_dept], 0.02, 0.95)
    # if you own a dog you buy dog food: "requires" categories are near certain once eligible
    p[:, catalog.cat_requires != ""] = 0.95
    return p


def build_population(cfg: WorldConfig, catalog: Catalog, rng: np.random.Generator) -> Population:
    segs = spec.SEGMENTS
    n = cfg.n_shoppers
    dim = cfg.embed_dim
    shares = np.array([s.share for s in segs])
    segment = rng.choice(len(segs), size=n, p=shares / shares.sum())

    def seg_attr(attr):
        return np.array([getattr(s, attr) for s in segs])[segment]

    def lognormal(mean, sd):
        return mean * np.exp(rng.normal(0, sd, n) - sd**2 / 2)

    price_sensitivity = lognormal(seg_attr("price_sensitivity"), 0.35)
    quality_weight = lognormal(seg_attr("quality_weight"), 0.3)
    store_brand_affinity = seg_attr("store_brand_affinity") + rng.normal(0, 0.4, n)
    brand_loyalty = lognormal(0.8, 0.4)
    habit_strength = lognormal(1.0, 0.4)
    visits_per_week = np.clip(lognormal(seg_attr("visits_per_week"), 0.4), 0.1, 6)
    discretionary_per_visit = lognormal(seg_attr("discretionary_per_visit"), 0.3)
    household_size = np.clip(1 + rng.poisson(seg_attr("household_size") - 1), 1, 8)
    has = {
        "dog": rng.random(n) < seg_attr("p_dog"),
        "cat": rng.random(n) < seg_attr("p_cat"),
        "baby": rng.random(n) < seg_attr("p_baby"),
        "car": rng.random(n) < seg_attr("p_car"),
    }
    shop_hour = np.clip(seg_attr("shop_hour") + rng.normal(0, 1.5, n), 7, 21)

    seg_center = rng.normal(size=(len(segs), dim)) / np.sqrt(dim)
    taste = seg_center[segment] + rng.normal(size=(n, dim)) / np.sqrt(dim)

    # department interest: segment boost times a personal random factor
    dept_boost_seg = np.array([[s.dept_boost.get(d, 1.0) for d in spec.DEPARTMENTS] for s in segs])
    dept_boost = dept_boost_seg[segment] * rng.gamma(4.0, 0.25, (n, len(spec.DEPARTMENTS)))

    elig = eligibility(catalog, has)
    staple = catalog.cat_staple
    uses_p = staple_use_prob(catalog, dept_boost, cfg.rules.staple_use_base)
    staple_uses = elig & staple[None, :] & (rng.random(elig.shape) < uses_p)
    hh_factor = (2.5 / household_size) ** 0.5
    staple_cycle = (catalog.cat_cycle[None, :] * hh_factor[:, None]
                    * np.exp(rng.normal(0, 0.2, elig.shape)))
    staple_cycle = np.maximum(staple_cycle, 2.0).astype(np.float32)
    disc_weight = (dept_boost[:, catalog.cat_dept] * rng.gamma(1.0, 1.0, elig.shape)
                   * elig * ~staple[None, :]).astype(np.float32)

    # favourite brand per department: the one with the best fit to their price/quality taste
    n_dept = len(spec.DEPARTMENTS)
    fav_brand = np.empty((n, n_dept), dtype=np.int64)
    for d in range(n_dept):
        options = np.flatnonzero(catalog.brand_dept == d)
        score = (2 * quality_weight[:, None] * (catalog.brand_quality[options] - 0.5)
                 - price_sensitivity[:, None] * catalog.brand_premium[options]
                 + store_brand_affinity[:, None] * catalog.brand_is_store[options])
        score = score + rng.gumbel(size=score.shape)
        fav_brand[:, d] = options[np.argmax(score, axis=1)]

    # life events: a small share of shoppers have a baby or move house mid-run
    n_events = int(round(cfg.life_event_rate * n))
    event_shopper = rng.choice(n, size=n_events, replace=False)
    event_type = np.where(~has["baby"][event_shopper] & (rng.random(n_events) < 0.5),
                          "new_baby", "moved_house")
    event_day = rng.integers(int(0.15 * cfg.n_days), int(0.85 * cfg.n_days), n_events)

    age_probs = np.array([s.age_probs for s in segs])
    age_cdf = np.cumsum(age_probs, axis=1)[segment]
    age_idx = np.minimum((rng.random(n)[:, None] > age_cdf).sum(axis=1), len(spec.AGE_BANDS) - 1)
    start = np.datetime64(cfg.start_date)
    signup = start - rng.integers(1, 1500, n).astype("timedelta64[D]")

    return Population(
        segment=segment,
        price_sensitivity=price_sensitivity.astype(np.float32),
        quality_weight=quality_weight.astype(np.float32),
        store_brand_affinity=store_brand_affinity.astype(np.float32),
        brand_loyalty=brand_loyalty.astype(np.float32),
        habit_strength=habit_strength.astype(np.float32),
        visits_per_week=visits_per_week.astype(np.float32),
        discretionary_per_visit=discretionary_per_visit.astype(np.float32),
        household_size=household_size,
        has_dog=has["dog"],
        has_cat=has["cat"],
        has_baby=has["baby"],
        has_car=has["car"],
        shop_hour=shop_hour.astype(np.float32),
        taste=taste.astype(np.float32),
        fav_brand=fav_brand,
        staple_uses=staple_uses,
        staple_cycle=staple_cycle,
        disc_weight=disc_weight,
        event_shopper=event_shopper,
        event_type=event_type,
        event_day=event_day,
        region=np.array(spec.REGIONS)[rng.integers(0, len(spec.REGIONS), n)],
        age_band=np.array(spec.AGE_BANDS)[age_idx],
        signup_date=signup,
    )
