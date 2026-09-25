"""Settings for generating a fake store world."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace


@dataclass(frozen=True)
class Rules:
    """Constants of the true shopper behaviour. These are part of the answer key.

    A shopper's utility for product i is:
        taste_scale * (taste . style_i)
      + quality_weight * 2 * (quality_i - 0.5)
      + store_brand_affinity * is_store_brand_i
      + brand_loyalty * [brand_i is their favourite in that department]
      + habit_strength * [i is the product they bought last time in this category]
      - price_sensitivity * log(price_i / category_median_price)
      + promo_salience * [i is on promotion]
      + Gumbel noise
    and they buy the best viewed product if it beats the "buy nothing" option.
    """

    taste_scale: float = 2.5
    promo_salience: float = 0.3
    outside_staple: float = 1.5  # "buy nothing" utility when a staple is due
    outside_discretionary: float = 4.0  # "buy nothing" utility when browsing (higher: often just looking)
    staple_due_visit_prob: float = 0.85  # chance a due staple is on this visit's list
    lost_to_competitor_prob: float = 0.5  # a due staple not bought here is often bought elsewhere
    staple_use_base: float = 0.35  # base chance a shopper uses a given staple category
    views_staple: int = 4  # products viewed in a staple category
    views_discretionary: int = 8  # products viewed in a discretionary category
    view_taste_weight: float = 0.5  # how much taste steers which products get viewed
    view_promo_boost: float = 1.0
    view_habit_boost: float = 2.0
    abandon_cart_prob: float = 0.08  # chance the runner-up product is carted and abandoned
    weekday_visit_mult: tuple[float, ...] = (0.85, 0.8, 0.85, 0.9, 1.1, 1.35, 1.15)  # Mon..Sun


@dataclass(frozen=True)
class WorldConfig:
    name: str = "small"
    seed: int = 42
    n_shoppers: int = 10_000
    n_products: int = 5_000
    n_days: int = 180
    start_date: str = "2026-01-05"  # a Monday
    warmup_days: int = 28  # simulated before day 0 but not recorded
    embed_dim: int = 16
    brands_per_department: int = 8
    promo_share: float = 0.04  # share of products on promotion in a given week
    price_change_prob: float = 0.06  # weekly chance a product's regular price is changed
    price_change_sd: float = 0.08  # size of a regular price change (log scale)
    life_event_rate: float = 0.04  # share of shoppers who have a life event during the run
    drift_sd: float = 0.02  # weekly taste drift per dimension
    chunk_pairs: int = 10_000  # memory control for the simulator
    rules: Rules = Rules()

    def to_dict(self) -> dict:
        return asdict(self)


PRESETS: dict[str, WorldConfig] = {
    "tiny": WorldConfig(name="tiny", n_shoppers=1_000, n_products=1_000, n_days=60),
    "small": WorldConfig(name="small"),
    "full": WorldConfig(name="full", n_shoppers=100_000, n_days=180),
}


def get_config(preset: str = "small", **overrides) -> WorldConfig:
    """Return a preset, optionally with some fields changed, e.g. get_config("tiny", seed=7)."""
    return replace(PRESETS[preset], **overrides)
