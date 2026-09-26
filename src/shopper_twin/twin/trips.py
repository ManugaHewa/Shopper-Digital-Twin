"""Turn the event log into shopping trips: one row per (session, category) the shopper looked at.

A trip is the unit the twin learns from. For each trip we know which products the shopper viewed
(in order), which one they bought (if any), the week (so the price they saw), and two things known
*before* the trip: the product they bought last time in this category (habit) and how often they
chose each brand in the department's *other* categories before (brand loyalty; other categories only,
because repeat buys in the same category are already explained by habit).
"""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np
import pandas as pd

MAX_VIEWS = 8  # the most products viewed in one trip


@dataclass
class Catalogue:
    """Public product facts the twin uses, as arrays indexed by product id."""

    category: np.ndarray  # [P] category id
    department: np.ndarray  # [P] department id
    brand: np.ndarray  # [P] brand id (0..n_brands-1)
    is_store: np.ndarray  # [P] 1.0 for the store's own brand
    price: np.ndarray  # [P, W] shelf price per week (the store's price plan)
    promo: np.ndarray  # [P, W] 1.0 when on promotion
    regular: np.ndarray  # [P, W] regular price (before any promotion discount)
    median: np.ndarray  # [C] median base price per category
    members: np.ndarray  # [C, M] product ids per category, padded with -1
    cat_brand: np.ndarray  # [P] id of the product's (category, brand) pair

    @property
    def n_products(self) -> int:
        return len(self.category)

    @property
    def n_categories(self) -> int:
        return len(self.median)

    @property
    def n_weeks(self) -> int:
        return self.price.shape[1]

    def log_price_ratio(self) -> np.ndarray:
        """[P, W] log(price / category median): the price signal shoppers react to."""
        return np.log(self.price / self.median[self.category][:, None]).astype(np.float32)

    @classmethod
    def from_world(cls, world) -> "Catalogue":
        p = world.products.sort_values("product_id")
        n = len(p)
        category = p.category_id.to_numpy().astype(np.int64)
        department = pd.factorize(p.department)[0].astype(np.int64)
        brand = pd.factorize(p.department + "|" + p.brand)[0].astype(np.int64)
        ph = world.price_history
        n_weeks = int(ph.week.max()) + 1
        price = np.zeros((n, n_weeks), dtype=np.float32)
        promo = np.zeros((n, n_weeks), dtype=np.float32)
        regular = np.zeros((n, n_weeks), dtype=np.float32)
        price[ph.product_id.to_numpy(), ph.week.to_numpy()] = ph.price.to_numpy()
        regular[ph.product_id.to_numpy(), ph.week.to_numpy()] = ph.regular_price.to_numpy()
        promo[ph.product_id.to_numpy(), ph.week.to_numpy()] = (ph.promo_discount.to_numpy() > 0)
        n_cat = int(category.max()) + 1
        median = np.array([np.median(p.base_price.to_numpy()[category == c]) for c in range(n_cat)], np.float32)
        counts = np.bincount(category, minlength=n_cat)
        members = -np.ones((n_cat, counts.max()), dtype=np.int64)
        order = np.argsort(category, kind="stable")
        starts = np.r_[0, np.cumsum(counts)[:-1]]
        slot = np.arange(n) - np.repeat(starts, counts)
        members[category[order], slot] = order
        cat_brand = pd.factorize(category * (int(brand.max()) + 1) + brand)[0].astype(np.int64)
        return cls(category=category, department=department, brand=brand,
                   is_store=p.is_store_brand.to_numpy().astype(np.float32), price=price, promo=promo,
                   regular=regular, median=median, members=members, cat_brand=cat_brand)

    def frozen(self, week: int) -> tuple[np.ndarray, np.ndarray]:
        """Price and promo arrays as if nothing after `week` were known: every later week keeps that week's
        regular price and has no promotions."""
        price, promo = self.price.copy(), self.promo.copy()
        price[:, week + 1:] = self.regular[:, [week]]
        promo[:, week + 1:] = 0
        return price, promo


@dataclass
class Trips:
    """One row per trip, sorted by time. Product slots hold -1 when unused."""

    shopper: np.ndarray  # [T] int64
    category: np.ndarray  # [T]
    day: np.ndarray  # [T]
    viewed: np.ndarray  # [T, MAX_VIEWS] product ids in view order
    chosen: np.ndarray  # [T] slot of the purchased product in `viewed`, or -1 (bought nothing)
    last_bought: np.ndarray  # [T] product bought last time in this category before the trip, or -1
    brand_share: np.ndarray  # [T, MAX_VIEWS] share of the shopper's earlier purchases in the department's
    #                          other categories that were the viewed product's brand

    def __len__(self) -> int:
        return len(self.shopper)

    @property
    def week(self) -> np.ndarray:
        return self.day // 7

    @property
    def bought(self) -> np.ndarray:
        return self.chosen >= 0

    def subset(self, mask: np.ndarray) -> "Trips":
        """The trips where `mask` is True."""
        return Trips(**{f.name: getattr(self, f.name)[mask] for f in fields(self)})


def build_trips(events: pd.DataFrame, cat: Catalogue) -> Trips:
    """Group view and purchase events into trips.

    `events` needs session_id. With a timestamp column the views are put in time order; without one,
    the rows must already be in the order they happened (as in the event log).
    """
    need = {"session_id", "shopper_id", "product_id", "event_type", "day"}
    missing = need - set(events.columns)
    if missing:
        raise ValueError(f"events are missing columns {missing}; load them with extra_columns=('session_id', ...)")
    product = events.product_id.to_numpy()
    n_cat = cat.n_categories
    # one id per (session, category): a trip. Sessions are numbered in time order, so ids sort by time.
    key = events.session_id.to_numpy().astype(np.int64) * n_cat + cat.category[product]

    # views grouped by trip, in the order they happened inside each trip (lexsort is stable)
    v = np.flatnonzero((events.event_type == "view").to_numpy())
    if "timestamp" in events.columns:
        v = v[np.lexsort((events.timestamp.to_numpy()[v], key[v]))]
    else:
        v = v[np.argsort(key[v], kind="stable")]
    trip_keys, first, counts = np.unique(key[v], return_index=True, return_counts=True)
    n = len(trip_keys)
    trip_of_view = np.repeat(np.arange(n), counts)
    slot = np.arange(len(v)) - first[trip_of_view]
    keep = slot < MAX_VIEWS
    viewed = -np.ones((n, MAX_VIEWS), dtype=np.int64)
    viewed[trip_of_view[keep], slot[keep]] = product[v[keep]]
    shopper = events.shopper_id.to_numpy()[v[first]].astype(np.int64)
    day = events.day.to_numpy()[v[first]].astype(np.int64)
    category = (trip_keys % n_cat).astype(np.int64)
    del v, trip_of_view, slot, keep

    # what was bought on each trip (the first purchase in the log; there is at most one)
    b = np.flatnonzero((events.event_type == "purchase").to_numpy())
    buy_keys, buy_first = np.unique(key[b], return_index=True)
    pos = np.minimum(np.searchsorted(trip_keys, buy_keys), max(n - 1, 0))
    ok = trip_keys[pos] == buy_keys if n else np.zeros(0, dtype=bool)
    bought = -np.ones(n, dtype=np.int64)
    bought[pos[ok]] = product[b[buy_first[ok]]]
    match = viewed == bought[:, None]
    chosen = np.where(match.any(axis=1) & (bought >= 0), match.argmax(axis=1), -1)

    return Trips(shopper=shopper, category=category, day=day, viewed=viewed, chosen=chosen,
                 last_bought=_last_bought_before(shopper, category, bought, n_cat),
                 brand_share=_brand_share_before(shopper, viewed, bought, cat))


def _last_bought_before(shopper: np.ndarray, category: np.ndarray, bought: np.ndarray, n_categories: int
                        ) -> np.ndarray:
    """For each trip (rows in time order), the product bought on the latest earlier trip in that category."""
    n = len(shopper)
    group = shopper * n_categories + category
    buy_trip = np.flatnonzero(bought >= 0)
    # each purchase gets a stamp group * n + trip, so sorted stamps run by (shopper, category), then time
    stamp = group[buy_trip] * n + buy_trip
    order = np.argsort(stamp)
    stamps = stamp[order]
    idx = np.searchsorted(stamps, group * n + np.arange(n), side="left") - 1  # latest purchase stamp before
    ok = idx >= 0
    ok[ok] = stamps[idx[ok]] // n == group[ok]  # ...and it belongs to the same shopper and category
    out = -np.ones(n, dtype=np.int64)
    out[ok] = bought[buy_trip[order[idx[ok]]]]
    return out


def _brand_share_before(shopper: np.ndarray, viewed: np.ndarray, bought: np.ndarray, cat: Catalogue) -> np.ndarray:
    """For each viewed product: the share of the shopper's earlier purchases in the department's other
    categories that were that product's brand (purchases on earlier trips only)."""
    n = len(shopper)
    buy_trip = np.flatnonzero(bought >= 0)
    buy_product = bought[buy_trip]
    t_idx, s_idx = np.nonzero(viewed >= 0)  # every (trip, viewed product)
    prod = viewed[t_idx, s_idx]

    def count_before(key: np.ndarray) -> np.ndarray:
        """Purchases by the same shopper with the same key (brand, department, ...) on earlier trips."""
        k = int(key.max()) + 1
        stamps = np.sort((shopper[buy_trip] * k + key[buy_product]) * n + buy_trip)
        group = shopper[t_idx] * k + key[prod]
        return (np.searchsorted(stamps, group * n + t_idx, side="left")
                - np.searchsorted(stamps, group * n, side="left")).astype(np.float32)

    own_brand = count_before(cat.brand) - count_before(cat.cat_brand)  # this brand, other categories
    others = count_before(cat.department) - count_before(cat.category)  # anything, other categories
    out = np.zeros(viewed.shape, dtype=np.float32)
    out[t_idx, s_idx] = own_brand / (others + 1.0)
    return out
