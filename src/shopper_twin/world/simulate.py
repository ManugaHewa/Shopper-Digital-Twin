"""Run the fake shoppers day by day and record what they view, cart and buy.

Every step is vectorised over all shoppers visiting that day, so a 100k-shopper
world runs in minutes. Randomness is seeded per day, which keeps two runs with
different prices as comparable as possible (used for what-if checks later).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np

from . import spec
from .catalog import Catalog, PriceSchedule
from .config import WorldConfig
from .population import Population

VIEW, CART, PURCHASE = 0, 1, 2
EVENT_TYPES = ("view", "cart", "purchase")
MOVE_BOOST_DAYS = 60


@dataclass
class DayLog:
    day: int
    # one entry per event
    session_id: np.ndarray
    shopper_id: np.ndarray
    ts: np.ndarray  # seconds since the start date
    product_id: np.ndarray
    event_type: np.ndarray  # VIEW / CART / PURCHASE
    quantity: np.ndarray
    price: np.ndarray
    on_promo: np.ndarray
    # one entry per session
    s_session_id: np.ndarray
    s_shopper_id: np.ndarray
    s_start_ts: np.ndarray


class Simulator:
    def __init__(self, cfg: WorldConfig, catalog: Catalog, population: Population, prices: PriceSchedule):
        self.cfg = cfg
        self.cat = catalog
        self.pop = population
        self.prices = prices
        self.start_taste: np.ndarray | None = None  # taste on day 0 (after warm-up drift)
        self.final_taste: np.ndarray | None = None

    def run(self, prices: PriceSchedule | None = None) -> Iterator[DayLog]:
        """Yield one DayLog per simulated day. Pass `prices` to run a what-if price scenario."""
        cfg, cat, pop, r = self.cfg, self.cat, self.pop, self.cfg.rules
        prices = prices or self.prices
        price_table = prices.price
        promo_table = prices.discount > 0

        # mutable state (copied so the same Simulator can be run again)
        taste = pop.taste.copy()
        staple_uses = pop.staple_uses.copy()
        disc_weight = pop.disc_weight.copy()
        visit_mult = np.ones(pop.n, dtype=np.float32)
        init_rng = np.random.default_rng([cfg.seed, 104729])
        # start each staple part-way through its cycle; the warm-up then settles things into a steady state
        typical_gap = pop.staple_cycle + 7 / pop.visits_per_week[:, None]
        last_buy = (-cfg.warmup_days - init_rng.random(staple_uses.shape) * typical_gap).astype(np.float32)
        last_product = -np.ones(staple_uses.shape, dtype=np.int64)

        baby_staples = np.flatnonzero((cat.cat_requires == "baby") & cat.cat_staple)
        baby_disc = np.flatnonzero((cat.cat_requires == "baby") & ~cat.cat_staple)
        home_cats = np.flatnonzero((cat.cat_dept == spec.DEPARTMENTS.index("Home & Garden")) & ~cat.cat_staple)
        reverts: dict[int, list[int]] = {}

        start = np.datetime64(cfg.start_date)
        start_dow = int((start - np.datetime64("1970-01-05")).astype(int) % 7)  # 0 = Monday
        start_doy = int((start - start.astype("datetime64[Y]")).astype(int))
        weekday_mult = np.array(r.weekday_visit_mult)
        next_session = 0

        # warm-up days (negative) are simulated but not recorded, so day 0 already has habits and history
        for day in range(-cfg.warmup_days, cfg.n_days):
            rng = np.random.default_rng([cfg.seed, 7919, day + cfg.warmup_days])
            doy = (start_doy + day) % 365
            week = max(day, 0) // 7
            price_today = price_table[:, week]
            promo_today = promo_table[:, week]

            # ---- life events that happen today
            for u, kind in zip(pop.event_shopper[pop.event_day == day], pop.event_type[pop.event_day == day]):
                if kind == "new_baby":
                    staple_uses[u, baby_staples] = True
                    last_buy[u, baby_staples] = day - pop.staple_cycle[u, baby_staples]  # due now
                    disc_weight[u, baby_disc] = 3.0
                    visit_mult[u] *= 1.15
                else:
                    disc_weight[u, home_cats] *= 6.0
                    visit_mult[u] *= 1.3
                    reverts.setdefault(day + MOVE_BOOST_DAYS, []).append(u)
            for u in reverts.pop(day, []):
                disc_weight[u, home_cats] /= 6.0
                visit_mult[u] /= 1.3

            # ---- who visits today
            season_visits = 1 + 0.1 * np.cos(2 * np.pi * (doy - 355) / 365)
            p_visit = pop.visits_per_week / 7 * visit_mult * weekday_mult[(start_dow + day) % 7] * season_visits
            visitors = np.flatnonzero(rng.random(pop.n) < np.minimum(p_visit, 0.95))

            # ---- what's on each visitor's list: due staples + a few categories to browse
            n_vis = len(visitors)
            days_since = day - last_buy[visitors]
            due = (staple_uses[visitors] & (days_since >= pop.staple_cycle[visitors])
                   & (rng.random((n_vis, cat.n_categories)) < r.staple_due_visit_prob))
            season = 1 + cat.cat_amp * np.cos(2 * np.pi * (doy - cat.cat_peak) / 365)
            w = disc_weight[visitors] * season[None, :].astype(np.float32)
            k = rng.poisson(pop.discretionary_per_visit[visitors])
            with np.errstate(divide="ignore"):
                keys = np.where(w > 0, np.log(w) + rng.gumbel(size=w.shape), -np.inf)
            ranks = np.empty_like(keys, dtype=np.int64)
            np.put_along_axis(ranks, np.argsort(-keys, axis=1),
                              np.broadcast_to(np.arange(cat.n_categories), keys.shape), axis=1)
            browse = (ranks < k[:, None]) & (w > 0)
            vi, ci = np.nonzero(due | browse)  # visit-major order

            parts = [self._shop(rng, visitors[vi[s:s + cfg.chunk_pairs]], ci[s:s + cfg.chunk_pairs],
                                vi[s:s + cfg.chunk_pairs], taste, last_product, price_today, promo_today)
                     for s in range(0, len(vi), cfg.chunk_pairs)]
            if parts:
                ev = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
            else:
                ev = {key: np.zeros(0, dtype=np.int64) for key in
                      ("visit", "product", "etype", "qty", "pair_u", "pair_c", "pair_prod")}
                ev["pair_buy"] = np.zeros(0, dtype=bool)

            # ---- update memory: last product bought per category, last staple purchase day
            bu, bc, bp = ev["pair_u"][ev["pair_buy"]], ev["pair_c"][ev["pair_buy"]], ev["pair_prod"][ev["pair_buy"]]
            last_product[bu, bc] = bp
            staple_bought = cat.cat_staple[bc]
            last_buy[bu[staple_bought], bc[staple_bought]] = day
            # a due staple they walked away from is sometimes bought at a competitor: a lost sale
            miss = cat.cat_staple[ev["pair_c"]] & ~ev["pair_buy"]
            lost = miss & (rng.random(len(miss)) < r.lost_to_competitor_prob)
            last_buy[ev["pair_u"][lost], ev["pair_c"][lost]] = day

            # ---- sessions and timestamps
            active = np.unique(vi)  # visits that browsed at least one category
            sess_of_visit = -np.ones(n_vis, dtype=np.int64)
            sess_of_visit[active] = next_session + np.arange(len(active))
            hour = np.clip(pop.shop_hour[visitors[active]] + rng.normal(0, 1.2, len(active)), 7, 21.5)
            s_start = day * 86400 + (hour * 3600).astype(np.int64)
            start_of_visit = np.zeros(n_vis, dtype=np.int64)
            start_of_visit[active] = s_start
            ev_visit = ev["visit"]
            seq = np.arange(len(ev_visit)) - np.searchsorted(ev_visit, ev_visit, side="left")
            ts = start_of_visit[ev_visit] + seq * 35 + rng.integers(0, 25, len(ev_visit))

            # ---- tastes drift a little each week
            if day % 7 == 6:
                taste += rng.normal(0, cfg.drift_sd, taste.shape).astype(np.float32)
            if day < 0:
                continue
            if day == 0:
                self.start_taste = taste.copy()
            yield DayLog(
                day=day,
                session_id=sess_of_visit[ev_visit],
                shopper_id=visitors[ev_visit].astype(np.int32),
                ts=ts,
                product_id=ev["product"].astype(np.int32),
                event_type=ev["etype"].astype(np.int8),
                quantity=ev["qty"].astype(np.int8),
                price=price_today[ev["product"]],
                on_promo=promo_today[ev["product"]],
                s_session_id=next_session + np.arange(len(active)),
                s_shopper_id=visitors[active].astype(np.int32),
                s_start_ts=s_start,
            )
            next_session += len(active)

        self.final_taste = taste

    def _shop(self, rng, u, c, visit, taste, last_product, price_today, promo_today) -> dict:
        """One (shopper, category) trip each: view a few products, maybe buy one."""
        cat, pop, r = self.cat, self.pop, self.cfg.rules
        n = len(u)
        rows = np.arange(n)
        staple = cat.cat_staple[c]

        prods = cat.cat_products[c]
        valid = prods >= 0
        p = np.where(valid, prods, 0)
        taste_fit = np.einsum("nd,npd->np", taste[u], cat.style[p]) * r.taste_scale
        promo = promo_today[p]
        habit = (p == last_product[u, c][:, None]) & valid

        # which products get viewed: relevant, popular, promoted, or bought before
        view_score = (r.view_taste_weight * taste_fit + cat.popularity[p]
                      + r.view_promo_boost * promo + r.view_habit_boost * habit)
        keys = np.where(valid, view_score + rng.gumbel(size=p.shape), -np.inf)
        k_max = min(max(r.views_staple, r.views_discretionary), p.shape[1])
        order = np.argsort(-keys, axis=1)[:, :k_max]
        n_views = np.where(staple, r.views_staple, r.views_discretionary)
        viewed = np.take_along_axis(p, order, axis=1)
        viewed_ok = np.take_along_axis(valid, order, axis=1) & (np.arange(k_max)[None, :] < n_views[:, None])

        # the true utility of each viewed product
        is_fav = cat.brand[p] == pop.fav_brand[u, cat.cat_dept[c]][:, None]
        utility = (taste_fit
                   + pop.quality_weight[u, None] * 2 * (cat.quality[p] - 0.5)
                   + pop.store_brand_affinity[u, None] * cat.is_store[p]
                   + pop.brand_loyalty[u, None] * is_fav
                   + pop.habit_strength[u, None] * habit
                   - pop.price_sensitivity[u, None] * np.log(price_today[p] / cat.cat_median[c][:, None])
                   + r.promo_salience * promo)
        vu = np.take_along_axis(utility, order, axis=1) + rng.gumbel(size=viewed.shape)
        vu[~viewed_ok] = -np.inf
        outside = np.where(staple, r.outside_staple, r.outside_discretionary) + rng.gumbel(size=n)

        best = np.argmax(vu, axis=1)
        buy = vu[rows, best] > outside
        chosen = viewed[rows, best]
        vu[rows, best] = -np.inf
        second = np.argmax(vu, axis=1)
        abandon = (vu[rows, second] > outside) & (rng.random(n) < r.abandon_cart_prob)
        qty = np.where(staple, 1 + rng.poisson(0.15 * pop.household_size[u]), 1)

        # event grid per trip: k views, abandoned cart, cart, purchase
        grid_prod = np.concatenate([viewed, viewed[rows, second][:, None], chosen[:, None], chosen[:, None]], 1)
        grid_ok = np.concatenate([viewed_ok, abandon[:, None], buy[:, None], buy[:, None]], 1)
        grid_type = np.array([VIEW] * k_max + [CART, CART, PURCHASE])
        grid_qty = np.concatenate([np.zeros((n, k_max), int), np.ones((n, 1), int), qty[:, None], qty[:, None]], 1)
        rr, cc = np.nonzero(grid_ok)
        return {
            "visit": visit[rr],
            "product": grid_prod[rr, cc],
            "etype": grid_type[cc],
            "qty": grid_qty[rr, cc],
            "pair_u": u,
            "pair_c": c,
            "pair_buy": buy,
            "pair_prod": chosen,
        }
