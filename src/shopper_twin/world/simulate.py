"""Run the fake shoppers day by day and record what they view, cart and buy.

Every step is vectorised over all shoppers visiting that day, so a 100k-shopper
world runs in minutes. Randomness is seeded per day.

For what-if checks (milestone 4 onwards) the simulator can also switch, from a chosen day, to
"common random numbers": every random number is keyed by what it is for (day, shopper, category,
product), so a run with a price change differs from the unchanged run only where the change
really mattered. Days before that point use the original per-day random streams, so the public
data is reproduced exactly.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Iterator

import numpy as np

from .. import keyed_random as kr
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


@dataclass
class SimState:
    """Everything that carries over from one day to the next."""

    day: int  # the next day to simulate
    taste: np.ndarray
    staple_uses: np.ndarray
    disc_weight: np.ndarray
    visit_mult: np.ndarray
    last_buy: np.ndarray
    last_product: np.ndarray
    reverts: dict = field(default_factory=dict)
    next_session: int = 0

    def copy(self) -> "SimState":
        return copy.deepcopy(self)


class _SequentialNoise:
    """The original per-day random stream (draw order matters)."""

    def __init__(self, rng: np.random.Generator):
        self.rng = rng

    def visit(self, n):
        return self.rng.random(n)

    def due(self, u, n_categories):
        return self.rng.random((len(u), n_categories))

    def browse_count(self, u, lam):
        return self.rng.poisson(lam)

    def browse(self, u, n_categories):
        return self.rng.gumbel(size=(len(u), n_categories))

    def view(self, u, p):
        return self.rng.gumbel(size=p.shape)

    def choice(self, u, viewed):
        return self.rng.gumbel(size=viewed.shape)

    def outside(self, u, c):
        return self.rng.gumbel(size=len(u))

    def abandon(self, u, c):
        return self.rng.random(len(u))

    def quantity(self, u, c, lam):
        return self.rng.poisson(lam)

    def lost(self, u, c):
        return self.rng.random(len(u))

    def hour(self, u):
        return self.rng.normal(0, 1.2, len(u))

    def jitter(self, u, seq):
        return self.rng.integers(0, 25, len(u))

    def drift(self, shape, week):
        return self.rng.normal(0, 1.0, shape)


class _KeyedNoise:
    """Common random numbers: each draw depends only on its keys."""

    def __init__(self, seed: int, replicate: int, day: int):
        self.k = (seed, replicate, day)

    def visit(self, n):
        return kr.keyed_uniform(*self.k, kr.VISIT, np.arange(n))

    def due(self, u, n_categories):
        return kr.keyed_uniform(*self.k, kr.DUE, u[:, None], np.arange(n_categories)[None, :])

    def browse_count(self, u, lam):
        return kr.keyed_poisson(lam, *self.k, kr.BROWSE_COUNT, u)

    def browse(self, u, n_categories):
        return kr.keyed_gumbel(*self.k, kr.BROWSE, u[:, None], np.arange(n_categories)[None, :])

    def view(self, u, p):
        return kr.keyed_gumbel(*self.k, kr.VIEW, u[:, None], p)

    def choice(self, u, viewed):
        return kr.keyed_gumbel(*self.k, kr.CHOICE, u[:, None], viewed)

    def outside(self, u, c):
        return kr.keyed_gumbel(*self.k, kr.OUTSIDE, u, c)

    def abandon(self, u, c):
        return kr.keyed_uniform(*self.k, kr.ABANDON, u, c)

    def quantity(self, u, c, lam):
        return kr.keyed_poisson(lam, *self.k, kr.QUANTITY, u, c)

    def lost(self, u, c):
        return kr.keyed_uniform(*self.k, kr.LOST, u, c)

    def hour(self, u):
        return 1.2 * kr.keyed_normal(*self.k, kr.HOUR, u)

    def jitter(self, u, seq):
        return np.floor(25 * kr.keyed_uniform(*self.k, kr.TIMESTAMP, u, seq)).astype(np.int64)

    def drift(self, shape, week):
        n, d = shape
        return kr.keyed_normal(self.k[0], self.k[1], week, kr.DRIFT, np.arange(n)[:, None], np.arange(d)[None, :])


class _Scenario:
    """Prices and availability for one run (the planned schedule unless a what-if overrides them)."""

    def __init__(self, prices, unavailable: np.ndarray | None):
        # `prices` needs .price and .discount arrays [products, weeks]; any object with those works,
        # so what-if prices can be passed without being re-rounded
        self.price = np.asarray(prices.price)
        self.promo = np.asarray(prices.discount) > 0
        self.unavailable = unavailable


class Simulator:
    def __init__(self, cfg: WorldConfig, catalog: Catalog, population: Population, prices: PriceSchedule):
        self.cfg = cfg
        self.cat = catalog
        self.pop = population
        self.prices = prices
        self.start_taste: np.ndarray | None = None  # taste on day 0 (after warm-up drift)
        self.final_taste: np.ndarray | None = None

    # ---- state
    def initial_state(self) -> SimState:
        cfg, pop = self.cfg, self.pop
        init_rng = np.random.default_rng([cfg.seed, 104729])
        # start each staple part-way through its cycle; the warm-up then settles things into a steady state
        typical_gap = pop.staple_cycle + 7 / pop.visits_per_week[:, None]
        last_buy = (-cfg.warmup_days - init_rng.random(pop.staple_uses.shape) * typical_gap).astype(np.float32)
        return SimState(day=-cfg.warmup_days, taste=pop.taste.copy(), staple_uses=pop.staple_uses.copy(),
                        disc_weight=pop.disc_weight.copy(), visit_mult=np.ones(pop.n, dtype=np.float32),
                        last_buy=last_buy, last_product=-np.ones(pop.staple_uses.shape, dtype=np.int64))

    def state_at(self, day: int) -> SimState:
        """The state at the start of `day`, reached with the original random streams (as in the public data)."""
        state = self.initial_state()
        for _ in self.run(state=state, stop_day=day):
            pass
        return state

    # ---- simulation
    def run(self, prices=None, unavailable: np.ndarray | None = None, state: SimState | None = None,
            stop_day: int | None = None, crn_from_day: int | None = None, replicate: int = 0) -> Iterator[DayLog]:
        """Yield one DayLog per simulated day.

        prices:        what-if prices (anything with .price and .discount [products, weeks]); default = planned
        unavailable:   [products, weeks] bool, products that cannot be viewed or bought (stock-outs)
        state:         start from this state (see `state_at`); it is updated in place
        stop_day:      stop before this day (default: the end of the run)
        crn_from_day:  from this day on, use common random numbers (keyed draws)
        replicate:     which set of common random numbers (0, 1, 2, ...) for repeated what-if runs
        """
        cfg, cat, pop, r = self.cfg, self.cat, self.pop, self.cfg.rules
        scen = _Scenario(prices or self.prices, unavailable)
        st = state if state is not None else self.initial_state()
        stop = cfg.n_days if stop_day is None else stop_day

        baby_staples = np.flatnonzero((cat.cat_requires == "baby") & cat.cat_staple)
        baby_disc = np.flatnonzero((cat.cat_requires == "baby") & ~cat.cat_staple)
        home_cats = np.flatnonzero((cat.cat_dept == spec.DEPARTMENTS.index("Home & Garden")) & ~cat.cat_staple)

        start = np.datetime64(cfg.start_date)
        start_dow = int((start - np.datetime64("1970-01-05")).astype(int) % 7)  # 0 = Monday
        start_doy = int((start - start.astype("datetime64[Y]")).astype(int))
        weekday_mult = np.array(r.weekday_visit_mult)

        # warm-up days (negative) are simulated but not recorded, so day 0 already has habits and history
        while st.day < stop:
            day = st.day
            if crn_from_day is not None and day >= crn_from_day:
                noise = _KeyedNoise(cfg.seed, replicate, day)
            else:
                noise = _SequentialNoise(np.random.default_rng([cfg.seed, 7919, day + cfg.warmup_days]))
            doy = (start_doy + day) % 365
            week = max(day, 0) // 7
            price_today = scen.price[:, week]
            promo_today = scen.promo[:, week]
            avail_today = None if scen.unavailable is None else ~scen.unavailable[:, week]

            # ---- life events that happen today
            for u, kind in zip(pop.event_shopper[pop.event_day == day], pop.event_type[pop.event_day == day]):
                if kind == "new_baby":
                    st.staple_uses[u, baby_staples] = True
                    st.last_buy[u, baby_staples] = day - pop.staple_cycle[u, baby_staples]  # due now
                    st.disc_weight[u, baby_disc] = 3.0
                    st.visit_mult[u] *= 1.15
                else:
                    st.disc_weight[u, home_cats] *= 6.0
                    st.visit_mult[u] *= 1.3
                    st.reverts.setdefault(day + MOVE_BOOST_DAYS, []).append(u)
            for u in st.reverts.pop(day, []):
                st.disc_weight[u, home_cats] /= 6.0
                st.visit_mult[u] /= 1.3

            # ---- who visits today
            season_visits = 1 + 0.1 * np.cos(2 * np.pi * (doy - 355) / 365)
            p_visit = pop.visits_per_week / 7 * st.visit_mult * weekday_mult[(start_dow + day) % 7] * season_visits
            visitors = np.flatnonzero(noise.visit(pop.n) < np.minimum(p_visit, 0.95))

            # ---- what's on each visitor's list: due staples + a few categories to browse
            n_vis = len(visitors)
            days_since = day - st.last_buy[visitors]
            due = (st.staple_uses[visitors] & (days_since >= pop.staple_cycle[visitors])
                   & (noise.due(visitors, cat.n_categories) < r.staple_due_visit_prob))
            season = 1 + cat.cat_amp * np.cos(2 * np.pi * (doy - cat.cat_peak) / 365)
            w = st.disc_weight[visitors] * season[None, :].astype(np.float32)
            k = noise.browse_count(visitors, pop.discretionary_per_visit[visitors])
            with np.errstate(divide="ignore"):
                keys = np.where(w > 0, np.log(w) + noise.browse(visitors, cat.n_categories), -np.inf)
            ranks = np.empty_like(keys, dtype=np.int64)
            np.put_along_axis(ranks, np.argsort(-keys, axis=1),
                              np.broadcast_to(np.arange(cat.n_categories), keys.shape), axis=1)
            browse = (ranks < k[:, None]) & (w > 0)
            vi, ci = np.nonzero(due | browse)  # visit-major order

            parts = [self._shop(noise, visitors[vi[s:s + cfg.chunk_pairs]], ci[s:s + cfg.chunk_pairs],
                                vi[s:s + cfg.chunk_pairs], st.taste, st.last_product, price_today, promo_today,
                                avail_today)
                     for s in range(0, len(vi), cfg.chunk_pairs)]
            if parts:
                ev = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
            else:
                ev = {key: np.zeros(0, dtype=np.int64) for key in
                      ("visit", "product", "etype", "qty", "pair_u", "pair_c", "pair_prod")}
                ev["pair_buy"] = np.zeros(0, dtype=bool)

            # ---- update memory: last product bought per category, last staple purchase day
            bu, bc, bp = ev["pair_u"][ev["pair_buy"]], ev["pair_c"][ev["pair_buy"]], ev["pair_prod"][ev["pair_buy"]]
            st.last_product[bu, bc] = bp
            staple_bought = cat.cat_staple[bc]
            st.last_buy[bu[staple_bought], bc[staple_bought]] = day
            # a due staple they walked away from is sometimes bought at a competitor: a lost sale
            miss = cat.cat_staple[ev["pair_c"]] & ~ev["pair_buy"]
            lost = miss & (noise.lost(ev["pair_u"], ev["pair_c"]) < r.lost_to_competitor_prob)
            st.last_buy[ev["pair_u"][lost], ev["pair_c"][lost]] = day

            # ---- sessions and timestamps
            active = np.unique(vi)  # visits that browsed at least one category
            sess_of_visit = -np.ones(n_vis, dtype=np.int64)
            sess_of_visit[active] = st.next_session + np.arange(len(active))
            hour = np.clip(pop.shop_hour[visitors[active]] + noise.hour(visitors[active]), 7, 21.5)
            s_start = day * 86400 + (hour * 3600).astype(np.int64)
            start_of_visit = np.zeros(n_vis, dtype=np.int64)
            start_of_visit[active] = s_start
            ev_visit = ev["visit"]
            seq = np.arange(len(ev_visit)) - np.searchsorted(ev_visit, ev_visit, side="left")
            ts = start_of_visit[ev_visit] + seq * 35 + noise.jitter(visitors[ev_visit], seq)

            # ---- tastes drift a little each week
            if day % 7 == 6:
                st.taste += (cfg.drift_sd * noise.drift(st.taste.shape, day // 7)).astype(np.float32)
            st.day = day + 1
            if day < 0:
                continue
            if day == 0:
                self.start_taste = st.taste.copy()
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
                s_session_id=st.next_session + np.arange(len(active)),
                s_shopper_id=visitors[active].astype(np.int32),
                s_start_ts=s_start,
            )
            st.next_session += len(active)

        self.final_taste = st.taste

    def _shop(self, noise, u, c, visit, taste, last_product, price_today, promo_today, avail_today=None) -> dict:
        """One (shopper, category) trip each: view a few products, maybe buy one."""
        cat, pop, r = self.cat, self.pop, self.cfg.rules
        n = len(u)
        rows = np.arange(n)
        staple = cat.cat_staple[c]

        prods = cat.cat_products[c]
        valid = prods >= 0
        if avail_today is not None:
            valid &= avail_today[np.maximum(prods, 0)]
        p = np.where(valid, prods, 0)
        taste_fit = np.einsum("nd,npd->np", taste[u], cat.style[p]) * r.taste_scale
        promo = promo_today[p]
        habit = (p == last_product[u, c][:, None]) & valid

        # which products get viewed: relevant, popular, promoted, or bought before
        view_score = (r.view_taste_weight * taste_fit + cat.popularity[p]
                      + r.view_promo_boost * promo + r.view_habit_boost * habit)
        keys = np.where(valid, view_score + noise.view(u, prods), -np.inf)
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
        vu = np.take_along_axis(utility, order, axis=1) + noise.choice(u, viewed)
        vu[~viewed_ok] = -np.inf
        outside = np.where(staple, r.outside_staple, r.outside_discretionary) + noise.outside(u, c)

        best = np.argmax(vu, axis=1)
        buy = vu[rows, best] > outside
        chosen = viewed[rows, best]
        vu[rows, best] = -np.inf
        second = np.argmax(vu, axis=1)
        abandon = (vu[rows, second] > outside) & (noise.abandon(u, c) < r.abandon_cart_prob)
        qty = np.where(staple, 1 + noise.quantity(u, c, 0.15 * pop.household_size[u]), 1)

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
