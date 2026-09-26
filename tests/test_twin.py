"""Checks the shopper twin: the trip table, the maths inside the model, and the fitted twin on a small world."""

import copy
import dataclasses

import numpy as np
import pandas as pd
import pytest
import torch

from shopper_twin.baselines import Random
from shopper_twin.eval import evaluate, make_split
from shopper_twin.twin import ShopperTwin, build_trips
from shopper_twin.twin.choice import ordered_view_loglik
from shopper_twin.twin.trips import Catalogue
from shopper_twin.twin.incidence import fit_trip_rates, snapshot
from shopper_twin.twin.twin import choice_given_trip, inclusion_probabilities

FAST = {"epochs": 2, "dim": 8}


# ---------- the trip table, on hand-made events


def _catalogue(n_products=6):
    """Products 0-2 in category 0, 3-5 in category 1, all in one department; brands A, B, A / A, B, B."""
    cat = np.array([0, 0, 0, 1, 1, 1])[:n_products]
    brand = np.array([0, 1, 0, 0, 1, 1])[:n_products]
    ones = np.ones((n_products, 2), np.float32)
    return Catalogue(category=cat, department=np.zeros(n_products, np.int64), brand=brand,
                     is_store=np.zeros(n_products, np.float32), price=ones, promo=0 * ones, regular=ones,
                     median=np.ones(2, np.float32), members=np.array([[0, 1, 2], [3, 4, 5]]),
                     cat_brand=cat * 2 + brand)


def _events(rows):
    """rows: (session, shopper, day, event_type, product) in the order they happened."""
    ev = pd.DataFrame(rows, columns=["session_id", "shopper_id", "day", "event_type", "product_id"])
    ev["timestamp"] = np.arange(len(ev))
    return ev


EVENTS = _events([
    (0, 0, 0, "view", 2), (0, 0, 0, "view", 0), (0, 0, 0, "purchase", 0),   # trip 0: cat 0, buys 0
    (0, 0, 0, "view", 4), (0, 0, 0, "view", 3),                             # trip 1: cat 1, buys nothing
    (1, 0, 3, "view", 1), (1, 0, 3, "view", 0), (1, 0, 3, "purchase", 1),   # trip 2: cat 0, buys 1
    (1, 0, 3, "view", 5), (1, 0, 3, "purchase", 5),                         # trip 3: cat 1, buys 5 (brand B)
    (2, 0, 9, "view", 0), (2, 0, 9, "view", 1),                             # trip 4: cat 0, buys nothing
    (3, 1, 9, "view", 3), (3, 1, 9, "purchase", 3),                         # trip 5: another shopper
])


def test_trips_keep_view_order_and_the_chosen_slot():
    t = build_trips(EVENTS, _catalogue())
    assert len(t) == 6
    np.testing.assert_array_equal(t.viewed[0, :2], [2, 0])
    np.testing.assert_array_equal(t.viewed[1, :2], [4, 3])
    assert (t.viewed[:, 2:] == -1).all()
    np.testing.assert_array_equal(t.chosen, [1, -1, 0, 0, -1, 0])


def test_habit_only_uses_earlier_trips():
    t = build_trips(EVENTS, _catalogue())
    # first trip in each (shopper, category) has no known habit; later ones see the previous purchase
    np.testing.assert_array_equal(t.last_bought, [-1, -1, 0, -1, 1, -1])


def test_view_order_comes_from_timestamps_when_rows_are_shuffled():
    shuffled = EVENTS.sample(frac=1, random_state=3)
    a, b = build_trips(EVENTS, _catalogue()), build_trips(shuffled, _catalogue())
    np.testing.assert_array_equal(a.viewed, b.viewed)
    np.testing.assert_array_equal(a.chosen, b.chosen)


def test_brand_loyalty_counts_only_other_categories_and_earlier_trips():
    t = build_trips(EVENTS, _catalogue())
    # trip 2 (cat 0): before it, the only other-category purchase is none, so every share is 0
    assert (t.brand_share[2] == 0).all()
    # trip 4 (cat 0): before it, one other-category purchase (product 5, brand B): share = 1 / (1 + 1)
    np.testing.assert_allclose(t.brand_share[4, :2], [0.0, 0.5])  # product 0 is brand A, product 1 brand B
    # trip 3 (cat 1, same session as trip 2): sees the cat-0 purchases of trips 0 and 2 (brands A, B)
    np.testing.assert_allclose(t.brand_share[3, 0], 1 / 3)  # product 5 is brand B: 1 of 2, smoothed


# ---------- the maths


def test_inclusion_probabilities_add_up_to_k_and_match_simulation():
    rng = np.random.default_rng(0)
    scores = rng.normal(0, 1.5, (1, 12))
    incl = inclusion_probabilities(scores, 4)
    np.testing.assert_allclose(incl.sum(), 4, rtol=1e-6)
    gumbel = -np.log(-np.log(rng.random((40_000, 12))))
    top = np.argsort(-(scores + gumbel), axis=1)[:, :4]
    freq = np.bincount(top.ravel(), minlength=12) / 40_000
    np.testing.assert_allclose(incl[0], freq, atol=0.03)


def test_inclusion_probabilities_when_almost_everything_is_looked_at():
    rng = np.random.default_rng(1)
    scores = rng.normal(0, 3, size=(50, 9))
    np.testing.assert_array_equal(inclusion_probabilities(scores[:, :8], 8), 1.0)  # k = number of products
    np.testing.assert_allclose(inclusion_probabilities(scores, 8).sum(axis=1), 8, atol=1e-6)


def test_choice_given_trip_matches_simulated_trips():
    """Draw the k viewed products one by one (probability ~ exp(view score)), pick the best of them or nothing
    with logit noise, many times: the formula should give the same chances, and the same reaction to a price rise."""
    rng = np.random.default_rng(0)
    n, m, k, sims = 6, 40, 4, 60_000  # 40 products, 4 looked at: like an everyday category
    v, s, outside = rng.normal(0, 0.8, (n, m)), rng.normal(0, 1, (n, m)), rng.normal(-0.5, 0.5, n)

    def simulate(v):
        viewed = np.argsort(-(s[:, None, :] + rng.gumbel(size=(n, sims, m))), axis=2)[:, :, :k]
        util = np.take_along_axis(np.broadcast_to(v[:, None, :], (n, sims, m)), viewed, 2)
        util = util + rng.gumbel(size=util.shape)
        nothing = outside[:, None] + rng.gumbel(size=(n, sims))
        pick = np.where(util.max(2) > nothing, np.take_along_axis(viewed, util.argmax(2)[..., None], 2)[..., 0], -1)
        return np.stack([np.bincount(pick[i][pick[i] >= 0], minlength=m) / sims for i in range(n)])

    incl = inclusion_probabilities(s, k)
    totals = []
    for shift in (0.0, -0.3):  # the second is a price rise on every product
        formula, simulated = choice_given_trip(v + shift, outside, incl, k), simulate(v + shift)
        np.testing.assert_allclose(formula, simulated, atol=0.01)  # each product
        np.testing.assert_allclose(formula.sum(axis=1), simulated.sum(axis=1), atol=0.06)  # buying anything
        totals.append((formula.sum(), simulated.sum()))
    (f0, s0), (f1, s1) = totals
    assert f1 / f0 - 1 == pytest.approx(s1 / s0 - 1, abs=0.005)  # the reaction to the price rise


def test_choice_given_trip_is_a_valid_probability():
    rng = np.random.default_rng(1)
    v, s = rng.normal(0, 2, (5, 10)), rng.normal(0, 1, (5, 10))
    incl = inclusion_probabilities(s, 4)
    q = choice_given_trip(v, rng.normal(0, 1, 5), incl, 4)
    assert (q >= 0).all() and (q <= incl + 1e-12).all()
    assert (q.sum(axis=1) < 1).all()


def test_ordered_view_loglik_matches_step_by_step_softmax():
    scores = torch.tensor([[0.3, -1.0, 2.0, 0.5]])
    positions = torch.tensor([[2, 0, -1]])
    expected = (torch.log_softmax(scores, 1)[0, 2]
                + torch.log_softmax(scores[:, [0, 1, 3]], 1)[0, 0])
    torch.testing.assert_close(ordered_view_loglik(scores, positions)[0], expected)


# ---------- the fitted twin on a small generated world


@pytest.fixture(scope="module")
def world(tiny_world):
    return tiny_world


@pytest.fixture(scope="module")
def events(world):
    return world.events(extra_columns=("session_id", "timestamp", "quantity"))


@pytest.fixture(scope="module")
def split(world, events):
    return make_split(events, world.n_shoppers, world.n_products, world.products.category_id.to_numpy(),
                      cutoff_day=62, horizon=28)


@pytest.fixture(scope="module")
def twin(world, split):
    return ShopperTwin(world, **FAST).fit(split.train)


def test_scores_are_probabilities_and_repeatable(world, split, twin):
    users = split.task.users[:40]
    a = twin.score(users)
    assert a.shape == (40, world.n_products)
    assert np.isfinite(a).all() and (a >= 0).all() and (a <= 1).all()
    b = ShopperTwin(world, **FAST).fit(split.train).score(users)
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-9)


def test_each_row_belongs_to_its_own_shopper(split, twin):
    users = split.task.users[[5, 1, 9]]
    together = twin.score(users)
    for j, u in enumerate(users):
        np.testing.assert_allclose(together[j], twin.score(np.array([u]))[0], rtol=1e-6, atol=1e-12)


def test_calibration_matches_purchases_per_trip(world, split, twin):
    """Replaying the last 28 days before the cutoff, the calibrated twin predicts as many purchases as happened."""
    trips = build_trips(split.train.events, twin.cat)
    start = split.train.cutoff_day - 28
    replay = copy.copy(twin)  # shallow copy with the memory of day `start`
    replay._memory_at_cutoff(trips.subset(trips.day < start))
    window = trips.subset(trips.day >= start)
    predicted = actual = 0.0
    for c in np.unique(window.category)[:10]:
        sel = window.category == c
        users, row = np.unique(window.shopper[sel], return_inverse=True)
        weeks, col = np.unique(window.week[sel], return_inverse=True)
        _, outside, per_week = replay._trip_terms(users, c, twin.cat.price, twin.cat.promo, None, weeks)
        for j, (_, _, v, incl, k) in enumerate(per_week):
            q = choice_given_trip(v, outside, incl, k).sum(axis=1) * twin.purchase_scale[c]
            predicted += q[row[col == j]].sum()
        actual += window.bought[sel].sum()
    # exact unless a category's factor hit its 0.5-2 limit (or the +1 smoothing matters)
    assert predicted == pytest.approx(actual, rel=0.02)


def test_trip_rates_add_up_on_the_training_snapshots(world, split, twin):
    trips = build_trips(split.train.events, twin.cat)
    cut, horizon = split.train.cutoff_day - 28, 28
    x, y = snapshot(trips.subset(trips.day < cut + horizon), world.n_shoppers, twin.cat.n_categories,
                    twin.cat_department, cut, horizon)
    category = np.tile(np.arange(twin.cat.n_categories), world.n_shoppers)
    rates = twin.rate_model.rate(x, category)
    # the tiny world has one training snapshot, so every category's total comes out right (to within one trip,
    # because the correction is (observed + 1) / (predicted + 1) so that an empty category stays finite)
    observed = np.bincount(category, weights=y, minlength=twin.cat.n_categories)
    predicted = np.bincount(category, weights=rates, minlength=twin.cat.n_categories)
    assert np.abs(predicted - observed).max() < 1


def test_trip_rates_fit_when_nobody_shops_on_the_last_day_of_a_window(world, split, twin):
    trips = build_trips(split.train.events, twin.cat)
    cutoff = split.train.cutoff_day
    gap = trips.subset(trips.day != cutoff - 1)  # the training window [cutoff - 28, cutoff) ends on an empty day
    model = fit_trip_rates(gap, world.n_shoppers, twin.cat.n_categories, twin.cat_department, cutoff, 28, epochs=1)
    assert np.isfinite(model.log_correction.numpy()).all()


def test_twin_beats_random(split, twin):
    random_ndcg = evaluate(Random().fit(split.train), split, "all")["ndcg@10"]
    assert evaluate(twin.with_prices(False), split, "all")["ndcg@10"] > 5 * random_ndcg


def test_fit_refuses_events_from_the_prediction_window(world, split, events):
    leaked = dataclasses.replace(split.train, events=events)  # includes days after the cutoff
    with pytest.raises(ValueError, match="before day"):
        ShopperTwin(world, **FAST).fit(leaked)


def test_future_prices_do_not_change_fair_scores(split, twin):
    """The fair twin must not use the store's price plan for the weeks after the cutoff."""
    users = split.task.users[:30]
    fair = twin.with_prices(False).score(users)
    changed = twin.with_prices(False)
    changed.cat = dataclasses.replace(twin.cat)
    later = np.arange(twin.cat.n_weeks) > (split.train.cutoff_day - 1) // 7
    changed.cat.price = twin.cat.price.copy()
    changed.cat.price[:, later] *= 1.5
    changed.cat.promo = twin.cat.promo.copy()
    changed.cat.promo[:, later] = 1.0
    np.testing.assert_allclose(changed.score(users), fair, rtol=1e-6, atol=1e-12)
    # ...which the price-plan twin does react to
    planned = twin.with_prices(True)
    planned.cat = changed.cat
    assert not np.allclose(planned.score(users), twin.with_prices(True).score(users))


def test_save_and_load_give_the_same_scores(world, split, twin, tmp_path):
    path = twin.save(tmp_path / "twin.pt")
    loaded = ShopperTwin.load(path, world)
    assert loaded.cutoff_day == twin.cutoff_day == 62
    users = split.task.users[:25]
    for planned in (True, False):
        np.testing.assert_array_equal(loaded.with_prices(planned).score(users), twin.with_prices(planned).score(users))


def test_save_accepts_numpy_numbers(world, split, twin, tmp_path):
    odd = copy.copy(twin)
    odd.cutoff, odd.horizon, odd.n_shoppers, odd.dim = (np.int16(twin.cutoff_day), np.int64(twin.horizon),
                                                       np.int64(twin.n_shoppers), np.int64(twin.dim))
    loaded = ShopperTwin.load(odd.save(tmp_path / "odd.pt"), world)
    assert loaded.cutoff_day == twin.cutoff_day and type(loaded.cutoff_day) is int


def test_with_prices_does_not_change_the_original(twin):
    before = twin.planned_prices
    other = twin.with_prices(not before)
    assert twin.planned_prices == before and other.planned_prices != before
    assert other.name != twin.name


def test_higher_price_lowers_the_chance_of_buying(split, twin):
    users = split.task.users[:50]
    base = twin.purchase_probability(users)
    price = twin.cat.price.copy()
    product = int(np.argmax(base.sum(axis=0)))
    price[product] *= 1.3
    after = twin.purchase_probability(users, price=price, promo=twin.cat.promo)
    assert after[:, product].sum() < base[:, product].sum()
    # and the other products in its category pick up some of the lost sales
    others = twin.cat.members[twin.cat.category[product]]
    others = others[(others >= 0) & (others != product)]
    assert after[:, others].sum() > base[:, others].sum()


def test_harness_refuses_a_twin_trained_for_another_window(world, events, split, twin):
    for cutoff in (40, 61):  # a window it has seen the answers to, and one that starts a day before its own
        other = make_split(events, world.n_shoppers, world.n_products, world.products.category_id.to_numpy(),
                           cutoff_day=cutoff, horizon=28)
        with pytest.raises(ValueError, match="line up"):
            evaluate(twin, other, "all")
