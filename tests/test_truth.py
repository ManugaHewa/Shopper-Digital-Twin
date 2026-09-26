"""Checks the evaluation against the true store: the store can be re-run exactly, a change only moves what it
touches, and the simpler comparison methods and scores compute what they claim."""

import numpy as np
import pandas as pd
import pytest

from shopper_twin import keyed_random as kr
from shopper_twin.eval import make_split
from shopper_twin.eval.comparators import StoreElasticity, no_reaction, without_personal_traits
from shopper_twin.eval.recovery import calibration, rank_correlation, trait_recovery
from shopper_twin.eval.truth import Outcome, TrueStore, true_effect
from shopper_twin.sim import PriceChange, PricePlan, Scenario, Select
from shopper_twin.twin import ShopperTwin, build_trips
from shopper_twin.world.simulate import PURCHASE

FIRST, LAST = 62, 89  # a 4-week window at the end of the 90-day test world
ORDER = ["day", "shopper_id", "product_id", "quantity"]


@pytest.fixture(scope="module")
def store(tiny_world):
    return TrueStore(tiny_world.path)


@pytest.fixture(scope="module")
def products(tiny_world):
    return tiny_world.products.sort_values("product_id").reset_index(drop=True)


def _purchases(store, plan=None, replicate=0):
    return store.purchases(FIRST, LAST, plan, replicate=replicate).sort_values(ORDER).reset_index(drop=True)


# ---------- common random numbers


def test_keyed_random_numbers_depend_only_on_their_keys():
    a = kr.keyed_uniform(1, 2, np.arange(1000))
    np.testing.assert_array_equal(a, kr.keyed_uniform(1, 2, np.arange(1000)))
    assert not np.array_equal(a, kr.keyed_uniform(1, 3, np.arange(1000)))
    assert ((a > 0) & (a < 1)).all()
    many = kr.keyed_uniform(7, np.arange(200_000))
    assert many.mean() == pytest.approx(0.5, abs=0.005)
    assert kr.keyed_gumbel(7, np.arange(200_000)).mean() == pytest.approx(0.5772, abs=0.01)


# ---------- the true store


def test_true_store_replays_the_public_data(tiny_world, store):
    """Started from its state on day FIRST with the original random numbers, the rebuilt store sells exactly
    what the public data says it sold."""
    rows = []
    for log in store.sim.run(state=store.state_at(FIRST), stop_day=LAST + 1):
        buy = log.event_type == PURCHASE
        rows.append(pd.DataFrame({"shopper_id": log.shopper_id[buy].astype(np.int64),
                                  "product_id": log.product_id[buy].astype(np.int64),
                                  "quantity": log.quantity[buy].astype(np.int64), "day": log.day}))
    replay = pd.concat(rows).sort_values(ORDER).reset_index(drop=True)
    public = tiny_world.events(types=("purchase",), extra_columns=("quantity",))
    public = public[(public.day >= FIRST) & (public.day <= LAST)][["shopper_id", "product_id", "quantity", "day"]]
    public = public.astype(np.int64).sort_values(ORDER).reset_index(drop=True)
    pd.testing.assert_frame_equal(replay[public.columns].astype(np.int64), public)


def test_same_plan_same_purchases_and_replicates_differ(store):
    first = _purchases(store)
    pd.testing.assert_frame_equal(first, _purchases(store))
    other = _purchases(store, replicate=1)
    assert len(other) != len(first) or not other.equals(first)
    # but both are the same store: totals within a few percent
    assert other.quantity.sum() == pytest.approx(first.quantity.sum(), rel=0.05)


def test_no_change_gives_no_true_effect(tiny_world, store, products):
    plan = PricePlan.from_world(tiny_world)
    same = Scenario("nothing", [PriceChange(Select(category="Coffee"), percent=0)]).apply(plan, products)
    pd.testing.assert_frame_equal(_purchases(store), _purchases(store, same))


def test_a_coffee_price_change_only_moves_coffee(tiny_world, store, products):
    plan = PricePlan.from_world(tiny_world)
    dearer = Scenario("coffee", [PriceChange(Select(category="Coffee"), percent=30)]).apply(plan, products)
    base, new = _purchases(store), _purchases(store, dearer)
    coffee = set(products.product_id[products.category == "Coffee"])
    is_coffee = base.product_id.isin(coffee)
    pd.testing.assert_frame_equal(base[~is_coffee].reset_index(drop=True),
                                  new[~new.product_id.isin(coffee)].reset_index(drop=True))
    assert new[new.product_id.isin(coffee)].quantity.sum() < base[is_coffee].quantity.sum()


def test_true_effect_averages_the_runs():
    def outcome(units):
        units = np.asarray(units, dtype=float)
        return Outcome(units=units, revenue=2 * units, shopper_units=pd.Series(dtype=float))

    base = [outcome([10, 10, 5]), outcome([12, 8, 5])]
    new = [outcome([8, 11, 5]), outcome([9, 10, 5])]
    effect = true_effect(base, new, target=np.array([True, False, False]), in_categories=np.array([True, True, False]))
    assert effect["target_units"]["pct"] == pytest.approx(100 * (8.5 / 11 - 1))
    assert effect["target_units"]["replicates"] == pytest.approx([-20.0, -25.0])
    assert effect["target_units"]["se"] == pytest.approx(2.5)
    assert effect["category_revenue"]["pct"] == pytest.approx(100 * (19 / 20 - 1))


# ---------- the simpler methods


def test_store_elasticity_recovers_a_known_elasticity():
    """Weekly sales made up with a switching elasticity of -1.5 and a promotion lift of 0.6 (log)."""
    rng = np.random.default_rng(0)
    n_products, n_weeks = 300, 30
    category = np.repeat(np.arange(30), 10)
    regular = rng.uniform(2, 10, size=(n_products, 1)) * np.ones((1, n_weeks))
    discount = np.where(rng.random((n_products, n_weeks)) < 0.1, 0.2, 0.0)
    price = regular * np.exp(rng.normal(0, 0.1, size=(n_products, n_weeks))) * (1 - discount)
    plan = PricePlan(price=price, discount=discount, regular=regular, available=np.ones_like(price, dtype=bool))
    level = rng.normal(3.5, 0.5, size=(n_products, 1)) + rng.normal(0, 0.2, size=(30, n_weeks))[category]
    units = rng.poisson(np.exp(level - 1.5 * np.log(price) + 0.6 * plan.promo))
    rows = np.nonzero(units)
    purchases = pd.DataFrame({"product_id": rows[0], "day": rows[1] * 7, "quantity": units[rows]})
    model = StoreElasticity().fit(purchases, plan, category, first_week=n_weeks)
    assert model.elasticity == pytest.approx(-1.5, abs=0.1)
    assert model.promo_lift == pytest.approx(0.6, abs=0.1)


def test_store_elasticity_predictions():
    """A whole-category change moves the category by its category elasticity; a one-product change mostly
    moves sales between products."""
    model = StoreElasticity()
    model.category, model.n_categories = np.array([0, 0, 1]), 2
    model.elasticity, model.promo_lift, model.category_elasticity, model.category_promo_lift = -2.0, 0.5, -0.5, 0.1
    model.level = np.array([10.0, 30.0, 5.0])
    model.share = np.array([0.25, 0.75, 1.0])
    ones = np.ones((3, 1))
    base = PricePlan(price=ones.copy(), discount=0 * ones, regular=ones.copy(), available=ones > 0)
    weeks, days = np.array([0]), np.array([7.0])
    everything = PricePlan(price=1.2 * ones, discount=0 * ones, regular=1.2 * ones, available=ones > 0)
    r = model.predict(base, everything, np.array([True, True, False]), np.array([True, True, False]), weeks, days)
    assert r["category_units"]["pct"] == pytest.approx(100 * (1.2 ** -0.5 - 1))
    one = base.copy()
    one.price[0] = 1.2
    r = model.predict(base, one, np.array([True, False, False]), np.array([True, True, False]), weeks, days)
    pull = 0.25 * 1.2 ** -2.0
    assert r["target_units"]["pct"] == pytest.approx(100 * (1.2 ** (-0.5 * 0.25) * pull / (pull + 0.75) / 0.25 - 1))
    assert r["category_units"]["pct"] == pytest.approx(100 * (1.2 ** (-0.5 * 0.25) - 1))
    gone = base.copy()
    gone.available[0] = False
    r = no_reaction(base, gone, model.level, np.array([True, False, False]), np.array([True, True, False]), weeks,
                    days)
    assert r["target_units"]["pct"] == -100
    assert r["category_units"]["pct"] == pytest.approx(-25.0)


# ---------- scores


def test_rank_correlation_and_its_range():
    x = np.arange(1000.0)
    r = rank_correlation(x, x ** 3)
    assert r["r"] == pytest.approx(1.0)
    noisy = rank_correlation(x, np.random.default_rng(0).permutation(x))
    assert noisy["range"][0] < 0 < noisy["range"][1]


class _Constant:
    """A 'twin' that gives every shopper the same chance of buying each product."""

    def __init__(self, p):
        self.p = p

    def purchase_probability(self, users):
        return np.full((len(users), len(self.p)), self.p)


def test_calibration_scores_add_up():
    purchases = pd.DataFrame({"shopper_id": [0, 1, 2, 3], "product_id": [0, 0, 1, 0], "day": [5, 5, 5, 25]})
    result = calibration(_Constant(np.array([0.5, 0.25])), purchases, n_shoppers=4, n_products=2, cutoff=20,
                         horizon=10, batch_size=3)
    twin = result["methods"]["twin"]
    assert result["pairs"] == 8 and result["pairs_bought"] == 1
    # shopper 3 buys product 0 (predicted 0.5); everything else is a miss
    assert twin["brier"] == pytest.approx((0.25 + 3 * 0.25 + 4 * 0.0625) / 8)
    assert twin["pairs_bought_predicted"] == pytest.approx(3.0)
    assert sum(b["pairs"] for b in twin["bins"]) == 8


# ---------- the twin's learned traits


@pytest.fixture(scope="module")
def fitted(tiny_world):
    events = tiny_world.events(types=("view", "purchase"), extra_columns=("session_id", "timestamp", "quantity"))
    split = make_split(events, tiny_world.n_shoppers, tiny_world.n_products,
                       tiny_world.products.category_id.to_numpy(), cutoff_day=FIRST, horizon=28)
    twin = ShopperTwin(tiny_world, epochs=2, dim=8)
    trips = build_trips(split.train.events, twin.cat)
    return twin.fit(split.train, trips), trips


@pytest.fixture(scope="module")
def twin(fitted):
    return fitted[0]


def test_twin_learns_price_sensitivity_beyond_the_profile(tiny_world, twin):
    truth = pd.read_parquet(tiny_world.path / "answer_key" / "shoppers_truth.parquet").sort_values("shopper_id")
    history = np.ones(tiny_world.n_shoppers)
    result = trait_recovery(twin, truth.reset_index(drop=True), history)["price sensitivity"]
    assert result["twin"]["r"] > 0.3
    assert result["twin"]["r"] > result["group_average_only"]["r"]


def test_twin_without_personal_traits_gives_everyone_their_group_values(fitted):
    twin, trips = fitted
    pooled = without_personal_traits(twin, trips)
    traits = pooled.shopper_traits()
    groups = pooled.choice.group.numpy()
    assert (traits.groupby(groups).price_sens.nunique() == 1).all()
    assert float(pooled.choice.taste_dev.abs().max()) == 0
    assert twin.shopper_traits().price_sens.nunique() > len(np.unique(groups))  # the original is untouched
    # calibrated the same way, so normal sales stay close to the twin's
    users = np.arange(200)
    assert pooled.expected_purchases(users).sum() == pytest.approx(twin.expected_purchases(users).sum(), rel=0.1)
