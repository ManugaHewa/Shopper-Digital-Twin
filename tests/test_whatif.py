"""Checks the what-if engine: scenarios turn into the right prices, and the twin reacts in the right direction."""

import numpy as np
import pandas as pd
import pytest

from shopper_twin.eval import make_split
from shopper_twin.sim import (PriceChange, PricePlan, Promotion, Scenario, Select, StockOut, WhatIf,
                              preset_scenarios, realised_price_change)
from shopper_twin.twin import ShopperTwin

FAST = {"epochs": 2, "dim": 8}


# ---------- scenarios, on a hand-made catalogue


@pytest.fixture
def products():
    return pd.DataFrame({
        "product_id": [0, 1, 2, 3],
        "category": ["Coffee", "Coffee", "Coffee", "Milk"],
        "department": ["Grocery"] * 4,
        "brand": ["A", "B", "Store", "A"],
        "is_store_brand": [False, False, True, False],
    })


@pytest.fixture
def plan():
    regular = np.full((4, 3), 10.0)
    discount = np.zeros((4, 3))
    discount[1, 1] = 0.2  # product 1 is on promotion in week 1
    return PricePlan(price=regular * (1 - discount), discount=discount, regular=regular,
                     available=np.ones((4, 3), dtype=bool))


def test_select_combines_filters(products):
    assert Select(category="Coffee").mask(products).tolist() == [True, True, True, False]
    assert Select(category="Coffee", store_brand=False).mask(products).tolist() == [True, True, False, False]
    assert Select(brand="A").mask(products).tolist() == [True, False, False, True]
    assert Select(product_ids=(2, 3)).mask(products).tolist() == [False, False, True, True]
    with pytest.raises(ValueError, match="no product has category"):
        Select(category="Tea").mask(products)


def test_price_change_keeps_planned_promotions(products, plan):
    new = Scenario("up", [PriceChange(Select(category="Coffee"), percent=20)]).apply(plan, products)
    np.testing.assert_allclose(new.regular[:3], 12.0)
    np.testing.assert_allclose(new.price[1, 1], 12.0 * 0.8)  # the promotion's discount applies to the new price
    np.testing.assert_allclose(new.price[3], 10.0)  # milk untouched
    np.testing.assert_allclose(plan.price[0], 10.0)  # the original plan is not changed
    assert realised_price_change(plan, new, np.array([0, 1, 2]), np.arange(3)) == pytest.approx(20.0)


def test_promotion_and_stock_out_only_in_their_weeks(products, plan):
    promo = Scenario("promo", [Promotion(Select(brand="A"), percent_off=25)], weeks=(1, 2)).apply(plan, products)
    np.testing.assert_allclose(promo.price[0], [10.0, 7.5, 7.5])
    assert promo.promo[3].tolist() == [0.0, 1.0, 1.0]
    out = Scenario("gone", [StockOut(Select(product_ids=(2,)))], weeks=(2, 2)).apply(plan, products)
    assert out.available[2].tolist() == [True, True, False]
    assert out.available[[0, 1, 3]].all()
    with pytest.raises(ValueError, match="between 0 and 100"):
        Scenario("bad", [Promotion(Select(brand="A"), percent_off=120)]).apply(plan, products)


def test_scenario_describes_itself():
    s = Scenario("x", [PriceChange(Select(category="Milk", store_brand=True), percent=-15),
                       StockOut(Select(product_ids=(7,)))])
    assert s.describe() == "store-brand Milk price -15%; product 7 out of stock"


# ---------- the twin's answers, on the small test world


@pytest.fixture(scope="module")
def world(tiny_world):
    return tiny_world


@pytest.fixture(scope="module")
def whatif(world):
    events = world.events(types=("view", "purchase"), extra_columns=("session_id", "timestamp", "quantity"))
    split = make_split(events, world.n_shoppers, world.n_products, world.products.category_id.to_numpy(),
                       cutoff_day=62, horizon=28)
    twin = ShopperTwin(world, **FAST).fit(split.train)
    return WhatIf(twin, world, batch_size=300)


def _coffee_brand(world):
    coffee = world.products[(world.products.category == "Coffee") & ~world.products.is_store_brand]
    return coffee.brand.value_counts().index[0]


def test_no_change_means_no_difference(whatif):
    r = whatif.run(Scenario("nothing", [PriceChange(Select(category="Coffee"), percent=0)]), n_boot=20)
    assert r["target_units"]["pct"] == pytest.approx(0, abs=1e-9)
    assert r["category_revenue"]["pct"] == pytest.approx(0, abs=1e-9)


def test_dearer_brand_loses_sales_to_the_rest_of_its_category(world, whatif):
    brand = _coffee_brand(world)
    r = whatif.run(Scenario("brand up", [PriceChange(Select(category="Coffee", brand=brand), percent=30)]),
                   n_boot=20)
    assert r["target_units"]["pct"] < -3
    assert r["substitution"]["other_products_change"] > 0  # some shoppers switch brands
    assert r["category_units"]["pct"] > r["target_units"]["pct"]  # the category loses less than the brand
    lo, hi = r["target_units"]["band_pct"]
    assert lo <= r["target_units"]["pct"] <= hi


def test_promotion_and_price_cut_raise_sales(world, whatif):
    brand = _coffee_brand(world)
    promo = whatif.run(Scenario("promo", [Promotion(Select(category="Coffee", brand=brand), percent_off=25)]),
                       n_boot=20)
    cut = whatif.run(Scenario("cut", [PriceChange(Select(category="Coffee", brand=brand), percent=-25)]),
                     n_boot=20)
    assert promo["target_units"]["pct"] > 0 and cut["target_units"]["pct"] > 0
    # a promotion is the same price cut plus the promotion's own pull, so it sells more
    assert promo["target_units"]["pct"] > cut["target_units"]["pct"]


def test_stock_out_moves_sales_to_other_products(world, whatif):
    coffee = world.products[world.products.category == "Coffee"].product_id.to_numpy()
    r = whatif.run(Scenario("gone", [StockOut(Select(product_ids=(int(coffee[0]),)))]), n_boot=20)
    assert r["target_units"]["new"] == 0
    assert r["substitution"]["other_products_change"] > 0
    assert r["segments_measure"] == "category"
    table = r["products"]
    assert table.loc[table.target, "new_units"].sum() == 0


def test_a_stock_out_in_one_week_only_removes_that_week(world, whatif):
    twin = whatif.twin
    users = np.arange(50)
    coffee = int(world.products[world.products.category == "Coffee"].product_id.iloc[0])
    plan = whatif.base_plan
    weeks, frac = twin.window_weeks()
    full = twin.expected_purchases(users, plan.price, plan.promo)
    gone = plan.available.copy()
    gone[coffee, weeks[0]] = False
    part = twin.expected_purchases(users, plan.price, plan.promo, gone)
    assert (part[:, coffee] < full[:, coffee]).all()
    # the first window week holds frac[0] of the days; removing it leaves roughly the rest
    np.testing.assert_allclose(part[:, coffee].sum(), full[:, coffee].sum() * (1 - frac[0]), rtol=0.3)


def test_category_totals_match_the_twin(whatif):
    twin = whatif.twin
    totals = whatif.category_units()
    plan = whatif.base_plan
    expected = twin.expected_purchases(whatif.users, plan.price, plan.promo)
    units = expected * twin.quantity[whatif.users][:, twin.cat.category]
    np.testing.assert_allclose(totals.units.sum(), units.sum(), rtol=1e-9)


def test_presets_build_on_any_world(world):
    purchases = world.events(types=("purchase",), extra_columns=("quantity",))
    scenarios = preset_scenarios(world, purchases, before_day=62)
    assert len(scenarios) == 5
    for s in scenarios:
        assert s.targets(world.products.sort_values("product_id")).any()
