"""Checks that the fake world is valid, reproducible and behaves like a real store."""

import hashlib

import numpy as np
import pandas as pd
import pytest

from shopper_twin.world import build_world, generate_world, get_config


@pytest.fixture(scope="session")
def world(tmp_path_factory):
    cfg = get_config("tiny", n_days=45)
    path = generate_world(cfg, root=tmp_path_factory.mktemp("worlds"), verbose=False)
    events = pd.read_parquet(path / "public" / "events.parquet")
    return {
        "path": path,
        "events": events,
        "buys": events[events.event_type == "purchase"],
        "products": pd.read_parquet(path / "public" / "products.parquet"),
        "shoppers": pd.read_parquet(path / "public" / "shoppers.parquet"),
        "truth": pd.read_parquet(path / "answer_key" / "shoppers_truth.parquet"),
        "cats": pd.read_parquet(path / "answer_key" / "categories_truth.parquet"),
    }


def _fingerprint(cfg) -> str:
    *_, sim = build_world(cfg)
    h = hashlib.sha256()
    for log in sim.run():
        h.update(log.product_id.tobytes())
        h.update(log.event_type.tobytes())
    return h.hexdigest()


def test_same_seed_gives_same_world():
    cfg = get_config("tiny", n_days=10)
    assert _fingerprint(cfg) == _fingerprint(cfg)


def test_different_seed_gives_different_world():
    assert _fingerprint(get_config("tiny", n_days=10)) != _fingerprint(get_config("tiny", n_days=10, seed=7))


def test_public_files_hide_the_answer_key(world):
    hidden = {"segment", "price_sensitivity", "quality_weight", "quality", "taste", "popularity"}
    assert not hidden & set(world["products"].columns)
    assert not hidden & set(world["shoppers"].columns)


def test_every_purchase_was_viewed_in_the_same_session(world):
    ev = world["events"]
    viewed = set(zip(ev.session_id[ev.event_type == "view"], ev.product_id[ev.event_type == "view"]))
    bought = zip(world["buys"].session_id, world["buys"].product_id)
    assert all(pair in viewed for pair in bought)


def test_events_are_in_time_order_within_sessions(world):
    ev = world["events"]
    assert ev.groupby("session_id")["timestamp"].apply(lambda t: t.is_monotonic_increasing).all()


def test_prices_are_positive_and_store_brand_is_cheaper(world):
    p = world["products"]
    assert (p.base_price > 0).all() and (world["events"].price > 0).all()
    ratio = p.base_price / p.groupby("category_id").base_price.transform("median")
    assert ratio[p.is_store_brand].median() < ratio[~p.is_store_brand].median()


def test_only_pet_owners_buy_pet_food(world):
    b = world["buys"].merge(world["products"][["product_id", "category"]], on="product_id")
    b = b.merge(world["truth"][["shopper_id", "has_dog", "has_cat"]], on="shopper_id")
    assert b[b.category == "Dog Food"].has_dog.all()
    assert b[b.category == "Cat Food"].has_cat.all()


def test_staples_are_bought_repeatedly_more_than_discretionary(world):
    b = world["buys"].merge(world["products"][["product_id", "category_id"]], on="product_id")
    b = b.merge(world["cats"][["category_id", "staple"]], on="category_id")
    per_pair = b.groupby(["shopper_id", "category_id", "staple"]).size().reset_index(name="n")
    means = per_pair.groupby("staple").n.mean()
    assert means[True] > 1.5 * means[False]


def test_price_sensitive_shoppers_buy_more_store_brand(world):
    b = world["buys"].merge(world["products"][["product_id", "is_store_brand"]], on="product_id")
    b = b.merge(world["truth"][["shopper_id", "price_sensitivity"]], on="shopper_id")
    high = b.price_sensitivity > b.price_sensitivity.quantile(0.75)
    low = b.price_sensitivity < b.price_sensitivity.quantile(0.25)
    assert b[high].is_store_brand.mean() > b[low].is_store_brand.mean() + 0.1


def test_a_price_rise_lowers_demand():
    """Running the same world with coffee 40% dearer should sell clearly less coffee (what-if sanity)."""
    cfg = get_config("tiny", n_days=42, n_shoppers=2_000)
    catalog, _, prices, sim = build_world(cfg)
    coffee = np.flatnonzero(catalog.cat_name[catalog.category] == "Coffee")

    def coffee_units(schedule):
        return sum(int(log.quantity[(log.event_type == 2) & np.isin(log.product_id, coffee)].sum())
                   for log in sim.run(prices=schedule))

    dearer = type(prices)(regular=prices.regular.copy(), discount=prices.discount.copy())
    dearer.regular[coffee] *= 1.4
    assert coffee_units(dearer) < 0.95 * coffee_units(prices)
