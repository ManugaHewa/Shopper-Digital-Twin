"""Checks that each baseline does what it claims, on toy data and on a small generated world."""

import numpy as np
import pandas as pd
import pytest

from shopper_twin.baselines import ALS, Blend, BuyAgain, ItemKNN, Popularity, Random, RepurchaseCycle
from shopper_twin.data import load_public
from shopper_twin.eval import TrainData, fit_and_evaluate, make_split
from shopper_twin.world import generate_world, get_config

ALL_MODELS = [Random, Popularity, BuyAgain, RepurchaseCycle, ItemKNN, lambda: ALS(iterations=4), Blend]


def _train(rows, n_users, n_items, cutoff_day, horizon=28, categories=None):
    ev = pd.DataFrame(rows, columns=["shopper_id", "product_id", "day"]).assign(event_type="purchase")
    cats = np.zeros(n_items, int) if categories is None else np.asarray(categories)
    return TrainData(n_users=n_users, n_items=n_items, cutoff_day=cutoff_day, horizon=horizon, events=ev,
                     item_category=cats)


@pytest.fixture(scope="module")
def tiny_split(tmp_path_factory):
    path = generate_world(get_config("tiny", n_days=90), root=tmp_path_factory.mktemp("w"), verbose=False)
    world = load_public(path)
    return make_split(world.events(), world.n_shoppers, world.n_products, world.products.category_id.to_numpy(),
                      cutoff_day=62, horizon=28)


@pytest.mark.parametrize("make", ALL_MODELS)
def test_scores_have_the_right_shape_and_are_repeatable(make, tiny_split):
    users = tiny_split.task.users[:50]
    a = make().fit(tiny_split.train).score(users)
    b = make().fit(tiny_split.train).score(users)
    assert a.shape == (50, tiny_split.train.n_items)
    assert np.isfinite(a).all()
    np.testing.assert_allclose(a, b, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("make", ALL_MODELS)
def test_each_row_belongs_to_its_own_shopper(make, tiny_split):
    model = make().fit(tiny_split.train)
    users = tiny_split.task.users[[5, 1, 9, 3]]
    together = model.score(users)
    for j, u in enumerate(users):
        np.testing.assert_allclose(together[j], model.score(np.array([u]))[0], rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("make", ALL_MODELS[1:])
def test_every_baseline_beats_random(make, tiny_split):
    random_ndcg = fit_and_evaluate(Random, tiny_split, modes=("all",))[0]["ndcg@10"]
    assert fit_and_evaluate(make, tiny_split, modes=("all",))[0]["ndcg@10"] > 3 * random_ndcg


@pytest.mark.parametrize("make", [BuyAgain, RepurchaseCycle, Blend])
def test_personal_baselines_clearly_beat_popularity(make, tiny_split):
    popularity = fit_and_evaluate(Popularity, tiny_split, modes=("all",))[0]["ndcg@10"]
    assert fit_and_evaluate(make, tiny_split, modes=("all",))[0]["ndcg@10"] > 1.5 * popularity


@pytest.mark.parametrize("make", [Popularity, BuyAgain, RepurchaseCycle, ItemKNN, lambda: ALS(iterations=4), Blend])
def test_new_mode_never_recommends_something_already_bought(make, tiny_split):
    from shopper_twin.eval import top_k

    task = tiny_split.task
    recs = top_k(make().fit(tiny_split.train), task.users, 20, task.seen)
    seen = task.seen.toarray() > 0
    assert not np.take_along_axis(seen, np.maximum(recs, 0), axis=1)[recs >= 0].any()


def test_popularity_counts_buyers_not_units():
    # item 0: one shopper buys it 5 times; item 1: three shoppers buy it once each
    data = _train([(0, 0, d) for d in range(5)] + [(u, 1, 1) for u in (1, 2, 3)], 4, 3, 10)
    assert Popularity().fit(data).score(np.array([0]))[0].argmax() == 1


def test_popularity_window_only_counts_recent_sales():
    rows = [(u, 0, 1) for u in range(5)] + [(u, 1, 95) for u in (5, 6)]  # item 0 sold long ago, item 1 recently
    data = _train(rows, 7, 2, 100)
    assert Popularity().fit(data).score(np.array([0]))[0].argmax() == 0
    assert Popularity(window_days=14).fit(data).score(np.array([0]))[0].argmax() == 1


def test_buy_again_prefers_frequent_and_recent():
    rows = [(0, 0, 90), (0, 0, 95), (0, 0, 99), (0, 1, 10), (1, 2, 50), (2, 2, 50)]
    scores = BuyAgain(half_life_days=30).fit(_train(rows, 3, 4, 100)).score(np.array([0]))[0]
    assert scores[0] > scores[1] > scores[2] > scores[3]  # item 2: bought by others only (popularity tie-break)


def test_repurchase_cycle_ranks_due_categories_first():
    # category 0 (item 0) is bought every 7 days and last bought 10 days ago -> due
    # category 1 (item 1) is bought every 60 days and last bought 2 days ago -> not due
    rows = [(0, 0, d) for d in (55, 62, 69, 76, 83, 90)] + [(0, 1, 38), (0, 1, 98)]
    data = _train(rows, 1, 2, 100, horizon=7, categories=[0, 1])
    scores = RepurchaseCycle().fit(data).score(np.array([0]))[0]
    assert scores[0] > scores[1]


def test_repurchase_cycle_rejects_a_zero_prior():
    with pytest.raises(ValueError):
        RepurchaseCycle(prior_strength=0)


def test_repurchase_cycle_longer_window_makes_a_category_more_due():
    rows = [(0, 0, d) for d in (40, 70)]  # 30-day habit, last bought 30 days before the cutoff... minus 20
    short = RepurchaseCycle().fit(_train(rows, 1, 1, 80, horizon=1)).score(np.array([0]))[0, 0]
    long = RepurchaseCycle().fit(_train(rows, 1, 1, 80, horizon=14)).score(np.array([0]))[0, 0]
    assert long > short


def test_repurchase_cycle_splits_a_category_by_product_share():
    rows = [(0, 0, d) for d in (70, 77, 84)] + [(0, 1, 91)]  # same category, switched to item 1 recently
    data = _train(rows, 1, 2, 100, horizon=7, categories=[0, 0])
    s = RepurchaseCycle(half_life_days=7).fit(data).score(np.array([0]))[0]
    assert s[1] > s[0] > 0


def test_item_to_item_recommends_co_purchased_products():
    # shoppers 0-9 buy items 0 and 1 together; shoppers 10-19 buy 2 and 3; shopper 20 bought only item 0
    rows = [(u, i, 5) for u in range(10) for i in (0, 1)] + [(u, i, 5) for u in range(10, 20) for i in (2, 3)]
    rows.append((20, 0, 8))
    scores = ItemKNN(neighbours=2, shrink=0).fit(_train(rows, 21, 4, 10)).score(np.array([20]))[0]
    assert scores[1] > scores[2] and scores[1] > scores[3]


def test_item_to_item_breaks_neighbour_ties_by_product_id():
    # items 1 and 2 are exactly as similar to item 0; with one neighbour kept, item 1 wins on every machine
    rows = [(u, i, 5) for u in range(5) for i in (0, 1, 2)] + [(5, 0, 8)]
    scores = ItemKNN(neighbours=1, shrink=0).fit(_train(rows, 6, 3, 10)).score(np.array([5]))[0]
    assert scores[1] > 0 and scores[2] == 0


def test_item_to_item_recent_purchases_count_more():
    rows = [(u, i, 5) for u in range(10) for i in (0, 1)] + [(u, i, 5) for u in range(10, 20) for i in (2, 3)]
    rows += [(20, 0, 10), (20, 2, 99)]  # shopper 20 bought item 0 long ago and item 2 just now
    data = _train(rows, 21, 4, 100)
    fresh = ItemKNN(neighbours=3, shrink=0, half_life_days=7).fit(data).score(np.array([20]))[0]
    flat = ItemKNN(neighbours=3, shrink=0, half_life_days=10**6).fit(data).score(np.array([20]))[0]
    assert fresh[3] > fresh[1] and flat[3] == pytest.approx(flat[1], rel=0.2)


def test_als_confidence_weight_changes_the_model():
    rng = np.random.default_rng(0)
    rows = [(u, int(i), 1) for u in range(50) for i in rng.choice(20, 5, replace=False)]
    data = _train(rows, 50, 20, 10)
    a = ALS(factors=4, alpha=1, iterations=3).fit(data).score(np.arange(50))
    b = ALS(factors=4, alpha=40, iterations=3).fit(data).score(np.arange(50))
    assert np.abs(a - b).max() > 0.05


def test_blend_weight_moves_from_cycle_to_als():
    from shopper_twin.baselines.blend import percentile_ranks

    class Stub:
        def __init__(self, s):
            self.s = np.asarray(s, float)

        def score(self, users):
            return self.s[users]

    cycle, als = Stub([[3.0, 2.0, 1.0, 0.0]]), Stub([[0.0, 1.0, 2.0, 3.0]])
    users = np.array([0])
    np.testing.assert_allclose(Blend.from_fitted(cycle, als, 0.0).score(users), percentile_ranks(cycle.s))
    np.testing.assert_allclose(Blend.from_fitted(cycle, als, 1.0).score(users), percentile_ranks(als.s))
    assert Blend.from_fitted(cycle, als, 0.3).score(users)[0].argmax() == 0
    assert Blend.from_fitted(cycle, als, 0.7).score(users)[0].argmax() == 3
    with pytest.raises(ValueError):
        Blend(weight=1.5)


def test_als_loss_goes_down_and_it_learns_block_structure():
    rng = np.random.default_rng(0)
    rows = []
    for u in range(200):  # two groups of shoppers, each buying from its own half of the catalogue
        items = rng.choice(20, 8, replace=False) + (0 if u < 100 else 20)
        rows += [(u, int(i), 1) for i in items]
    model = ALS(factors=8, iterations=8, reg=0.1, alpha=10).fit(_train(rows, 200, 40, 10))
    losses = np.array(model.loss_history)
    assert losses[-1] < losses[0]
    assert np.all(np.diff(losses) <= 1e-3 * np.abs(losses[:-1]))  # (almost) never goes up
    scores = model.score(np.arange(200))
    in_block = np.r_[scores[:100, :20].mean(), scores[100:, 20:].mean()]
    out_block = np.r_[scores[:100, 20:].mean(), scores[100:, :20].mean()]
    assert in_block.min() > out_block.max() + 0.2
