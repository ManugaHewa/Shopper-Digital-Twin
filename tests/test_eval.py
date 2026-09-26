"""Checks for the evaluation code: metrics, time splits and the scoring harness."""

import shutil

import numpy as np
import pandas as pd
import pytest

from shopper_twin.data import load_public
from shopper_twin.eval import evaluate, make_split, top_k, tune
from shopper_twin.eval.metrics import hit_rate, mrr, ndcg, precision, recall, summarise
from shopper_twin.world import generate_world, get_config


# ---------- metrics on hand-worked examples
def test_metrics_hand_worked_example():
    hits = np.array([[True, False, True, False]])
    n_rel = np.array([2])
    assert hit_rate(hits)[0] == 1
    assert precision(hits)[0] == 0.5
    assert recall(hits, n_rel)[0] == 1
    assert mrr(hits)[0] == 1
    expected = (1 + 1 / np.log2(4)) / (1 + 1 / np.log2(3))
    assert ndcg(hits, n_rel)[0] == pytest.approx(expected)


def test_metrics_with_no_hits_are_zero():
    hits = np.zeros((1, 5), dtype=bool)
    n_rel = np.array([3])
    for value in (hit_rate(hits), precision(hits), recall(hits, n_rel), ndcg(hits, n_rel), mrr(hits)):
        assert value[0] == 0


def test_ndcg_is_one_when_list_is_perfect_even_if_more_items_were_bought():
    hits = np.ones((1, 3), dtype=bool)
    assert ndcg(hits, np.array([10]))[0] == pytest.approx(1.0)
    assert recall(hits, np.array([10]))[0] == pytest.approx(0.3)


def test_mrr_uses_first_hit():
    assert mrr(np.array([[False, False, True, True]]))[0] == pytest.approx(1 / 3)


def test_summarise_only_looks_at_the_first_k_columns():
    m = summarise(np.array([[True, False, True, True]]), np.array([3]), 2)
    assert m["precision@2"] == 0.5 and m["recall@2"] == pytest.approx(1 / 3)


# ---------- splits
def _toy_events():
    # shopper 0 buys item 0 before and after the cutoff and item 1 only after (new);
    # shopper 1 only buys after the cutoff (no history, so not scored);
    # shopper 2 buys before only (nothing to predict, so not scored)
    rows = [
        (0, 0, 1, "purchase"), (0, 0, 12, "purchase"), (0, 1, 13, "purchase"), (0, 2, 3, "view"),
        (1, 0, 11, "purchase"),
        (2, 1, 2, "purchase"), (2, 1, 30, "purchase"),  # day 30 is after the window
    ]
    return pd.DataFrame(rows, columns=["shopper_id", "product_id", "day", "event_type"])


def test_split_never_trains_on_the_future():
    split = make_split(_toy_events(), n_users=3, n_items=3, item_category=np.zeros(3, int), cutoff_day=10, horizon=7)
    assert (split.train.events.day < 10).all()
    assert split.train.cutoff_day == 10


def test_split_boundaries_are_exact():
    # cutoff 10, horizon 7: day 9 trains, days 10 and 16 are in the window, day 17 is outside it
    ev = pd.DataFrame([(0, 0, 9, "purchase"), (0, 1, 10, "purchase"), (0, 2, 16, "purchase"), (0, 3, 17, "purchase")],
                      columns=["shopper_id", "product_id", "day", "event_type"])
    split = make_split(ev, n_users=1, n_items=4, item_category=np.zeros(4, int), cutoff_day=10, horizon=7)
    assert split.train.events.day.tolist() == [9]
    assert split.task.truth.toarray().tolist() == [[0, 1, 1, 0]]


def test_split_scores_only_shoppers_with_history_and_future_purchases():
    split = make_split(_toy_events(), n_users=3, n_items=3, item_category=np.zeros(3, int), cutoff_day=10, horizon=7)
    assert split.task.users.tolist() == [0]
    assert split.task.truth.toarray().tolist() == [[1, 1, 0]]
    assert split.task.truth_new.toarray().tolist() == [[0, 1, 0]]  # item 0 was bought before
    assert split.task.seen.toarray().tolist() == [[1, 0, 0]]  # views don't count as bought


# ---------- harness
class _Fixed:
    name = "fixed"

    def __init__(self, scores):
        self.scores = np.asarray(scores, dtype=float)

    def fit(self, data):
        return self

    def score(self, users):
        return self.scores[users]


def test_top_k_is_best_first_and_respects_exclusions():
    import scipy.sparse as sp

    model = _Fixed([[0.1, 0.9, 0.5, 0.7]])
    assert top_k(model, np.array([0]), 3).tolist() == [[1, 3, 2]]
    exclude = sp.csr_matrix(np.array([[0, 1, 0, 0]]))
    assert top_k(model, np.array([0]), 3, exclude).tolist() == [[3, 2, 0]]


def test_a_model_that_knows_the_answer_scores_perfectly():
    split = make_split(_toy_events(), n_users=3, n_items=3, item_category=np.zeros(3, int), cutoff_day=10, horizon=7)
    oracle = _Fixed([[1.0, 0.9, 0.0], [0, 0, 0], [0, 0, 0]])
    result = evaluate(oracle, split, "all", ks=(2,))
    assert result["ndcg@2"] == pytest.approx(1.0) and result["recall@2"] == pytest.approx(1.0)
    result_new = evaluate(oracle, split, "new", ks=(1,))
    assert result_new["hit_rate@1"] == 1.0  # item 1 is top once item 0 (already bought) is excluded


def test_new_mode_drops_shoppers_with_only_repeats_and_hides_their_history():
    # shopper 0 only re-buys item 0 (nothing new to predict); shopper 1 re-buys item 0 and tries item 1
    ev = pd.DataFrame([(0, 0, 1, "purchase"), (0, 0, 12, "purchase"),
                       (1, 0, 1, "purchase"), (1, 0, 12, "purchase"), (1, 1, 13, "purchase")],
                      columns=["shopper_id", "product_id", "day", "event_type"])
    split = make_split(ev, n_users=2, n_items=3, item_category=np.zeros(3, int), cutoff_day=10, horizon=7)
    oracle = _Fixed([[0, 0, 1], [1, 0.9, 0]])
    result = evaluate(oracle, split, "new", ks=(1, 2))
    assert result["users"] == 1
    assert result["hit_rate@1"] == 1 and result["recall@1"] == 1  # item 0 excluded, so item 1 comes first


def test_top_k_breaks_ties_by_product_id_and_matches_a_full_stable_sort():
    rng = np.random.default_rng(0)
    scores = rng.integers(0, 5, size=(50, 300)).astype(float)
    got = top_k(_Fixed(scores), np.arange(50), 10)
    assert (got == np.argsort(-scores, axis=1, kind="stable")[:, :10]).all()


def test_top_k_gives_the_same_answer_for_any_batch_size():
    import scipy.sparse as sp

    rng = np.random.default_rng(1)
    scores = rng.random((40, 30))
    exclude = sp.random(40, 30, density=0.3, random_state=2, format="csr")
    model, users = _Fixed(scores), rng.permutation(40)
    assert (top_k(model, users, 5, exclude, batch_size=7) == top_k(model, users, 5, exclude, batch_size=10**6)).all()


def test_top_k_pads_with_minus_one_when_too_few_products_are_allowed():
    import scipy.sparse as sp

    exclude = sp.csr_matrix(np.array([[0, 1, 1, 1, 0]]))
    assert top_k(_Fixed([[5, 4, 3, 2, 1]]), np.array([0]), 3, exclude).tolist() == [[0, 4, -1]]


def test_top_k_rejects_nan_scores():
    with pytest.raises(ValueError, match="NaN"):
        top_k(_Fixed([[0.1, np.nan]]), np.array([0]), 1)


def test_evaluate_is_the_same_for_any_batch_size():
    rng = np.random.default_rng(3)
    rows = [(u, int(i), int(d), "purchase") for u in range(60) for i, d in
            zip(rng.integers(0, 40, 12), rng.integers(0, 20, 12))]
    split = make_split(pd.DataFrame(rows, columns=["shopper_id", "product_id", "day", "event_type"]),
                       n_users=60, n_items=40, item_category=np.zeros(40, int), cutoff_day=10, horizon=10)
    model = _Fixed(rng.random((60, 40)))
    for mode in ("all", "new"):
        a = evaluate(model, split, mode, ks=(5,), batch_size=7)
        b = evaluate(model, split, mode, ks=(5,), batch_size=10**6)
        assert {k: v for k, v in a.items() if k != "ms_per_user"} == {k: v for k, v in b.items() if k != "ms_per_user"}


def test_tune_works_with_any_cut_off_and_picks_the_best():
    split = make_split(_toy_events(), n_users=3, n_items=3, item_category=np.zeros(3, int), cutoff_day=10, horizon=7)

    class Pick(_Fixed):
        def __init__(self, good):
            super().__init__([[1.0, 0.9, 0.0]] * 3 if good else [[0.0, 0.1, 1.0]] * 3)

    best, trials = tune(Pick, [{"good": False}, {"good": True}], split, modes=("all",), metric="ndcg@1")
    assert best == {"all": {"good": True}} and len(trials) == 2


# ---------- data loading never needs the answer key
def test_public_loader_works_without_the_answer_key(tmp_path):
    path = generate_world(get_config("tiny", n_days=10), root=tmp_path, verbose=False)
    shutil.rmtree(path / "answer_key")
    world = load_public(path)
    ev = world.events(("purchase",))
    assert len(ev) > 0 and set(ev.event_type) == {"purchase"}
    assert ev.day.between(0, 9).all()


def test_public_manifest_reveals_no_hidden_settings_or_future_totals(tmp_path):
    path = generate_world(get_config("tiny", n_days=10), root=tmp_path, verbose=False)
    manifest = load_public(path).manifest
    assert set(manifest) == {"name", "start_date", "n_days", "n_products", "n_shoppers"}
    text = (path / "manifest.json").read_text()
    for secret in ("seed", "rules", "counts", "life_event"):
        assert secret not in text


def test_old_worlds_with_a_revealing_manifest_are_refused(tmp_path):
    import json

    path = generate_world(get_config("tiny", n_days=10), root=tmp_path, verbose=False)
    (path / "manifest.json").write_text(json.dumps({"config": {"n_days": 10}}))
    with pytest.raises(ValueError, match="Regenerate"):
        load_public(path)


def test_event_days_match_timestamps(tmp_path):
    path = generate_world(get_config("tiny", n_days=10), root=tmp_path, verbose=False)
    world = load_public(path)
    ev = world.events(extra_columns=("timestamp",))
    expected = ((ev.timestamp - world.start) // pd.Timedelta(days=1)).to_numpy()
    assert (ev.day.to_numpy() == expected).all() and ev.day.min() == 0 and ev.day.max() == 9
