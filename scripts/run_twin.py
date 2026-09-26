"""Fit the shopper twin and score it exactly like the baselines.

    python scripts/run_twin.py --world data/worlds/small
    python scripts/run_twin.py --world data/worlds/small --tune     # also compare settings first (slower)

Protocol (the same as run_baselines.py, no peeking at the future):
  validation: train on days [0, n-56), score purchases in [n-56, n-28)  -> pick settings (with --tune)
  test:       train on days [0, n-28), score purchases in [n-28, n)      -> report
The fitted twin is scored twice on the test window:
  "Shopper twin"                     knows only what happened before the cutoff, like the baselines:
                                     every price stays at its last regular level and there are no promotions
  "Shopper twin (knows price plan)"  also uses the store's own planned prices and promotions for the window

Writes reports/<world>/twin.json, twin.md and twin.png (the twin next to the baselines from baselines.json)
and saves the test-window twin to models/<world>/twin_day<cutoff>.pt for the next milestones.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from shopper_twin.data import load_public  # noqa: E402
from shopper_twin.eval import MODES, evaluate, make_split  # noqa: E402
from shopper_twin.twin import ShopperTwin, build_trips, trip_loglik  # noqa: E402

HORIZON = 28
# chosen on the validation window (run with --tune to repeat the comparison)
DEFAULTS = {"dim": 16, "epochs": 8, "taste_reg": 1.0, "trait_reg": 2.0}
TUNE_GRID = [{"trait_reg": 0.5}, {"trait_reg": 2.0}, {"trait_reg": 8.0}]
TWIN, TWIN_PLAN = "Shopper twin", "Shopper twin (knows price plan)"
MODE_LABELS = {"all": "All purchases", "new": "New-to-shopper products"}


class Precomputed:
    """A fitted model's scores for every shopper in a window, computed once and shared by both tasks."""

    def __init__(self, model, users: np.ndarray, batch_size: int = 1024):
        self.name, self.cutoff_day = model.name, model.cutoff_day
        t0 = time.perf_counter()
        self.scores = np.vstack([model.score(users[s:s + batch_size]).astype(np.float32)
                                 for s in range(0, len(users), batch_size)])
        self.ms_per_user = 1000 * (time.perf_counter() - t0) / max(len(users), 1)
        self.row = np.full(int(users.max()) + 1, -1, dtype=np.int64)
        self.row[users] = np.arange(len(users))

    def score(self, users: np.ndarray) -> np.ndarray:
        rows = self.row[users]
        if (rows < 0).any():
            raise KeyError("scores were not computed for some of these shoppers")
        return self.scores[rows]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world", default="data/worlds/small")
    parser.add_argument("--out", default=None, help="defaults to reports/<world name>")
    parser.add_argument("--models", default="models", help="folder for the saved twin")
    parser.add_argument("--tune", action="store_true", help="compare a few settings on the validation window first")
    parser.add_argument("--epochs", type=int, default=None, help="training passes (default 8; fewer is faster)")
    args = parser.parse_args()
    t_start = time.perf_counter()

    world = load_public(args.world)
    out = Path(args.out or Path("reports") / world.path.name)
    out.mkdir(parents=True, exist_ok=True)
    if world.n_days < 3 * HORIZON + 28:
        raise SystemExit(f"World has {world.n_days} days; the twin needs at least {3 * HORIZON + 28}.")
    settings = dict(DEFAULTS, **({"epochs": args.epochs} if args.epochs else {}))
    test_cutoff = world.n_days - HORIZON

    t0 = time.perf_counter()
    events = world.events(types=("view", "purchase"), extra_columns=("session_id", "timestamp", "quantity"))
    categories = world.products.category_id.to_numpy()
    # one trip table for every fit below: each fit keeps only the trips before its own cutoff
    trips = build_trips(events[events.day < test_cutoff], ShopperTwin(world).cat)
    print(f"World {world.path.name}: {len(events):,} view and purchase events, {len(trips):,} shopping trips "
          f"({time.perf_counter() - t0:.0f}s)")

    def split_at(ev, cutoff):
        return make_split(ev, world.n_shoppers, world.n_products, categories, cutoff, HORIZON)

    trials = []
    if args.tune:
        val = split_at(events, test_cutoff - HORIZON)
        # held-out trips: every shopping trip in the validation window, with its history up to that trip
        held = trips.subset((trips.day >= val.train.cutoff_day) & (trips.day < test_cutoff))
        print(f"Validation: train on days 0-{val.train.cutoff_day - 1}, check on days {val.train.cutoff_day}-"
              f"{test_cutoff - 1} ({len(held):,} held-out trips)")
        for change in TUNE_GRID:
            params = dict(settings, **change)
            t0 = time.perf_counter()
            twin = ShopperTwin(world, **params).fit(val.train, trips)
            trial = {"params": params, **trip_loglik(twin.choice, held, twin.cat)}
            trial["loglik"] = trial["choice_loglik"] + trial["view_loglik"]
            fair = Precomputed(twin.with_prices(False), val.task.users)
            for m in MODES:
                trial[f"ndcg@10_{m}"] = evaluate(fair, val, m, ks=(10,))["ndcg@10"]
            trial["seconds"] = round(time.perf_counter() - t0, 1)
            trials.append(trial)
            print(f"  {change}: held-out log-likelihood {trial['loglik']:.4f} per trip, NDCG@10 "
                  f"{trial['ndcg@10_all']:.3f} (all) / {trial['ndcg@10_new']:.3f} (new) ({trial['seconds']:.0f}s)")
            del twin, fair
        settings = max(trials, key=lambda t: t["loglik"])["params"]
        print(f"  best settings (highest held-out log-likelihood): {settings}")
        del val, held

    test = split_at(events, test_cutoff)
    del events  # the split keeps what it needs
    print(f"Test: train on days 0-{test_cutoff - 1}, score purchases on days {test_cutoff}-{world.n_days - 1} "
          f"for {len(test.task.users):,} shoppers")
    t0 = time.perf_counter()
    twin = ShopperTwin(world, verbose=True, **settings).fit(test.train, trips)
    fit_seconds = time.perf_counter() - t0
    del trips
    model_path = twin.save(Path(args.models) / world.path.name / f"twin_day{twin.cutoff_day}.pt")
    print(f"  fitted in {fit_seconds:.0f}s, saved to {model_path}")

    results = []
    for label, planned in ((TWIN, False), (TWIN_PLAN, True)):
        scored = Precomputed(twin.with_prices(planned), test.task.users)
        for m in MODES:
            r = evaluate(scored, test, m)
            r.update(model=label, fit_seconds=round(fit_seconds, 2), params=settings,
                     ms_per_user=round(scored.ms_per_user, 4))
            results.append(r)
            print(f"  {label:32s} {MODE_LABELS[m]:24s} NDCG@10 {r['ndcg@10']:.3f}")
        del scored

    baselines = load_baselines(out / "baselines.json", test_cutoff)
    report = {
        "world": world.path.name,
        "protocol": {"horizon_days": HORIZON, "test_cutoff_day": test_cutoff, "tuned": args.tune,
                     "validation_cutoff_day": test_cutoff - HORIZON if args.tune else None,
                     "selection_metric": "held-out trip log-likelihood" if args.tune else None},
        "settings": settings,
        "test": results,
        "validation_trials": trials,
        "learned": learned_summary(twin),
        "model_file": model_path.as_posix(),
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    (out / "twin.json").write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    table = markdown(results, baselines, report)
    (out / "twin.md").write_text(table, encoding="utf-8")
    chart(results, baselines, out / "twin.png")
    print()
    print(table)
    print(f"Saved to {out.as_posix()}/ (total {report['runtime_seconds'] / 60:.1f} min)")


def load_baselines(path: Path, cutoff: int) -> list[dict]:
    if not path.exists():
        print(f"  ({path} not found: run scripts/run_baselines.py to compare with the baselines)")
        return []
    report = json.loads(path.read_text(encoding="utf-8"))
    if report["protocol"]["test_cutoff_day"] != cutoff:
        print(f"  ({path} used a different test window, so it is left out)")
        return []
    return [r for r in report["test"] if r["model"] != "Random"]


def learned_summary(twin: ShopperTwin) -> dict:
    """A few population-level numbers the twin learned, for the report."""
    tr = twin.shopper_traits()
    m = twin.choice
    q = tr.price_sens.quantile([0.1, 0.5, 0.9]).round(2).tolist()
    return {
        "price_sensitivity_p10_median_p90": q,
        # how much less likely the median shopper is to pick a product over the others after a 10% price rise
        "median_shopper_10pct_price_rise_effect": round(1 - 1.1 ** -q[1], 3),
        "share_liking_store_brand": round(float((tr.store > 0).mean()), 3),
        "habit_median": round(float(tr.habit.median()), 2),
        "promo_effect_on_choice": round(float(m.promo), 2),
        "promo_effect_on_views": round(float(m.view_promo), 2),
        "habit_effect_on_views": round(float(m.view_habit), 2),
        "expected_trips_per_shopper_in_window": round(float(twin.trip_rates.sum(axis=1).mean()), 1),
    }


def markdown(results: list[dict], baselines: list[dict], report: dict) -> str:
    lines = []
    for m in MODES:
        rows = sorted((r for r in results + baselines if r["mode"] == m), key=lambda r: -r["ndcg@10"])
        lines += [f"### {MODE_LABELS[m]} ({rows[0]['users']:,} shoppers, next {HORIZON} days)", "",
                  "| Model | NDCG@10 | Hit rate@10 | Recall@10 | Precision@10 | Coverage@10 | Fit (s) | ms/shopper |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            name = f"**{r['model']}**" if r["model"] in (TWIN, TWIN_PLAN) else r["model"]
            lines.append(f"| {name} | {r['ndcg@10']:.3f} | {r['hit_rate@10']:.3f} | {r['recall@10']:.3f} | "
                         f"{r['precision@10']:.3f} | {r['coverage@10']:.3f} | {r['fit_seconds']:.1f} | "
                         f"{r['ms_per_user']:.3f} |")
        lines.append("")
    how = ("settings picked on the validation window by held-out log-likelihood" if report["protocol"]["tuned"]
           else "default settings")
    lines.append(f"Twin: {how}; scored once on the test window. Baselines from baselines.json.")
    lines += ["", "### What the twin learned", ""]
    learned = report["learned"]
    p10, p50, p90 = learned["price_sensitivity_p10_median_p90"]
    lines += [
        f"- Price sensitivity: median {p50}, from {p10} (10% least sensitive) to {p90} (10% most sensitive). "
        f"For the median shopper, a 10% price rise makes a product "
        f"{learned['median_shopper_10pct_price_rise_effect']:.0%} less likely to be picked over the others.",
        f"- {learned['share_liking_store_brand']:.0%} of shoppers lean towards the store's own brand.",
        f"- Habit: buying the same product as last time adds {learned['habit_median']} to its appeal "
        f"(median shopper), and makes it {learned['habit_effect_on_views']} more likely to be looked at (log scale).",
        f"- Promotions add {learned['promo_effect_on_choice']} to a product's appeal and "
        f"{learned['promo_effect_on_views']} to its chance of being looked at (log scale).",
        f"- Expected shopping trips per shopper in the next {HORIZON} days, summed over categories: "
        f"{learned['expected_trips_per_shopper_in_window']}.",
    ]
    return "\n".join(lines) + "\n"


def chart(results: list[dict], baselines: list[dict], path: Path) -> None:
    blue, orange, ink, ink2, grid_c, bg = "#2a78d6", "#e8702a", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6), facecolor=bg)
    for ax, m in zip(axes, MODES):
        rows = sorted((r for r in results + baselines if r["mode"] == m), key=lambda r: r["ndcg@10"])
        names = [r["model"] for r in rows]
        vals = [r["ndcg@10"] for r in rows]
        colours = [orange if n in (TWIN, TWIN_PLAN) else blue for n in names]
        bars = ax.barh(names, vals, color=colours, height=0.6, zorder=3)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9, color=ink)
        ax.set_facecolor(bg)
        ax.set_title(MODE_LABELS[m], loc="left", fontweight="bold", fontsize=12, color=ink)
        ax.set_xlabel("NDCG@10 (higher is better)", color=ink2)
        ax.set_xlim(0, max(vals) * 1.22)
        ax.grid(axis="x", color=grid_c, zorder=0)
        ax.tick_params(colors=ink2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid_c)
    fig.suptitle("Shopper twin (orange) vs the baselines: predicting the next 28 days of purchases", x=0.01,
                 ha="left", fontweight="bold", fontsize=13, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":
    main()
