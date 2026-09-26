"""Tune and score the baseline recommenders on a generated world.

    python scripts/run_baselines.py --world data/worlds/small
    python scripts/run_baselines.py --world data/worlds/small --quick     # default settings, no tuning

Protocol (no peeking at the future, no tuning on the test window):
  validation: train on days [0, n-56), score purchases in [n-56, n-28)  -> pick settings
  test:       train on days [0, n-28), score purchases in [n-28, n)      -> report
Two tasks: "all" = every product bought in the window; "new" = only products the
shopper had never bought before (their past purchases are removed from the list).

Writes reports/<world>/baselines.json, baselines.md and baselines.png.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from shopper_twin.baselines import ALS, Blend, BuyAgain, ItemKNN, Popularity, Random, RepurchaseCycle  # noqa: E402
from shopper_twin.data import load_public  # noqa: E402
from shopper_twin.eval import MODES, evaluate, grid, make_split  # noqa: E402

HORIZON = 28
MODELS = [
    ("Random", Random, [{}]),
    ("Popularity", Popularity, grid(window_days=[None, 14, 28, 56])),
    ("Buy again", BuyAgain, grid(half_life_days=[7, 14, 30, 60, 120])),
    ("Repurchase cycle", RepurchaseCycle, grid(half_life_days=[4, 7, 15, 30], prior_strength=[0.5, 1, 2])),
    ("Item-to-item", ItemKNN, grid(neighbours=[1, 2, 5, 20, 100], shrink=[10, 100], half_life_days=[90, 365])),
    ("Matrix factorisation (ALS)", ALS,
     grid(factors=[32, 64, 128, 256], alpha=[2, 5], use_views=[False])
     + [{"factors": 64, "alpha": 2, "use_views": True}]),  # do views and carts help as weak signals?
]
BLEND = "Blend (cycle + ALS)"
BLEND_WEIGHTS = [0.0, 0.03, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0]
MODE_LABELS = {"all": "All purchases", "new": "New-to-shopper products"}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world", default="data/worlds/small")
    parser.add_argument("--out", default=None, help="defaults to reports/<world name>")
    parser.add_argument("--quick", action="store_true", help="skip tuning and use default settings")
    args = parser.parse_args()
    t_start = time.perf_counter()

    world = load_public(args.world)
    out = Path(args.out or Path("reports") / world.path.name)
    out.mkdir(parents=True, exist_ok=True)
    events = world.events()
    categories = world.products.category_id.to_numpy()
    n_events = len(events)

    def split_at(ev, cutoff):
        return make_split(ev, world.n_shoppers, world.n_products, categories, cutoff, HORIZON)

    tuned = not args.quick
    if tuned and world.n_days < 3 * HORIZON:
        print(f"World has {world.n_days} days; tuning needs at least {3 * HORIZON}. Using default settings.")
        tuned = False
    test = split_at(events, world.n_days - HORIZON)
    val = split_at(events, world.n_days - 2 * HORIZON) if tuned else None
    del events
    print(f"World {world.path.name}: {n_events:,} events. Test window days {world.n_days - HORIZON}-"
          f"{world.n_days - 1}, {len(test.task.users):,} shoppers scored.")

    cache = {}

    def fitted(cls, params, split):
        """Fit each (model, settings, split) once and reuse it."""
        key = (cls.__name__, json.dumps(params, sort_keys=True), split.train.cutoff_day)
        if key not in cache:
            t0 = time.perf_counter()
            cache[key] = (cls(**params).fit(split.train), time.perf_counter() - t0)
        return cache[key]

    def pick_best(trials, mode):
        scored = [t for t in trials if t["mode"] == mode and np.isfinite(t["ndcg@10"])]
        return max(scored, key=lambda t: t["ndcg@10"])["params"]

    best, trials = {}, []
    for label, cls, params_grid in MODELS:
        t0 = time.perf_counter()
        if not tuned:
            best[label] = {m: {} for m in MODES}
            continue
        model_trials = [{"params": params, **evaluate(fitted(cls, params, val)[0], val, m), "model": label}
                        for params in params_grid for m in MODES]
        best[label] = {m: pick_best(model_trials, m) for m in MODES}
        trials += model_trials
        print(f"  tuned {label:28s} in {time.perf_counter() - t0:5.1f}s  best: {best[label]}")

    # the blend reuses the tuned cycle and ALS models and only tunes the mixing weight
    best[BLEND] = {}
    for m in MODES:
        if not tuned:
            best[BLEND][m] = {"weight": 0.3, "cycle": {}, "als": {}}
            continue
        cycle = fitted(RepurchaseCycle, best["Repurchase cycle"][m], val)[0]
        als = fitted(ALS, best["Matrix factorisation (ALS)"][m], val)[0]
        blend_trials = [{"params": {"weight": w, "cycle": best["Repurchase cycle"][m],
                                    "als": best["Matrix factorisation (ALS)"][m]},
                         **evaluate(Blend.from_fitted(cycle, als, w), val, m), "model": BLEND}
                        for w in BLEND_WEIGHTS]
        best[BLEND][m] = pick_best(blend_trials, m)
        trials += blend_trials
    if tuned:
        print(f"  tuned {BLEND:28s} best weight: { {m: best[BLEND][m]['weight'] for m in MODES} }")

    results = []
    for label, cls in [(label, cls) for label, cls, _ in MODELS] + [(BLEND, Blend)]:
        for m in MODES:
            params = best[label][m]
            if cls is Blend:
                cycle, t_cycle = fitted(RepurchaseCycle, params["cycle"], test)
                als, t_als = fitted(ALS, params["als"], test)
                model, fit_seconds = Blend.from_fitted(cycle, als, params["weight"]), t_cycle + t_als
            else:
                model, fit_seconds = fitted(cls, params, test)
            results.append({**evaluate(model, test, m), "model": label, "fit_seconds": round(fit_seconds, 2),
                            "params": params})

    report = {
        "world": world.path.name,
        "protocol": {"horizon_days": HORIZON, "validation_cutoff_day": world.n_days - 2 * HORIZON if tuned else None,
                     "test_cutoff_day": world.n_days - HORIZON, "tuned": tuned, "selection_metric": "ndcg@10"},
        "test": results,
        "validation_trials": trials,
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    (out / "baselines.json").write_text(json.dumps(report, indent=2, default=str))
    (out / "baselines.md").write_text(markdown_table(results, report))
    chart(results, out / "baselines.png")
    print()
    print(markdown_table(results, report))
    print(f"Saved to {out}/ (total {report['runtime_seconds']}s)")


def markdown_table(results: list[dict], report: dict) -> str:
    lines = []
    for m in MODES:
        rows = sorted((r for r in results if r["mode"] == m), key=lambda r: -r["ndcg@10"])
        lines += [f"### {MODE_LABELS[m]} ({rows[0]['users']:,} shoppers, next {HORIZON} days)", "",
                  "| Model | NDCG@10 | Hit rate@10 | Recall@10 | Precision@10 | Coverage@10 | Fit (s) | ms/shopper |",
                  "|---|---|---|---|---|---|---|---|"]
        for r in rows:
            lines.append(f"| {r['model']} | {r['ndcg@10']:.3f} | {r['hit_rate@10']:.3f} | {r['recall@10']:.3f} | "
                         f"{r['precision@10']:.3f} | {r['coverage@10']:.3f} | {r['fit_seconds']:.1f} | "
                         f"{r['ms_per_user']:.3f} |")
        lines.append("")
    tuned = "tuned on the validation window" if report["protocol"]["tuned"] else "default settings (--quick)"
    lines.append(f"Settings {tuned}; scored once on the test window.")
    return "\n".join(lines) + "\n"


def chart(results: list[dict], path: Path) -> None:
    blue, ink, ink2, grid_c, bg = "#2a78d6", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), facecolor=bg)
    for ax, m in zip(axes, MODES):
        rows = sorted((r for r in results if r["mode"] == m), key=lambda r: r["ndcg@10"])
        names = [r["model"] for r in rows]
        vals = [r["ndcg@10"] for r in rows]
        bars = ax.barh(names, vals, color=blue, height=0.6, zorder=3)
        ax.bar_label(bars, fmt="%.3f", padding=3, fontsize=9, color=ink)
        ax.set_facecolor(bg)
        ax.set_title(MODE_LABELS[m], loc="left", fontweight="bold", fontsize=12, color=ink)
        ax.set_xlabel("NDCG@10 (higher is better)", color=ink2)
        ax.set_xlim(0, max(vals) * 1.2)
        ax.grid(axis="x", color=grid_c, zorder=0)
        ax.tick_params(colors=ink2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid_c)
    fig.suptitle("Baseline recommenders: predicting the next 28 days of purchases", x=0.01, ha="left",
                 fontweight="bold", fontsize=13, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":
    main()
