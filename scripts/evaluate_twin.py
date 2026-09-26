"""Score the twin against the true store: how right are its what-if answers?

    python scripts/evaluate_twin.py --world data/worlds/small

Because the store is simulated, it can be re-run with any price change to see what really happens. No real
retailer can do that. This script asks the same what-if questions to the twin and to three simpler methods, and
scores all of them against the truth:

  Scenarios    the 5 standard questions from milestone 4 plus 24 random ones (price rises and cuts, promotions,
               stock-outs), each over the test window (days 152-179 on the small world)
  Methods      the twin; the twin without personal traits (everyone gets their group's average); store-wide
               elasticities (the classic pricing-team model); no reaction at all
  Segments     does the twin know WHICH shoppers react? Changes by true price-sensitivity quarter
  Traits       learned vs true price sensitivity, store-brand liking, brand loyalty, habit and taste
  Calibration  when the twin says "30% chance", does the shopper buy about 30% of the time?

Needs the world's answer_key folder and the twin saved by scripts/run_twin.py. Takes about 15 minutes (--quick:
about 7). Writes reports/<world>/evaluation.json, evaluation.md and evaluation.png. --report-only redraws the
.md and .png from an existing evaluation.json in a few seconds.
"""

import argparse
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import FuncFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from shopper_twin.data import load_public  # noqa: E402
from shopper_twin.eval.comparators import StoreElasticity, no_reaction, without_personal_traits  # noqa: E402
from shopper_twin.eval.recovery import calibration, taste_recovery, trait_recovery  # noqa: E402
from shopper_twin.eval.truth import Outcome, TrueStore, true_effect  # noqa: E402
from shopper_twin.sim import PricePlan, Promotion, StockOut, WhatIf, preset_scenarios, random_scenarios  # noqa: E402
from shopper_twin.twin import ShopperTwin, build_trips  # noqa: E402

HORIZON = 28
METHODS = {"twin": "Twin", "pooled": "Twin without personal traits", "elasticity": "Store-wide elasticity model",
           "no_reaction": "No reaction"}
MEASURES = {"target_units": "Changed products' units", "category_units": "Whole category's units",
            "category_revenue": "Category revenue"}
QUARTERS = ["lowest", "low", "high", "highest"]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world", default="data/worlds/small")
    parser.add_argument("--out", default=None, help="defaults to reports/<world name>")
    parser.add_argument("--models", default="models", help="folder with the saved twin")
    parser.add_argument("--replicates", type=int, default=3, help="runs of the true store per scenario")
    parser.add_argument("--random", type=int, default=24, help="random scenarios on top of the 5 standard ones")
    parser.add_argument("--bootstrap", type=int, default=200, help="resamples of the shoppers for the twin's bands")
    parser.add_argument("--quick", action="store_true", help="2 replicates and 4 random scenarios (a quick check)")
    parser.add_argument("--report-only", action="store_true", help="redraw the report from evaluation.json")
    args = parser.parse_args()
    if args.quick:
        args.replicates, args.random = 2, 4
    t_start = time.perf_counter()

    world = load_public(args.world)
    answer_key = world.path / "answer_key"
    if not (answer_key / "manifest.json").exists():
        raise SystemExit(f"{answer_key.as_posix()} not found: the evaluation needs the world's answer key "
                         f"(regenerate the world with scripts/generate_world.py).")
    out = Path(args.out or Path("reports") / world.path.name)
    out.mkdir(parents=True, exist_ok=True)
    cutoff = world.n_days - HORIZON
    last_day = cutoff + HORIZON - 1
    model_path = Path(args.models) / world.path.name / f"twin_day{cutoff}.pt"
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found: run scripts/run_twin.py --world {args.world} first.")
    twin = ShopperTwin.load(model_path, world, planned_prices=True)
    truth_shoppers = pd.read_parquet(answer_key / "shoppers_truth.parquet").sort_values("shopper_id")
    truth_shoppers = truth_shoppers.reset_index(drop=True)
    if args.report_only:
        report = json.loads((out / "evaluation.json").read_text(encoding="utf-8"))
        report.update(summaries(report["scenarios"]))
        write_report(report, out, truth_shoppers, twin)
        return
    products = world.products.sort_values("product_id").reset_index(drop=True)
    purchases = world.events(types=("purchase",), extra_columns=("quantity",))
    print(f"Twin {model_path.as_posix()}: trained on days 0-{cutoff - 1}; scoring what-if answers for days "
          f"{cutoff}-{last_day}")

    # ---- the true store, and its answer with no change (several runs, each with its own random numbers)
    t0 = time.perf_counter()
    store = TrueStore(world.path)
    store.state_at(cutoff)
    print(f"Rebuilt the true store from the answer key and replayed it to day {cutoff} "
          f"({time.perf_counter() - t0:.0f}s)")
    t0 = time.perf_counter()
    base = [Outcome.from_purchases(store.purchases(cutoff, last_day, replicate=r), world.n_products)
            for r in range(args.replicates)]
    truth_run_seconds = (time.perf_counter() - t0) / args.replicates
    print(f"True store with its own plan: {np.mean([b.units.sum() for b in base]):,.0f} units per run "
          f"({args.replicates} runs, {truth_run_seconds:.0f}s each)")

    # ---- the simpler methods, from data before the window only
    base_plan = PricePlan.from_world(world)
    weeks, frac = twin.window_weeks()
    week_days = frac * HORIZON
    window = (int(weeks.min()), int(weeks.max()))
    category = twin.cat.category
    elasticity = StoreElasticity().fit(purchases[purchases.day < cutoff], base_plan, category,
                                       first_week=cutoff // 7)
    print(f"Store-wide elasticity model: switching elasticity {elasticity.elasticity:.2f}, promotion lift "
          f"x{np.exp(elasticity.promo_lift):.2f}; category elasticity {elasticity.category_elasticity:.2f}, "
          f"category promotion lift x{np.exp(elasticity.category_promo_lift):.2f}")
    whatif = WhatIf(twin, world)
    t0 = time.perf_counter()
    events = world.events(types=("view", "purchase"), extra_columns=("session_id", "timestamp", "quantity"))
    trips = build_trips(events[events.day < cutoff], twin.cat)
    del events
    pooled = WhatIf(without_personal_traits(twin, trips), world)
    print(f"Twin without personal traits fitted ({time.perf_counter() - t0:.0f}s)")
    quarter = np.asarray(pd.qcut(truth_shoppers.price_sensitivity.rank(method="first"), 4, labels=QUARTERS))

    # ---- every scenario, every method, against the truth
    scenarios = (preset_scenarios(world, purchases, cutoff)
                 + random_scenarios(world, purchases, cutoff, n=args.random, seed=0))
    rows, twin_seconds = [], []
    for i, scenario in enumerate(scenarios):
        t0 = time.perf_counter()
        plan = scenario.apply(base_plan, products, window)
        target = scenario.targets(products)
        in_categories = np.isin(category, np.unique(category[target]))
        new = [Outcome.from_purchases(store.purchases(cutoff, last_day, plan, replicate=r), world.n_products)
               for r in range(args.replicates)]
        truth = true_effect(base, new, target, in_categories)
        t1 = time.perf_counter()
        tw = whatif.run(scenario, n_boot=args.bootstrap, shopper_detail=True)
        twin_seconds.append(time.perf_counter() - t1)
        po = pooled.run(scenario, n_boot=0, shopper_detail=True)
        el = elasticity.predict(base_plan, plan, target, in_categories, weeks, week_days)
        nr = no_reaction(base_plan, plan, elasticity.level, target, in_categories, weeks, week_days)
        kind = scenario_kind(scenario)
        # who reacts: the changed products' units (for a stock-out, the category's) by true sensitivity quarter
        mask, key = (in_categories, "c") if kind == "stock-out" else (target, "t")
        segments = {"truth": _by_quarter(*_true_shopper_units(base, new, mask, world.n_shoppers), quarter),
                    "twin": _by_quarter(tw["per_shopper"][f"{key}_base"], tw["per_shopper"][f"{key}_new"], quarter),
                    "pooled": _by_quarter(po["per_shopper"][f"{key}_base"], po["per_shopper"][f"{key}_new"],
                                          quarter)}
        rows.append({
            "name": scenario.name, "description": scenario.describe(), "kind": kind,
            "n_target_products": int(target.sum()), "categories": tw["categories"],
            "price_change_pct": tw["price_change_pct"],
            "truth": truth,
            "predictions": {
                "twin": {k: tw[k]["pct"] for k in MEASURES},
                "pooled": {k: po[k]["pct"] for k in MEASURES},
                "elasticity": {k: el[k]["pct"] for k in MEASURES},
                "no_reaction": {k: nr[k]["pct"] for k in MEASURES},
            },
            "twin_bands": {k: tw[k]["band_pct"] for k in MEASURES},
            "segments_measure": "category" if key == "c" else "target",
            "segments": segments,
        })
        print(f"  [{i + 1}/{len(scenarios)}] {scenario.name}: changed products {truth['target_units']['pct']:+.1f}% "
              f"true vs {tw['target_units']['pct']:+.1f}% twin; category {truth['category_units']['pct']:+.1f}% vs "
              f"{tw['category_units']['pct']:+.1f}% ({time.perf_counter() - t0:.0f}s)", flush=True)

    # ---- did it learn the right things about each shopper?
    t0 = time.perf_counter()
    history = np.bincount(purchases.shopper_id[purchases.day < cutoff], minlength=world.n_shoppers)
    traits = trait_recovery(twin, truth_shoppers, history)
    arrays = np.load(answer_key / "arrays.npz")
    quality = pd.read_parquet(answer_key / "products_truth.parquet").sort_values("product_id").quality.to_numpy()
    taste = taste_recovery(twin, store.state_at(cutoff).taste, arrays["product_style"], quality,
                           truth_shoppers.quality_weight.to_numpy(), store.cfg.rules.taste_scale)
    calib = calibration(twin, purchases, world.n_shoppers, world.n_products, cutoff, HORIZON)
    print(f"Traits, taste and calibration checked ({time.perf_counter() - t0:.0f}s)")

    report = {
        "world": world.path.name,
        "twin": model_path.as_posix(),
        "window": {"first_day": cutoff, "last_day": last_day, "weeks": [int(w) for w in weeks]},
        "replicates": args.replicates,
        "comparator_fit": {"switching_elasticity": elasticity.elasticity,
                           "promo_lift": float(np.exp(elasticity.promo_lift)),
                           "category_elasticity": elasticity.category_elasticity,
                           "category_promo_lift": float(np.exp(elasticity.category_promo_lift))},
        **summaries(rows),
        "traits": traits,
        "taste": taste,
        "calibration": calib,
        "speed": {"twin_seconds_per_scenario": float(np.median(twin_seconds)),
                  "true_store_seconds_per_run": truth_run_seconds, "shoppers": int(world.n_shoppers)},
        "scenarios": rows,
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    write_report(report, out, truth_shoppers, twin)
    print(f"Total {report['runtime_seconds'] / 60:.1f} min")


def summaries(rows: list) -> dict:
    """Everything the report computes from the per-scenario results."""
    return {"accuracy": accuracy(rows), "twin_bands": band_check(rows),
            "truth_noise": {k: float(np.median([r["truth"][k]["se"] or 0 for r in rows])) for k in MEASURES},
            "segments": segment_summary(rows)}


def write_report(report: dict, out: Path, truth_shoppers: pd.DataFrame, twin) -> None:
    (out / "evaluation.json").write_text(json.dumps(report, indent=2, default=_plain), encoding="utf-8")
    text = markdown(report)
    (out / "evaluation.md").write_text(text, encoding="utf-8")
    chart(report, truth_shoppers, twin, out / "evaluation.png")
    print()
    print(text)
    print(f"Saved to {out.as_posix()}/")


def scenario_kind(scenario) -> str:
    kinds = {"stock-out" if isinstance(ch, StockOut) else "promotion" if isinstance(ch, Promotion) else "price"
             for ch in scenario.changes}
    return kinds.pop() if len(kinds) == 1 else "mixed"


def _true_shopper_units(base: list, new: list, mask: np.ndarray, n_shoppers: int) -> tuple[np.ndarray, np.ndarray]:
    """Units of the `mask` products per shopper, summed over the true store's runs."""
    def units(outcome: Outcome) -> np.ndarray:
        s = outcome.shopper_units
        shopper, product = s.index.get_level_values(0).to_numpy(), s.index.get_level_values(1).to_numpy()
        keep = mask[product]
        return np.bincount(shopper[keep], weights=s.to_numpy()[keep], minlength=n_shoppers)
    return sum(units(b) for b in base), sum(units(n) for n in new)


def _by_quarter(base: np.ndarray, new: np.ndarray, quarter: np.ndarray) -> dict:
    return {q: _pct(base[quarter == q].sum(), new[quarter == q].sum()) for q in QUARTERS}


def _pct(base: float, new: float) -> float:
    return float(100 * (new / base - 1)) if base > 0 else float("nan")


def _scored(rows: list, measure: str) -> list:
    """Scenarios a measure is scored on: a stock-out always takes the changed products to -100%, which every
    method gets right, so those only count for the category measures."""
    return [r for r in rows if not (measure == "target_units" and r["kind"] == "stock-out")]


def accuracy(rows: list) -> dict:
    """Per method and measure: average and median miss in percentage points, share of scenarios within 2 points
    of the truth, and the correlation between predicted and true changes."""
    out = {}
    for method in METHODS:
        out[method] = {}
        for measure in MEASURES:
            use = _scored(rows, measure)
            pred = np.array([r["predictions"][method][measure] for r in use])
            true = np.array([r["truth"][measure]["pct"] for r in use])
            err = np.abs(pred - true)
            corr = float(np.corrcoef(pred, true)[0, 1]) if pred.std() > 0 and true.std() > 0 else float("nan")
            out[method][measure] = {"scenarios": len(use), "mean_miss": float(err.mean()),
                                    "median_miss": float(np.median(err)), "within_2_points": float((err <= 2).mean()),
                                    # the miss as a share of the true change (only meaningful for big changes)
                                    "median_relative_miss": float(np.median(err / np.maximum(np.abs(true), 1))),
                                    "correlation": corr}
    return out


def band_check(rows: list) -> dict:
    """How often the twin's 90% band (shoppers resampled) contains the truth, and the miss that 90% of scenarios
    stay under: the honest '+/- X points' to quote with a twin answer."""
    out = {}
    for measure in MEASURES:
        use = [r for r in _scored(rows, measure) if r["twin_bands"][measure]]
        inside = [lo <= r["truth"][measure]["pct"] <= hi for r in use for lo, hi in [r["twin_bands"][measure]]]
        miss = np.array([abs(r["predictions"]["twin"][measure] - r["truth"][measure]["pct"]) for r in use])
        size = np.array([max(abs(r["truth"][measure]["pct"]), 1) for r in use])
        out[measure] = {"scenarios": len(use), "band_contains_truth": float(np.mean(inside)),
                        "miss_90th_percentile": float(np.percentile(miss, 90)),
                        "relative_miss_90th_percentile": float(np.percentile(miss / size, 90))}
    return out


def segment_summary(rows: list) -> dict:
    """Over all price and promotion scenarios: average miss per sensitivity quarter, and the average gap between
    the most and least price-sensitive quarters (how differently they react)."""
    use = [r for r in rows if r["kind"] != "stock-out"]
    out = {"scenarios": len(use)}
    for method in ("twin", "pooled"):
        miss = [abs(r["segments"][method][q] - r["segments"]["truth"][q]) for r in use for q in QUARTERS]
        out[method] = {"mean_miss": float(np.nanmean(miss))}
    for who in ("truth", "twin", "pooled"):
        gaps = [abs(r["segments"][who]["highest"] - r["segments"][who]["lowest"]) for r in use]
        out.setdefault("gap_highest_vs_lowest", {})[who] = float(np.nanmean(gaps))
    return out


def _plain(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    if isinstance(x, np.ndarray):
        return x.tolist()
    return str(x)


# ---------- the report


def markdown(report: dict) -> str:
    w, acc, bands, seg = report["window"], report["accuracy"], report["twin_bands"], report["segments"]
    n = len(report["scenarios"])
    twin_t, el_t = acc["twin"]["target_units"]["mean_miss"], acc["elasticity"]["target_units"]["mean_miss"]
    lines = [f"## How right is the twin? Scored against the true store (days {w['first_day']}-{w['last_day']})", "",
             f"{n} what-if scenarios were run through the true store ({report['replicates']} runs each, with the "
             f"answer key) and through each method. A miss is the gap in percentage points between the predicted "
             f"and the true % change: if the truth is -5.3% and the twin says -6.1%, the miss is 0.8 points.", "",
             f"On the changed products' units the twin misses by {twin_t:.1f} points on average, against "
             f"{el_t:.1f} for the store-wide elasticity model.", "",
             "### Average miss (percentage points, lower is better)", "",
             "| Method | " + " | ".join(MEASURES.values()) + " | Typical miss vs the true change (changed "
             "products) |",
             "|---|" + "---|" * (len(MEASURES) + 1)]
    for method, label in METHODS.items():
        a = acc[method]
        lines.append(f"| {label} | " + " | ".join(f"{a[m]['mean_miss']:.1f}" for m in MEASURES)
                     + f" | {a['target_units']['median_relative_miss']:.0%} |")
    lines += ["", f"Changed products' units are scored on the {acc['twin']['target_units']['scenarios']} price and "
              f"promotion scenarios (a stock-out always takes them to -100%, which every method gets right). "
              f"\"Typical miss vs the true change\" is the median of miss / size of the true change: 10% means an "
              f"answer of about -9% or -11% when the truth is -10%. The true answer itself has a small margin of "
              f"error from the random runs (median standard error {report['truth_noise']['target_units']:.2f} "
              f"points).", ""]

    lines += ["### The five standard questions", "",
              "Changed products' units (for the stock-out: the whole category's units), true vs predicted:", "",
              "| Scenario | Truth | Twin | Without personal traits | Store-wide elasticity | No reaction |",
              "|---|---|---|---|---|---|"]
    for r in report["scenarios"][:5]:
        m = "category_units" if r["kind"] == "stock-out" else "target_units"
        p = r["predictions"]
        lines.append(f"| {r['name']} | {r['truth'][m]['pct']:+.1f}% | " + " | ".join(
            f"{p[k][m]:+.1f}%" for k in METHODS) + " |")
    lines += ["", "### Is the twin's range honest?", "",
              "The 90% bands in the what-if report come from resampling shoppers, so they only show how much the "
              "answer depends on who is in the store. Against the truth:", "",
              "| | Band contains the truth | 90% of misses are under | ... or as a share of the true change |",
              "|---|---|---|---|"]
    for m, label in MEASURES.items():
        b = bands[m]
        # a share of the true change only means something when the change is big, as for the changed products
        rel = f"{b['relative_miss_90th_percentile']:.0%}" if m == "target_units" else "-"
        lines.append(f"| {label} | {b['band_contains_truth']:.0%} of {b['scenarios']} | "
                     f"{b['miss_90th_percentile']:.1f} points | {rel} |")
    lines += ["", "So quote a twin answer with the error it really has (the last two columns), not with the narrow "
              "band.", ""]

    lines += ["### Does it know which shoppers react?", "",
              f"The shoppers split into four equal groups by their TRUE price sensitivity (from the answer key). "
              f"Over the {seg['scenarios']} price and promotion scenarios:", "",
              "| | Average miss per group (points) | Gap between the most and least sensitive group |",
              "|---|---|---|",
              f"| Truth | | {seg['gap_highest_vs_lowest']['truth']:.1f} points |",
              f"| Twin | {seg['twin']['mean_miss']:.1f} | {seg['gap_highest_vs_lowest']['twin']:.1f} points |",
              f"| Twin without personal traits | {seg['pooled']['mean_miss']:.1f} | "
              f"{seg['gap_highest_vs_lowest']['pooled']:.1f} points |", ""]
    coffee = report["scenarios"][0]
    if coffee["kind"] != "stock-out":
        s = coffee["segments"]
        lines += [f"{coffee['name']}, change in the changed products' units by true price sensitivity:", "",
                  "| | " + " | ".join(QUARTERS) + " |", "|---|" + "---|" * len(QUARTERS)]
        for who, label in (("truth", "Truth"), ("twin", "Twin"), ("pooled", "Without personal traits")):
            lines.append(f"| {label} | " + " | ".join(f"{s[who][q]:+.1f}%" for q in QUARTERS) + " |")
        lines.append("")

    t = report["traits"]
    lines += ["### Did it learn each shopper's traits?", "",
              "Rank correlation between learned and true values over all shoppers (1 = same order, 0 = no "
              "relation; 90% range in brackets). \"Profile only\" is what the public profile (age band and "
              "household size) gives on its own.", "",
              "| Trait | Twin | Profile only | Least history | Most history | Median learned vs true |",
              "|---|---|---|---|---|---|"]
    for label, v in t.items():
        r, lo, hi = v["twin"]["r"], *v["twin"]["range"]
        med = (f"{v['median']['learned']:.2f} vs {v['median']['true']:.2f}" if v["same_scale"]
               else "different scale")
        lines.append(f"| {label} | {r:.2f} ({lo:.2f}-{hi:.2f}) | {v['group_average_only']['r']:.2f} | "
                     f"{v['by_history']['least']['r']:.2f} | {v['by_history']['most']['r']:.2f} | {med} |")
    tr = report["taste"]
    lines += ["", f"Taste: within each category, the twin orders products by appeal with an average rank correlation "
              f"of {tr['twin']:.2f} with each shopper's true order ({tr['shoppers']:,} shoppers sampled), against "
              f"{tr['popularity']:.2f} for the same popularity order for everyone.", ""]

    c = report["calibration"]
    lines += ["### Are its purchase chances right?", "",
              f"Every shopper x product ({c['pairs']:,} pairs), chance of buying in the window vs what happened "
              f"({c['pairs_bought']:,} pairs bought):", "",
              "| Forecast | Pairs bought (predicted) | Brier score | Log loss | Better than own history by |",
              "|---|---|---|---|---|"]
    ref = c["methods"]["own history"]
    for name, v in c["methods"].items():
        lines.append(f"| {name.capitalize()} | {v['pairs_bought_predicted']:,.0f} | {v['brier']:.6f} | "
                     f"{v['logloss']:.5f} | {1 - v['brier'] / ref['brier']:+.1%} (Brier), "
                     f"{1 - v['logloss'] / ref['logloss']:+.1%} (log loss) |")
    lines += ["", "Twin, by predicted chance:", "", "| Predicted chance | Pairs | Average predicted | Actually bought |",
              "|---|---|---|---|"]
    for b in c["methods"]["twin"]["bins"]:
        lines.append(f"| {b['from']:.1%}-{b['to']:.1%} | {b['pairs']:,} | {b['predicted']:.2%} | "
                     f"{b['actual']:.2%} |")

    worst = sorted(report["scenarios"], key=lambda r: -abs(r["predictions"]["twin"]["category_units"]
                                                           - r["truth"]["category_units"]["pct"]))[:3]
    sp = report["speed"]
    lines += ["", "### Where it misses most (whole category's units)", ""]
    for r in worst:
        lines.append(f"- {r['name']}: truth {r['truth']['category_units']['pct']:+.1f}%, twin "
                     f"{r['predictions']['twin']['category_units']:+.1f}%")
    lines += ["", f"Speed: the twin answers a scenario in {sp['twin_seconds_per_scenario']:.0f}s for "
              f"{sp['shoppers']:,} shoppers; one run of the true store takes {sp['true_store_seconds_per_run']:.0f}s "
              f"(and only exists because the store is simulated).", ""]
    return "\n".join(lines)


def chart(report: dict, truth_shoppers: pd.DataFrame, twin, path: Path) -> None:
    blue, orange, grey = "#2a78d6", "#e8702a", "#9a9892"
    ink, ink2, grid_c, bg = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    colours = {"twin": orange, "pooled": "#f2b27f", "elasticity": blue, "no_reaction": grey}
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), facecolor=bg)

    ax = axes[0, 0]
    x = np.arange(len(MEASURES))
    for i, (method, label) in enumerate(METHODS.items()):
        vals = [report["accuracy"][method][m]["mean_miss"] for m in MEASURES]
        bars = ax.bar(x + (i - 1.5) * 0.2, vals, width=0.2, color=colours[method], label=label, zorder=3)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8, color=ink)
    ax.set_xticks(x, list(MEASURES.values()))
    ax.set_ylabel("Average miss (percentage points)", color=ink2)
    ax.legend(frameon=False, fontsize=9)
    ax.set_title("How far off each method is", loc="left", fontweight="bold", fontsize=12, color=ink)

    ax = axes[0, 1]
    use = _scored(report["scenarios"], "target_units")
    true = np.array([r["truth"]["target_units"]["pct"] for r in use])
    for method in ("elasticity", "twin"):
        pred = np.array([r["predictions"][method]["target_units"] for r in use])
        ax.scatter(true, pred, s=28, color=colours[method], label=METHODS[method], zorder=3)
    lim = [min(true.min(), -40) * 1.2, max(true.max(), 40) * 1.2]
    ax.plot(lim, lim, color=ink2, lw=0.8, zorder=2)
    ax.set_xscale("symlog", linthresh=10)
    ax.set_yscale("symlog", linthresh=10)
    fmt = FuncFormatter(lambda v, _: f"{v:+.0f}%" if v else "0%")
    ax.xaxis.set_major_formatter(fmt)
    ax.yaxis.set_major_formatter(fmt)
    ax.set_xlabel("True change in the changed products' units", color=ink2)
    ax.set_ylabel("Predicted change", color=ink2)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Each price or promotion scenario", loc="left", fontweight="bold", fontsize=12, color=ink)

    ax = axes[1, 0]
    rng = np.random.default_rng(0)
    pick = rng.choice(len(truth_shoppers), size=min(2000, len(truth_shoppers)), replace=False)
    learned = twin.shopper_traits().price_sens.to_numpy()
    ax.scatter(truth_shoppers.price_sensitivity.to_numpy()[pick], learned[pick], s=6, alpha=0.4, color=orange,
               zorder=3)
    top = max(truth_shoppers.price_sensitivity.max(), learned.max())
    ax.plot([0, top], [0, top], color=ink2, lw=0.8, zorder=2)
    r = report["traits"]["price sensitivity"]["twin"]["r"]
    ax.set_xlabel("True price sensitivity (answer key)", color=ink2)
    ax.set_ylabel("Learned by the twin", color=ink2)
    ax.set_title(f"Price sensitivity per shopper (rank correlation {r:.2f})", loc="left", fontweight="bold",
                 fontsize=12, color=ink)

    ax = axes[1, 1]
    for name, colour in (("own history", blue), ("twin", orange)):
        bins = report["calibration"]["methods"][name]["bins"]
        ax.plot([b["predicted"] for b in bins], [b["actual"] for b in bins], "o-", color=colour, zorder=3,
                label="Twin" if name == "twin" else "Own-history forecast")
    ax.plot([1e-4, 1], [1e-4, 1], color=ink2, lw=0.8, zorder=2, label="Perfect")
    ax.set_xscale("log")
    ax.set_yscale("log")
    pct = FuncFormatter(lambda v, _: f"{v:.2%}" if v < 0.001 else f"{v:.1%}" if v < 0.01 else f"{v:.0%}")
    ax.xaxis.set_major_formatter(pct)
    ax.yaxis.set_major_formatter(pct)
    ax.set_xlabel("Predicted chance of buying a product", color=ink2)
    ax.set_ylabel("Share actually bought", color=ink2)
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    ax.set_title("Purchase chances: predicted vs actual", loc="left", fontweight="bold", fontsize=12, color=ink)

    for ax in axes.ravel():
        ax.set_facecolor(bg)
        ax.grid(color=grid_c, zorder=0)
        ax.tick_params(colors=ink2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid_c)
    w = report["window"]
    fig.suptitle(f"The shopper twin scored against the true store, days {w['first_day']}-{w['last_day']}", x=0.01,
                 ha="left", fontweight="bold", fontsize=13, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":
    main()
