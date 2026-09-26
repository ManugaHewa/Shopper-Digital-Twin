"""Ask the fitted twin what-if questions about the test window (days 152-179 on the small world).

    python scripts/run_whatif.py --world data/worlds/small

Needs the twin saved by scripts/run_twin.py (models/<world>/twin_day<cutoff>.pt). Two parts:

  Volume check  With the store's own price plan, does the twin predict how many units each category really sold
                in the window? Compared with a simple forecast: "the same as the last 4 weeks".
  Scenarios     Five changes (a price rise, a price cut, a stock-out, a promotion, a store-wide change), each
                compared with the store's own plan: units of the changed products and of their whole
                categories, revenue, where the lost (or won) sales go, and which shoppers react most.

The twin is used with the store's price plan for the window (a store always knows its own plan). Writes
reports/<world>/whatif.json, whatif.md and whatif.png.
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
from shopper_twin.sim import WhatIf, preset_scenarios  # noqa: E402
from shopper_twin.twin import ShopperTwin  # noqa: E402

HORIZON = 28


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--world", default="data/worlds/small")
    parser.add_argument("--out", default=None, help="defaults to reports/<world name>")
    parser.add_argument("--models", default="models", help="folder with the saved twin")
    parser.add_argument("--bootstrap", type=int, default=200, help="resamples of the shoppers for the 90%% bands")
    args = parser.parse_args()
    t_start = time.perf_counter()

    world = load_public(args.world)
    out = Path(args.out or Path("reports") / world.path.name)
    out.mkdir(parents=True, exist_ok=True)
    cutoff = world.n_days - HORIZON
    model_path = Path(args.models) / world.path.name / f"twin_day{cutoff}.pt"
    if not model_path.exists():
        raise SystemExit(f"{model_path} not found: run scripts/run_twin.py --world {args.world} first.")
    twin = ShopperTwin.load(model_path, world, planned_prices=True)
    whatif = WhatIf(twin, world)
    purchases = world.events(types=("purchase",), extra_columns=("quantity",))
    print(f"Twin {model_path.as_posix()}: trained on days 0-{cutoff - 1}, predicting days {cutoff}-"
          f"{cutoff + HORIZON - 1} for {twin.n_shoppers:,} shoppers")

    t0 = time.perf_counter()
    volume = volume_check(whatif, world, purchases, cutoff)
    print(f"Volume check ({time.perf_counter() - t0:.0f}s): predicted {volume['total']['twin']:,.0f} units, "
          f"actual {volume['total']['actual']:,.0f}; category error {volume['wape']['twin']:.1%} (twin) vs "
          f"{volume['wape']['last_4_weeks']:.1%} (same as the last 4 weeks)")

    results, tables = [], {}
    for scenario in preset_scenarios(world, purchases, cutoff):
        t0 = time.perf_counter()
        r = whatif.run(scenario, n_boot=args.bootstrap)
        tables[r["name"]] = r.pop("products")
        r["seconds"] = round(time.perf_counter() - t0, 1)
        results.append(r)
        print(f"  {r['name']}: targeted units {r['target_units']['pct']:+.1f}%, category units "
              f"{r['category_units']['pct']:+.1f}%, category revenue {r['category_revenue']['pct']:+.1f}% "
              f"({r['seconds']:.0f}s)")

    report = {
        "world": world.path.name,
        "twin": model_path.as_posix(),
        "window": {"first_day": cutoff, "last_day": cutoff + HORIZON - 1,
                   "weeks": [int(w) for w in whatif.weeks]},
        "volume_check": volume,
        "scenarios": results,
        "runtime_seconds": round(time.perf_counter() - t_start, 1),
    }
    (out / "whatif.json").write_text(json.dumps(report, indent=2, default=_plain), encoding="utf-8")
    text = markdown(report)
    (out / "whatif.md").write_text(text, encoding="utf-8")
    chart(report, out / "whatif.png")
    print()
    print(text)
    print(f"Saved to {out.as_posix()}/ (total {report['runtime_seconds'] / 60:.1f} min)")


def volume_check(whatif: WhatIf, world, purchases: pd.DataFrame, cutoff: int) -> dict:
    """Units per category in the window: the twin (store's own plan) vs what really sold vs the last 4 weeks."""
    cats = world.categories.sort_values("category_id")
    category_of = whatif.twin.cat.category[purchases.product_id.to_numpy()]

    def units(first: int, last: int) -> np.ndarray:
        m = ((purchases.day >= first) & (purchases.day <= last)).to_numpy()
        return np.bincount(category_of[m], weights=purchases.quantity.to_numpy()[m], minlength=len(cats))

    actual = units(cutoff, cutoff + HORIZON - 1)
    naive = units(cutoff - HORIZON, cutoff - 1)
    twin = whatif.category_units().units.to_numpy()
    table = pd.DataFrame({"category": cats.category.to_numpy(), "department": cats.department.to_numpy(),
                          "twin": twin, "actual": actual, "last_4_weeks": naive})
    table["twin_error_pct"] = 100 * (table.twin / table.actual - 1)

    def wape(pred: np.ndarray) -> float:
        return float(np.abs(pred - actual).sum() / actual.sum())

    return {
        "total": {"twin": float(twin.sum()), "actual": float(actual.sum()), "last_4_weeks": float(naive.sum())},
        "wape": {"twin": wape(twin), "last_4_weeks": wape(naive)},
        "correlation": {"twin": float(np.corrcoef(np.log1p(twin), np.log1p(actual))[0, 1]),
                        "last_4_weeks": float(np.corrcoef(np.log1p(naive), np.log1p(actual))[0, 1])},
        "categories": table.round(2).to_dict("records"),
    }


def _plain(x):
    if isinstance(x, (np.integer, np.floating)):
        return x.item()
    if isinstance(x, np.bool_):
        return bool(x)
    return str(x)


def _band(summary: dict) -> str:
    b = summary["band_pct"]
    return f"{summary['pct']:+.1f}% ({b[0]:+.1f} to {b[1]:+.1f})" if b else f"{summary['pct']:+.1f}%"


def _where_it_went(r: dict) -> str:
    s = r["substitution"]
    t, o, c = s["target_change"], s["other_products_change"], s["category_change"]
    if abs(t) < 1e-9:
        return "The changed products' sales don't move."
    if t < 0:
        lost = -t
        to_others = max(o, 0.0)
        return (f"Of the {lost:,.0f} units the changed products lose, {to_others:,.0f} "
                f"({to_others / lost:.0%}) move to other products in the same categories and "
                f"{max(-c, 0):,.0f} ({max(-c, 0) / lost:.0%}) aren't bought at all.")
    from_others = max(-o, 0.0)
    return (f"Of the {t:,.0f} extra units the changed products sell, {from_others:,.0f} ({from_others / t:.0%}) "
            f"are taken from other products in the same categories and {max(c, 0):,.0f} ({max(c, 0) / t:.0%}) "
            f"are extra purchases.")


def markdown(report: dict) -> str:
    v = report["volume_check"]
    w = report["window"]
    lines = [f"## What-if results from the twin (days {w['first_day']}-{w['last_day']}, "
             f"{len(report['scenarios'])} scenarios)", "",
             "Each scenario is compared with the store's own price plan for the same 4 weeks. Bands are 90% "
             "ranges from resampling the shoppers; they show how much the answer depends on which shoppers are in "
             "the store, not how wrong the twin might be (milestone 5 checks that against the true store).", "",
             "| Scenario | Price change | Changed products' units | Whole category's units | Category revenue |",
             "|---|---|---|---|---|"]
    for r in report["scenarios"]:
        price = "out of stock" if r["price_change_pct"] is None else f"{r['price_change_pct']:+.1f}%"
        lines.append(f"| {r['name']} | {price} | {_band(r['target_units'])} | {_band(r['category_units'])} | "
                     f"{_band(r['category_revenue'])} |")
    lines.append("")
    for r in report["scenarios"]:
        where = ", ".join(r["categories"]) if len(r["categories"]) <= 4 else f"{len(r['categories'])} categories"
        lines += [f"### {r['name']}", "",
                  f"{r['description'][0].upper() + r['description'][1:]}: {r['n_target_products']} "
                  f"product{'s' if r['n_target_products'] != 1 else ''} in "
                  f"{where}. "
                  f"Expected units of the changed products: {r['target_units']['base']:,.0f} -> "
                  f"{r['target_units']['new']:,.0f}; shoppers buying them at least once: "
                  f"{r['target_buyers']['base']:,.0f} -> {r['target_buyers']['new']:,.0f}.", "",
                  _where_it_went(r), ""]
        moved = r["substitution"]["most_affected_other_products"]
        if moved:
            lines.append("Other products that move most: " + "; ".join(
                f"{m['name']} {m['change']:+,.0f}" for m in moved[:3]) + ".")
            lines.append("")
        seg = r["by_segment"]
        measure = "the changed products' units" if r["segments_measure"] == "target" else "the category's units"
        lines.append(f"Change in {measure} by household size: " + ", ".join(
            f"{k} {v:+.1f}%" for k, v in seg["household size"].items()) + ". By learned price sensitivity: " +
            ", ".join(f"{k} {v:+.1f}%" for k, v in seg["price sensitivity (learned)"].items()) + ".")
        lines.append("")
    t = v["total"]
    lines += ["### Volume check: does the twin get the store's normal sales right?", "",
              "With the store's own price plan and no changes, units sold per category in the window:", "",
              "| | Total units | Error per category (WAPE) | Correlation with actual (log) |",
              "|---|---|---|---|",
              f"| Actual | {t['actual']:,.0f} | | |",
              f"| Twin | {t['twin']:,.0f} ({t['twin'] / t['actual'] - 1:+.1%}) | {v['wape']['twin']:.1%} | "
              f"{v['correlation']['twin']:.3f} |",
              f"| Same as the last 4 weeks | {t['last_4_weeks']:,.0f} ({t['last_4_weeks'] / t['actual'] - 1:+.1%}) | "
              f"{v['wape']['last_4_weeks']:.1%} | {v['correlation']['last_4_weeks']:.3f} |", ""]
    cats = pd.DataFrame(v["categories"])
    over = cats[cats.actual >= 200].nlargest(3, "twin_error_pct")
    under = cats[cats.actual >= 200].nsmallest(3, "twin_error_pct")
    lines += ["Largest misses (categories with at least 200 units): over "
              + ", ".join(f"{r.category} {r.twin_error_pct:+.0f}%" for r in over.itertuples())
              + "; under " + ", ".join(f"{r.category} {r.twin_error_pct:+.0f}%" for r in under.itertuples()) + ".",
              ""]
    return "\n".join(lines)


def chart(report: dict, path: Path) -> None:
    blue, orange, ink, ink2, grid_c, bg = "#2a78d6", "#e8702a", "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), facecolor=bg, gridspec_kw={"width_ratios": [1.5, 1]})

    ax = axes[0]
    rows = report["scenarios"][::-1]
    y = np.arange(len(rows))
    for offset, key, colour, label in ((0.2, "target_units", orange, "Changed products"),
                                        (-0.2, "category_units", blue, "Whole category")):
        vals = [r[key]["pct"] for r in rows]
        err = np.array([[r[key]["pct"] - r[key]["band_pct"][0], r[key]["band_pct"][1] - r[key]["pct"]]
                        for r in rows]).T
        bars = ax.barh(y + offset, vals, height=0.38, color=colour, label=label, zorder=3, xerr=err,
                       error_kw={"ecolor": ink2, "capsize": 2, "lw": 0.8})
        ax.bar_label(bars, fmt="%+.1f%%", padding=4, fontsize=8, color=ink)
    ax.axvline(0, color=ink2, lw=0.8)
    ax.set_yticks(y, [r["name"].split(" (")[0] for r in rows])  # without the product name in brackets
    ax.set_xlabel("Change in expected units vs the store's own plan", color=ink2)
    lo = min(min(r["target_units"]["pct"], r["category_units"]["pct"]) for r in rows)
    hi = max(max(r["target_units"]["pct"], r["category_units"]["pct"]) for r in rows)
    ax.set_xlim(min(lo * 1.5, -8), max(hi * 1.3, 8))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:+.0f}%" if x else "0%"))
    ax.legend(frameon=False, loc="lower right", fontsize=9)
    ax.set_title("What the twin predicts", loc="left", fontweight="bold", fontsize=12, color=ink)

    ax = axes[1]
    cats = pd.DataFrame(report["volume_check"]["categories"])
    ax.scatter(cats.actual, cats.last_4_weeks, s=14, color=blue, alpha=0.6, label="Same as last 4 weeks", zorder=3)
    ax.scatter(cats.actual, cats.twin, s=14, color=orange, label="Twin", zorder=4)
    top = cats[["actual", "twin", "last_4_weeks"]].to_numpy().max() * 1.3
    ax.plot([10, top], [10, top], color=ink2, lw=0.8, zorder=2)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("Units actually sold per category", color=ink2)
    ax.set_ylabel("Units predicted", color=ink2)
    ax.legend(frameon=False, loc="upper left", fontsize=9)
    ax.set_title("Volume check: normal sales", loc="left", fontweight="bold", fontsize=12, color=ink)

    for ax in axes:
        ax.set_facecolor(bg)
        ax.grid(color=grid_c, zorder=0)
        ax.tick_params(colors=ink2)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        for side in ("left", "bottom"):
            ax.spines[side].set_color(grid_c)
    w = report["window"]
    fig.suptitle(f"What-if questions for the shopper twin, days {w['first_day']}-{w['last_day']}", x=0.01,
                 ha="left", fontweight="bold", fontsize=13, color=ink)
    fig.tight_layout()
    fig.savefig(path, dpi=150, facecolor=bg)
    plt.close(fig)


if __name__ == "__main__":
    main()
