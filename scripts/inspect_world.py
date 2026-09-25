"""Sanity-check a generated world: does it behave like a real store?

    python scripts/inspect_world.py --world data/worlds/small

Prints a summary and saves charts to reports/<world name>/. This script is allowed
to read the answer key, because its job is to check the generator itself.
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

BLUE, ORANGE = "#2a78d6", "#eb6834"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
BLUES = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "figure.facecolor": "#fcfcfb", "axes.facecolor": "#fcfcfb", "savefig.facecolor": "#fcfcfb",
    "axes.edgecolor": GRID, "axes.labelcolor": INK_2, "xtick.color": INK_2, "ytick.color": INK_2,
    "text.color": INK, "axes.titleweight": "bold", "axes.titlesize": 12, "axes.titlelocation": "left",
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.8, "axes.spines.top": False,
    "axes.spines.right": False, "font.size": 10,
})


def load(world: Path) -> dict:
    pub, key = world / "public", world / "answer_key"
    products = pd.read_parquet(pub / "products.parquet")
    buys = pd.read_parquet(pub / "events.parquet", filters=[("event_type", "=", "purchase")],
                           columns=["shopper_id", "timestamp", "product_id", "quantity", "price", "on_promo"])
    buys = buys.merge(products[["product_id", "category_id", "department", "is_store_brand"]], on="product_id")
    truth = pd.read_parquet(key / "shoppers_truth.parquet")
    buys = buys.merge(truth[["shopper_id", "segment", "price_sensitivity"]], on="shopper_id")
    return {
        "buys": buys,
        "sessions": pd.read_parquet(pub / "sessions.parquet"),
        "cats": pd.read_parquet(key / "categories_truth.parquet"),
        "life": pd.read_parquet(key / "life_events.parquet"),
        "manifest": json.loads((world / "manifest.json").read_text()),
    }


def daily_revenue(d, out):
    s = d["sessions"].assign(date=lambda x: x.start_time.dt.floor("D")).groupby("date")["revenue"].sum()
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.plot(s.index, s.values / 1000, color="#b7d3f6", lw=1.5, label="Daily")
    ax.plot(s.index, s.rolling(7, center=True).mean() / 1000, color=BLUE, lw=2, label="7-day average")
    ax.set_title("Daily revenue: weekend peaks and seasonal drift")
    ax.set_ylabel("Revenue ($ thousands)")
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(out / "daily_revenue.png", dpi=150)
    plt.close(fig)
    weekday = s.groupby(s.index.dayofweek).mean()
    return {"weekend_vs_weekday_revenue": round(weekday[[5, 6]].mean() / weekday[[0, 1, 2, 3]].mean(), 2)}


def repurchase_cycles(d, out):
    b = d["buys"].sort_values("timestamp")
    b["gap"] = b.groupby(["shopper_id", "category_id"])["timestamp"].diff().dt.days
    cats = d["cats"].set_index("category_id")
    g = b.groupby("category_id")["gap"].median().to_frame("observed").join(cats)
    st = g[g.staple].dropna()
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.scatter(st.cycle_days, st.observed, s=36, color=BLUE, edgecolor="#fcfcfb", linewidth=1.5, zorder=3)
    lim = [4, 150]
    ax.plot(lim, lim, color=INK_2, lw=1, ls="--", zorder=2)
    ax.text(30, 22, "observed = designed", color=INK_2, fontsize=9, ha="left", rotation=0)
    for name in ("Milk", "Coffee", "Laundry Detergent", "Vitamins", "Motor Oil"):
        row = st[st.category == name]
        if len(row):
            ax.annotate(name, (row.cycle_days.iloc[0], row.observed.iloc[0]), xytext=(6, -3),
                        textcoords="offset points", fontsize=8.5, color=INK)
    ax.set(xscale="log", yscale="log", xlim=lim, ylim=lim,
           xlabel="Designed repurchase cycle (days)", ylabel="Observed median days between purchases")
    ax.set_title("Staples are re-bought on their cycles")
    fig.tight_layout()
    fig.savefig(out / "repurchase_cycles.png", dpi=150)
    plt.close(fig)
    # same-product repeat rate for staples (habit / loyalty)
    b["prev"] = b.groupby(["shopper_id", "category_id"])["product_id"].shift()
    rep = b[b.category_id.isin(st.index) & b.prev.notna()]
    corr = np.corrcoef(np.log(st.cycle_days), np.log(st.observed))[0, 1]
    return {"staple_cycle_log_correlation": round(float(corr), 3),
            "staple_same_product_repeat_rate": round(float((rep.product_id == rep.prev).mean()), 3),
            "discretionary_median_gap_days": float(g[~g.staple].observed.median())}


def price_sensitivity(d, out):
    b = d["buys"]
    labels = ["Least", "Low-mid", "High-mid", "Most"]
    b = b.assign(q=pd.qcut(b.price_sensitivity, 4, labels=labels))
    t = b.groupby("q", observed=True).agg(store=("is_store_brand", "mean"), promo=("on_promo", "mean")) * 100
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.6), sharey=False)
    for ax, col, title in zip(axes, ["store", "promo"], ["Store-brand share of purchases", "Promo share of purchases"]):
        bars = ax.bar(labels, t[col], color=BLUE, width=0.6, zorder=3)
        ax.bar_label(bars, fmt="%.0f%%", padding=3, fontsize=9, color=INK)
        ax.set_title(title)
        ax.set_xlabel("True price sensitivity (quartile)")
        ax.set_ylim(0, t[col].max() * 1.2)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("% of purchases")
    fig.tight_layout()
    fig.savefig(out / "price_sensitivity.png", dpi=150)
    plt.close(fig)
    return {"store_brand_share_by_quartile": t["store"].round(1).tolist(),
            "promo_share_by_quartile": t["promo"].round(1).tolist()}


def segment_departments(d, out):
    b = d["buys"]
    share = pd.crosstab(b.department, b.segment, normalize="columns") * 100
    share = share.loc[share.mean(axis=1).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(9, 6))
    from matplotlib.colors import LinearSegmentedColormap
    cmap = LinearSegmentedColormap.from_list("blues", BLUES)
    # log-ish colour scale so small departments still show differences
    ax.imshow(np.log1p(share.values), cmap=cmap, aspect="auto")
    ax.set_xticks(range(share.shape[1]), [s.replace("_", "\n") for s in share.columns])
    ax.set_yticks(range(share.shape[0]), share.index)
    ax.grid(False)
    vmax = np.log1p(share.values).max()
    for i in range(share.shape[0]):
        for j in range(share.shape[1]):
            v = share.values[i, j]
            ax.text(j, i, f"{v:.1f}", ha="center", va="center", fontsize=8.5,
                    color="white" if np.log1p(v) > 0.55 * vmax else INK)
    ax.set_title("Where each (hidden) shopper segment spends: % of purchases")
    fig.tight_layout()
    fig.savefig(out / "segment_departments.png", dpi=150)
    plt.close(fig)
    return {}


def life_events(d, out):
    life = d["life"]
    babies = life[life.event == "new_baby"]
    if babies.empty:
        return {}
    b = d["buys"][d["buys"].department == "Baby"]
    b = b.merge(babies[["shopper_id", "date"]], on="shopper_id")
    b["week"] = (b.timestamp - b.date).dt.days // 7
    weeks = np.arange(-8, 9)
    n_days = d["manifest"]["config"]["n_days"]
    # only count shoppers whose whole week falls inside the simulated period
    covered = [((babies.day + 7 * w >= 0) & (babies.day + 7 * (w + 1) <= n_days)).sum() for w in weeks]
    weekly = b.groupby("week").size().reindex(weeks, fill_value=0) / np.maximum(covered, 1)
    fig, ax = plt.subplots(figsize=(7, 3.6))
    ax.plot(weekly.index, weekly.values, color=BLUE, lw=2, marker="o", ms=5)
    ax.axvline(0, color=INK_2, lw=1, ls="--")
    ax.text(0.25, weekly.max() * 0.5, "baby arrives", color=INK_2, fontsize=9)
    ax.set(xlabel="Weeks relative to the birth", ylabel="Baby purchases per shopper per week")
    ax.set_title("Life events: new parents start buying baby products")
    fig.tight_layout()
    fig.savefig(out / "life_events.png", dpi=150)
    plt.close(fig)
    return {"baby_purchases_per_week_before": round(float(weekly.loc[-8:-1].mean()), 3),
            "baby_purchases_per_week_after": round(float(weekly.loc[1:8].mean()), 3)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--world", default="data/worlds/small")
    parser.add_argument("--out", default=None, help="defaults to reports/<world name>")
    args = parser.parse_args()
    world = Path(args.world)
    out = Path(args.out or Path("reports") / world.name)
    out.mkdir(parents=True, exist_ok=True)

    d = load(world)
    s, b = d["sessions"], d["buys"]
    summary = {
        "counts": d["manifest"]["counts"],
        "visits_per_shopper_per_week": round(len(s) / s.shopper_id.nunique() / (d["manifest"]["config"]["n_days"] / 7), 2),
        "avg_basket_items": round(float(s.n_purchases.mean()), 2),
        "avg_basket_value": round(float(s.revenue.mean()), 2),
        "view_to_purchase_rate": round(float(s.n_purchases.sum() / s.n_views.sum()), 3),
        "grocery_share_of_purchases": round(float((b.department == "Grocery").mean()), 3),
    }
    for check in (daily_revenue, repurchase_cycles, price_sensitivity, segment_departments, life_events):
        summary.update(check(d, out))
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(f"Charts saved to {out}/")


if __name__ == "__main__":
    main()
