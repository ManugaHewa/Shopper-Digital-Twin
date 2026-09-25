"""Generate a complete world and save it to disk.

Layout of a saved world:

    data/worlds/<name>/
      manifest.json                 settings, seed, row counts
      public/                       what a real store would have (models may read this)
        products.parquet
        categories.parquet
        shoppers.parquet
        price_history.parquet
        sessions.parquet
        events.parquet
      answer_key/                   the hidden truth (only evaluation may read this)
        rules.json
        shoppers_truth.parquet
        products_truth.parquet
        categories_truth.parquet
        brands_truth.parquet
        life_events.parquet
        arrays.npz                  taste vectors, product styles, category interests
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from . import spec
from .catalog import Catalog, PriceSchedule, build_catalog, build_price_schedule
from .config import WorldConfig
from .population import Population, build_population
from .simulate import EVENT_TYPES, Simulator


def build_world(cfg: WorldConfig) -> tuple[Catalog, Population, PriceSchedule, Simulator]:
    """Create the catalogue, shoppers and prices (no simulation yet). Same seed, same world."""
    catalog_seed, population_seed, price_seed = np.random.SeedSequence(cfg.seed).spawn(3)
    catalog = build_catalog(cfg, np.random.default_rng(catalog_seed))
    population = build_population(cfg, catalog, np.random.default_rng(population_seed))
    prices = build_price_schedule(cfg, catalog, np.random.default_rng(price_seed))
    return catalog, population, prices, Simulator(cfg, catalog, population, prices)


EVENT_SCHEMA = pa.schema([
    ("session_id", pa.int64()),
    ("shopper_id", pa.int32()),
    ("timestamp", pa.timestamp("s")),
    ("product_id", pa.int32()),
    ("event_type", pa.dictionary(pa.int8(), pa.string())),
    ("quantity", pa.int8()),
    ("price", pa.float32()),
    ("on_promo", pa.bool_()),
])


def generate_world(cfg: WorldConfig, root: str | Path = "data/worlds", verbose: bool = True) -> Path:
    t0 = time.time()
    out = Path(root) / cfg.name
    (out / "public").mkdir(parents=True, exist_ok=True)
    (out / "answer_key").mkdir(parents=True, exist_ok=True)

    catalog, population, prices, sim = build_world(cfg)
    start = np.datetime64(cfg.start_date, "s")
    event_dict = pa.array(EVENT_TYPES)

    n_events = {t: 0 for t in EVENT_TYPES}
    sessions = []
    revenue = 0.0
    with pq.ParquetWriter(out / "public" / "events.parquet", EVENT_SCHEMA, compression="zstd") as writer:
        for log in sim.run():
            table = pa.table({
                "session_id": log.session_id,
                "shopper_id": log.shopper_id,
                "timestamp": start + log.ts.astype("timedelta64[s]"),
                "product_id": log.product_id,
                "event_type": pa.DictionaryArray.from_arrays(pa.array(log.event_type, pa.int8()), event_dict),
                "quantity": log.quantity,
                "price": log.price,
                "on_promo": log.on_promo,
            }, schema=EVENT_SCHEMA)
            writer.write_table(table)

            counts = np.bincount(log.event_type, minlength=3)
            for t, c in zip(EVENT_TYPES, counts):
                n_events[t] += int(c)
            is_buy = log.event_type == 2
            spend = log.price * log.quantity * is_buy
            revenue += float(spend.sum())
            local = log.session_id - (log.s_session_id[0] if len(log.s_session_id) else 0)
            n_s = len(log.s_session_id)
            sessions.append(pd.DataFrame({
                "session_id": log.s_session_id,
                "shopper_id": log.s_shopper_id,
                "start_time": start + log.s_start_ts.astype("timedelta64[s]"),
                "n_views": np.bincount(local, weights=log.event_type == 0, minlength=n_s).astype(np.int32),
                "n_purchases": np.bincount(local, weights=is_buy, minlength=n_s).astype(np.int32),
                "revenue": np.round(np.bincount(local, weights=spend, minlength=n_s), 2),
            }))
            if verbose and (log.day + 1) % 30 == 0:
                print(f"  day {log.day + 1}/{cfg.n_days}: {sum(n_events.values()):,} events so far")

    sessions_df = pd.concat(sessions, ignore_index=True)
    sessions_df.to_parquet(out / "public" / "sessions.parquet", index=False)
    _write_public(out / "public", cfg, catalog, population, prices)
    _write_answer_key(out / "answer_key", cfg, catalog, population, sim)

    manifest = {
        "name": cfg.name,
        "created_seconds": round(time.time() - t0, 1),
        "config": cfg.to_dict(),
        "counts": {
            "products": catalog.n_products,
            "categories": catalog.n_categories,
            "shoppers": population.n,
            "sessions": len(sessions_df),
            "events": sum(n_events.values()),
            **{f"events_{t}": c for t, c in n_events.items()},
            "life_events": len(population.event_shopper),
        },
        "revenue": round(revenue, 2),
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    if verbose:
        print(f"Saved world '{cfg.name}' to {out} in {manifest['created_seconds']}s")
        print(json.dumps(manifest["counts"], indent=2))
    return out


def _write_public(out: Path, cfg: WorldConfig, cat: Catalog, pop: Population, prices: PriceSchedule) -> None:
    pd.DataFrame({
        "product_id": np.arange(cat.n_products),
        "name": cat.name,
        "department": np.array(spec.DEPARTMENTS)[cat.department],
        "category_id": cat.category,
        "category": cat.cat_name[cat.category],
        "brand": cat.brand_name[cat.brand],
        "is_store_brand": cat.is_store,
        "base_price": cat.base_price,
        "rating": cat.rating,
        "n_reviews": cat.n_reviews,
    }).to_parquet(out / "products.parquet", index=False)

    pd.DataFrame({
        "category_id": np.arange(cat.n_categories),
        "category": cat.cat_name,
        "department": np.array(spec.DEPARTMENTS)[cat.cat_dept],
    }).to_parquet(out / "categories.parquet", index=False)

    household_band = np.select([pop.household_size == 1, pop.household_size == 2, pop.household_size <= 4],
                               ["1", "2", "3-4"], "5+")
    pd.DataFrame({
        "shopper_id": np.arange(pop.n),
        "region": pop.region,
        "age_band": pop.age_band,
        "household_band": household_band,
        "signup_date": pop.signup_date,
    }).to_parquet(out / "shoppers.parquet", index=False)

    n, w = prices.regular.shape
    pd.DataFrame({
        "product_id": np.repeat(np.arange(n), w),
        "week": np.tile(np.arange(w), n),
        "week_start": np.tile(np.datetime64(cfg.start_date) + 7 * np.arange(w).astype("timedelta64[D]"), n),
        "regular_price": prices.regular.ravel(),
        "promo_discount": prices.discount.ravel(),
        "price": prices.price.ravel(),
    }).to_parquet(out / "price_history.parquet", index=False)


def _write_answer_key(out: Path, cfg: WorldConfig, cat: Catalog, pop: Population, sim: Simulator) -> None:
    (out / "rules.json").write_text(json.dumps(asdict(cfg.rules), indent=2))
    pd.DataFrame({
        "shopper_id": np.arange(pop.n),
        "segment": np.array([s.name for s in spec.SEGMENTS])[pop.segment],
        "price_sensitivity": pop.price_sensitivity,
        "quality_weight": pop.quality_weight,
        "store_brand_affinity": pop.store_brand_affinity,
        "brand_loyalty": pop.brand_loyalty,
        "habit_strength": pop.habit_strength,
        "visits_per_week": pop.visits_per_week,
        "discretionary_per_visit": pop.discretionary_per_visit,
        "household_size": pop.household_size,
        "has_dog": pop.has_dog,
        "has_cat": pop.has_cat,
        "has_baby": pop.has_baby,
        "has_car": pop.has_car,
        "shop_hour": pop.shop_hour,
    }).to_parquet(out / "shoppers_truth.parquet", index=False)
    pd.DataFrame({
        "product_id": np.arange(cat.n_products),
        "quality": cat.quality,
        "popularity": cat.popularity,
    }).to_parquet(out / "products_truth.parquet", index=False)
    pd.DataFrame({
        "category_id": np.arange(cat.n_categories),
        "category": cat.cat_name,
        "staple": cat.cat_staple,
        "cycle_days": cat.cat_cycle,
        "median_price": cat.cat_median,
        "peak_day": cat.cat_peak,
        "season_amp": cat.cat_amp,
        "requires": cat.cat_requires,
    }).to_parquet(out / "categories_truth.parquet", index=False)
    pd.DataFrame({
        "brand_id": np.arange(len(cat.brand_name)),
        "brand": cat.brand_name,
        "department": np.array(spec.DEPARTMENTS)[cat.brand_dept],
        "is_store_brand": cat.brand_is_store,
        "quality": cat.brand_quality,
        "price_premium": cat.brand_premium,
    }).to_parquet(out / "brands_truth.parquet", index=False)
    pd.DataFrame({
        "shopper_id": pop.event_shopper,
        "event": pop.event_type,
        "day": pop.event_day,
        "date": np.datetime64(cfg.start_date) + pop.event_day.astype("timedelta64[D]"),
    }).sort_values("day").to_parquet(out / "life_events.parquet", index=False)
    np.savez_compressed(
        out / "arrays.npz",
        taste_start=sim.start_taste,
        taste_end=sim.final_taste,
        product_style=cat.style,
        category_center=cat.cat_center,
        fav_brand=pop.fav_brand,
        staple_uses=pop.staple_uses,
        staple_cycle=pop.staple_cycle,
        disc_weight=pop.disc_weight,
    )
