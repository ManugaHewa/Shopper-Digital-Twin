"""The standard what-if questions used in the reports, the evaluation and the demo app."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .scenario import PriceChange, Promotion, Scenario, Select, StockOut


def preset_scenarios(world, purchases: pd.DataFrame, before_day: int) -> list[Scenario]:
    """Five scenarios: a price rise, a price cut, a stock-out, a promotion and a store-wide change.

    `purchases` (product_id, day, and quantity if present) picks the best-selling cereal and the most popular
    snack brand, counting only days before `before_day` so nothing from the prediction window is used.
    """
    products = world.products
    past = purchases[purchases.day < before_day]
    units = (past.groupby("product_id").quantity.sum() if "quantity" in past.columns
             else past.groupby("product_id").size())
    units = units.reindex(products.product_id, fill_value=0)

    cereal = products[products.category == "Breakfast Cereal"]
    best_cereal_id = int(units[cereal.product_id].idxmax())
    best_cereal = products.set_index("product_id").loc[best_cereal_id, "name"]

    snacks = products[(products.category == "Chips & Snacks") & ~products.is_store_brand]
    brand_units = units[snacks.product_id].groupby(snacks.brand.to_numpy()).sum()
    top_brand = str(brand_units.idxmax())

    return [
        Scenario("Coffee +20%", [PriceChange(Select(category="Coffee"), percent=20)]),
        Scenario("Store-brand milk -15%", [PriceChange(Select(category="Milk", store_brand=True), percent=-15)]),
        Scenario(f"Best-selling cereal out of stock ({best_cereal})",
                 [StockOut(Select(product_ids=(best_cereal_id,)))]),
        Scenario(f"25% off {top_brand} snacks",
                 [Promotion(Select(category="Chips & Snacks", brand=top_brand), percent_off=25)]),
        Scenario("All store-brand products +10%", [PriceChange(Select(store_brand=True), percent=10)]),
    ]


def random_scenarios(world, purchases: pd.DataFrame, before_day: int, n: int = 24, seed: int = 0
                     ) -> list[Scenario]:
    """`n` varied scenarios for testing: whole-category price changes, one brand's price change, a brand on
    promotion, and a top seller out of stock (a quarter of each). Only categories that sold at least 300 units
    and brands that sold at least 150 in the 4 weeks before `before_day` are used, so every effect is big
    enough to measure in the true store (a brand selling a handful of units gives a very noisy true answer)."""
    rng = np.random.default_rng(seed)
    products = world.products.sort_values("product_id").reset_index(drop=True)
    recent = purchases[(purchases.day >= before_day - 28) & (purchases.day < before_day)]
    units = recent.groupby("product_id").quantity.sum().reindex(products.product_id, fill_value=0).to_numpy()
    by_cat = pd.Series(units).groupby(products.category.to_numpy()).sum()
    categories = sorted(by_cat[by_cat >= 300].index)
    branded = products.assign(units=units)[~products.is_store_brand.to_numpy()]
    brand_units = branded.groupby(["category", "brand"]).units.sum()
    brand_units = brand_units[brand_units >= 150]
    big_brands = {c: sorted(brand_units.loc[c].index) for c in categories if c in brand_units.index}

    scenarios = []
    for i in range(n):
        kind = i % 4
        category = str(rng.choice(sorted(big_brands) if kind in (1, 2) else categories))
        in_cat = products[products.category == category]
        brands = big_brands.get(category, [])
        if kind == 0:
            pct = int(rng.choice([-20, -10, 10, 20]))
            scenarios.append(Scenario(f"{category} {pct:+d}%", [PriceChange(Select(category=category), pct)]))
        elif kind == 1 and brands:
            brand, pct = str(rng.choice(brands)), int(rng.choice([-30, -15, 15, 30]))
            scenarios.append(Scenario(f"{brand} {category} {pct:+d}%",
                                      [PriceChange(Select(category=category, brand=brand), pct)]))
        elif kind == 2 and brands:
            brand, pct = str(rng.choice(brands)), int(rng.choice([15, 25, 35]))
            scenarios.append(Scenario(f"{pct}% off {brand} {category}",
                                      [Promotion(Select(category=category, brand=brand), pct)]))
        else:
            top = in_cat.product_id.to_numpy()[np.argsort(-units[in_cat.product_id.to_numpy()])[:3]]
            pid = int(rng.choice(top))
            scenarios.append(Scenario(f"{products.name[pid]} out of stock", [StockOut(Select(product_ids=(pid,)))]))
    return scenarios
