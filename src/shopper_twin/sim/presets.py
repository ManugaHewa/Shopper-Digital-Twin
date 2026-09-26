"""The standard what-if questions used in the reports, the evaluation and the demo app."""

from __future__ import annotations

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
