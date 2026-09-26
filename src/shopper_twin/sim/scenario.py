"""What-if scenarios: price changes, promotions and stock-outs, turned into the store's weekly prices.

A scenario is a list of changes, each applied to a selection of products:

    Scenario("Coffee +20%", [PriceChange(Select(category="Coffee"), percent=20)])
    Scenario("Store-brand milk -15%", [PriceChange(Select(category="Milk", store_brand=True), percent=-15)])
    Scenario("Cereal stock-out", [StockOut(Select(product_ids=(812,)))])
    Scenario("25% off Crunchies", [Promotion(Select(brand="Crunchies"), percent_off=25)])

`Scenario.apply(plan)` returns a new `PricePlan`: the final shelf price, promotion discount and availability of
every product in every week. That one set of arrays is what everything else uses (the twin here, and the true
store in the evaluation), so both always see exactly the same prices. Nothing is rounded.

Rules:
  PriceChange  the regular price changes by `percent`; a planned promotion keeps its discount on the new price
  Promotion    the product is on promotion at `percent_off` below its regular price (replacing any planned one)
  StockOut     the product can't be seen or bought
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class Select:
    """Which products a change applies to. Every filter that is given must match."""

    category: str | None = None
    department: str | None = None
    brand: str | None = None
    store_brand: bool | None = None
    product_ids: tuple[int, ...] | None = None

    def mask(self, products: pd.DataFrame) -> np.ndarray:
        """[products] True for the selected products (`products` sorted by product_id)."""
        keep = np.ones(len(products), dtype=bool)
        for column, value in (("category", self.category), ("department", self.department), ("brand", self.brand)):
            if value is not None:
                if value not in set(products[column]):
                    raise ValueError(f"no product has {column} '{value}'")
                keep &= (products[column] == value).to_numpy()
        if self.store_brand is not None:
            keep &= products.is_store_brand.to_numpy() == self.store_brand
        if self.product_ids is not None:
            ids = np.zeros(len(products), dtype=bool)
            ids[list(self.product_ids)] = True
            keep &= ids
        return keep

    def describe(self) -> str:
        parts = []
        if self.store_brand is not None:
            parts.append("store-brand" if self.store_brand else "branded")
        if self.brand:
            parts.append(self.brand)
        if self.category:
            parts.append(self.category)
        if self.department:
            parts.append(f"{self.department} department")
        if self.product_ids:
            ids = ", ".join(map(str, self.product_ids))
            parts.append(f"product {ids}" if len(self.product_ids) == 1 else f"products {ids}")
        return " ".join(parts) or "all products"


@dataclass(frozen=True)
class PriceChange:
    select: Select
    percent: float  # +20 = 20% dearer, -15 = 15% cheaper


@dataclass(frozen=True)
class Promotion:
    select: Select
    percent_off: float  # 25 = 25% off the regular price


@dataclass(frozen=True)
class StockOut:
    select: Select


@dataclass
class PricePlan:
    """The store's prices for every product and week."""

    price: np.ndarray  # [P, W] shelf price paid
    discount: np.ndarray  # [P, W] promotion discount as a fraction (0 = no promotion)
    regular: np.ndarray  # [P, W] regular price
    available: np.ndarray  # [P, W] bool, False = out of stock

    @property
    def promo(self) -> np.ndarray:
        """[P, W] 1.0 when on promotion."""
        return (self.discount > 0).astype(np.float32)

    @property
    def unavailable(self) -> np.ndarray:
        return ~self.available

    @property
    def n_weeks(self) -> int:
        return self.price.shape[1]

    @classmethod
    def from_world(cls, world) -> "PricePlan":
        """The store's own plan, from the public price history."""
        ph = world.price_history
        shape = (world.n_products, int(ph.week.max()) + 1)
        arrays = {}
        for name, column in (("price", "price"), ("discount", "promo_discount"), ("regular", "regular_price")):
            a = np.zeros(shape, dtype=np.float64)
            a[ph.product_id.to_numpy(), ph.week.to_numpy()] = ph[column].to_numpy()
            arrays[name] = a
        return cls(available=np.ones(shape, dtype=bool), **arrays)

    def copy(self) -> "PricePlan":
        return PricePlan(self.price.copy(), self.discount.copy(), self.regular.copy(), self.available.copy())


@dataclass
class Scenario:
    name: str
    changes: list = field(default_factory=list)
    weeks: tuple[int, int] | None = None  # first and last week it applies to; None = every week given to apply()

    def apply(self, plan: PricePlan, products: pd.DataFrame, weeks: tuple[int, int] | None = None) -> PricePlan:
        """A copy of `plan` with the changes applied to the scenario's weeks (or to `weeks` if it has none)."""
        first, last = self.weeks or weeks or (0, plan.n_weeks - 1)
        cols = np.arange(max(first, 0), min(last, plan.n_weeks - 1) + 1)
        out = plan.copy()
        products = products.sort_values("product_id")
        for change in self.changes:
            rows = np.flatnonzero(change.select.mask(products))
            if len(rows) == 0:
                raise ValueError(f"'{change.select.describe()}' selects no products")
            r, c = np.ix_(rows, cols)
            old_regular, old_discount = out.regular[r, c].copy(), out.discount[r, c].copy()
            if isinstance(change, PriceChange):
                out.regular[r, c] *= 1 + change.percent / 100
            elif isinstance(change, Promotion):
                if not 0 < change.percent_off < 100:
                    raise ValueError("percent_off must be between 0 and 100")
                out.discount[r, c] = change.percent_off / 100
            elif isinstance(change, StockOut):
                out.available[r, c] = False
            else:
                raise TypeError(f"unknown change {change!r}")
            # scale the planned shelf price (rather than recomputing it) so an unchanged product keeps its exact
            # planned price, cents rounding included
            out.price[r, c] *= (out.regular[r, c] / old_regular) * (1 - out.discount[r, c]) / (1 - old_discount)
        return out

    def targets(self, products: pd.DataFrame) -> np.ndarray:
        """[products] True for every product one of the changes applies to."""
        products = products.sort_values("product_id")
        keep = np.zeros(len(products), dtype=bool)
        for change in self.changes:
            keep |= change.select.mask(products)
        return keep

    def with_weeks(self, first: int, last: int) -> "Scenario":
        return replace(self, weeks=(first, last))

    def describe(self) -> str:
        parts = []
        for ch in self.changes:
            if isinstance(ch, PriceChange):
                parts.append(f"{ch.select.describe()} price {ch.percent:+g}%")
            elif isinstance(ch, Promotion):
                parts.append(f"{ch.select.describe()} on promotion at {ch.percent_off:g}% off")
            else:
                parts.append(f"{ch.select.describe()} out of stock")
        return "; ".join(parts)


def realised_price_change(base: PricePlan, new: PricePlan, products: np.ndarray, weeks: np.ndarray) -> float:
    """Average % change in the shelf price of `products` over `weeks` (what shoppers actually see)."""
    b = base.price[np.ix_(products, weeks)]
    n = new.price[np.ix_(products, weeks)]
    if b.size == 0:
        return 0.0
    return float(100 * (n / b - 1).mean())
