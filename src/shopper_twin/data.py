"""Load a generated world for modelling.

Models only ever see the public data: this module never opens `answer_key/`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

EVENT_TYPES = ("view", "cart", "purchase")
PUBLIC_MANIFEST_KEYS = ("name", "start_date", "n_days", "n_products", "n_shoppers")


@dataclass
class PublicWorld:
    path: Path
    manifest: dict
    products: pd.DataFrame
    categories: pd.DataFrame
    shoppers: pd.DataFrame
    price_history: pd.DataFrame

    @property
    def n_products(self) -> int:
        return len(self.products)

    @property
    def n_shoppers(self) -> int:
        return len(self.shoppers)

    @property
    def n_days(self) -> int:
        return int(self.manifest["n_days"])

    @property
    def start(self) -> pd.Timestamp:
        return pd.Timestamp(self.manifest["start_date"])

    def events(self, types: tuple[str, ...] = EVENT_TYPES, extra_columns: tuple[str, ...] = ()) -> pd.DataFrame:
        """Events of the given types: shopper_id, product_id, event_type (categorical), day (0 = first day).

        Read in batches so memory stays close to the size of the result (~11 bytes per event).
        `extra_columns` adds any of session_id, quantity, price, on_promo, timestamp.
        """
        unknown = set(types) - set(EVENT_TYPES)
        if unknown:
            raise ValueError(f"unknown event types {unknown}")
        start = np.datetime64(self.start.to_datetime64(), "s")
        columns = ["shopper_id", "product_id", "event_type", "timestamp", *extra_columns]
        parts: dict[str, list] = {c: [] for c in ["shopper_id", "product_id", "event_code", "day", *extra_columns]}
        f = pq.ParquetFile(self.path / "public" / "events.parquet")
        for batch in f.iter_batches(columns=list(dict.fromkeys(columns)), batch_size=1_000_000):
            code = _event_codes(batch.column("event_type"))
            keep = np.isin(code, [EVENT_TYPES.index(t) for t in types])
            ts = batch.column("timestamp").to_numpy().astype("datetime64[s]")
            parts["day"].append(((ts[keep] - start) // np.timedelta64(1, "D")).astype(np.int16))
            parts["event_code"].append(code[keep])
            for c in ("shopper_id", "product_id", *extra_columns):
                parts[c].append(batch.column(c).to_numpy(zero_copy_only=False)[keep])
        data = {c: np.concatenate(v) if v else np.array([]) for c, v in parts.items()}
        codes = data.pop("event_code")
        ev = pd.DataFrame({
            "shopper_id": data.pop("shopper_id").astype(np.int32),
            "product_id": data.pop("product_id").astype(np.int32),
            "event_type": pd.Categorical.from_codes(codes, categories=list(EVENT_TYPES)),
            "day": data.pop("day"),
            **data,
        })
        return ev


def _event_codes(column: pa.Array) -> np.ndarray:
    """Event type as 0/1/2 (view/cart/purchase), using the parquet dictionary when there is one."""
    if pa.types.is_dictionary(column.type):
        lookup = np.array([EVENT_TYPES.index(v) for v in column.dictionary.to_pylist()], dtype=np.int8)
        return lookup[column.indices.to_numpy(zero_copy_only=False)]
    names = column.to_numpy(zero_copy_only=False)
    code = np.full(len(names), -1, dtype=np.int8)
    for i, name in enumerate(EVENT_TYPES):
        code[names == name] = i
    return code


def load_public(world_dir: str | Path) -> PublicWorld:
    path = Path(world_dir)
    pub = path / "public"
    manifest = json.loads((path / "manifest.json").read_text())
    if "config" in manifest:
        raise ValueError(f"{path} was generated before milestone 2 and its manifest.json reveals hidden "
                         "settings. Regenerate it with scripts/generate_world.py.")
    return PublicWorld(
        path=path,
        manifest={k: manifest[k] for k in PUBLIC_MANIFEST_KEYS},
        products=pd.read_parquet(pub / "products.parquet"),
        categories=pd.read_parquet(pub / "categories.parquet"),
        shoppers=pd.read_parquet(pub / "shoppers.parquet"),
        price_history=pd.read_parquet(pub / "price_history.parquet"),
    )
