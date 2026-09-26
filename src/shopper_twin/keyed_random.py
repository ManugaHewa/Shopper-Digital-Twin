"""Random numbers that depend only on *what* they are for, not on how many were drawn before.

A normal random generator hands out numbers in sequence, so if a price change makes one shopper
buy something else, every later draw shifts and the whole rest of the simulation changes. For
what-if comparisons we want the opposite: the same shopper, on the same day, looking at the same
product, always gets the same random number. Here each number is a hash of its keys
(seed, day, shopper, product, purpose), so two runs differ only where the scenario really
changed something ("common random numbers").
"""

from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import poisson

_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)

# purposes, so draws for different things never share a key
VISIT, DUE, BROWSE_COUNT, BROWSE, VIEW, CHOICE, OUTSIDE, ABANDON, QUANTITY, LOST, HOUR, TIMESTAMP, DRIFT, TRIPS = \
    range(1, 15)


def _mix(x: np.ndarray) -> np.ndarray:
    """splitmix64 finaliser: turns any 64-bit integer into a well-scrambled one."""
    x = (x ^ (x >> np.uint64(30))) * _M1
    x = (x ^ (x >> np.uint64(27))) * _M2
    return x ^ (x >> np.uint64(31))


def keyed_uniform(*keys) -> np.ndarray:
    """Uniform numbers in (0, 1), one per broadcast combination of the integer keys."""
    arrays = np.broadcast_arrays(*[np.asarray(k).astype(np.int64).astype(np.uint64) for k in keys])
    h = np.zeros(arrays[0].shape, dtype=np.uint64)
    with np.errstate(over="ignore"):
        for a in arrays:
            h = _mix(h + _GOLDEN + a)
    return ((h >> np.uint64(11)).astype(np.float64) + 0.5) / float(2 ** 53)


def keyed_gumbel(*keys) -> np.ndarray:
    return -np.log(-np.log(keyed_uniform(*keys)))


def keyed_normal(*keys) -> np.ndarray:
    return ndtri(keyed_uniform(*keys))


def keyed_poisson(lam, *keys) -> np.ndarray:
    return poisson.ppf(keyed_uniform(*keys), lam).astype(np.int64)
