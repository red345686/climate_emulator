"""Lazy index-based temporal windows with strict consecutive-year validation."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def window_indices(years: np.ndarray, endpoints: np.ndarray, history: int = 10) -> np.ndarray:
    """Return positions only when every required preceding year exists."""
    if len(np.unique(years)) != len(years) or np.any(np.diff(years) != 1):
        raise ValueError("Histories require unique consecutive years")
    lookup = {int(year): i for i, year in enumerate(years)}
    result = []
    for year in endpoints:
        needed = range(int(year) - history + 1, int(year) + 1)
        if any(value not in lookup for value in needed):
            raise ValueError(f"Incomplete input history ending in {year}")
        result.append([lookup[value] for value in needed])
    if not result:
        raise ValueError("No eligible target endpoints")
    return np.asarray(result, dtype=np.int64)


@dataclass
class HistoryStore:
    """Store each normalized map once; materialize only a requested batch."""
    maps: np.ndarray
    indices: np.ndarray

    def batch(self, positions: np.ndarray) -> np.ndarray:
        """Return (batch,history,lat,lon,channel) tensor."""
        return self.maps[self.indices[positions]]

    def __len__(self) -> int:
        return len(self.indices)
