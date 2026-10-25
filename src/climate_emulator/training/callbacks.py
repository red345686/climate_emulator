"""Fit deadlines and structured append-only training logs."""
from __future__ import annotations

import logging
import time
from pathlib import Path

from climate_emulator.utils import clean_json
import json


class FitBudgetExceeded(RuntimeError):
    """A configured fit was interrupted and must not be reported as completed."""


class Deadline:
    """Wall-clock deadline checked between compute steps."""

    def __init__(self, seconds: float):
        self.started, self.seconds = time.monotonic(), seconds

    def check(self) -> None:
        """Fail rather than silently accepting a truncated fit."""
        if time.monotonic() - self.started > self.seconds:
            raise FitBudgetExceeded(f"Fit exceeded {self.seconds} seconds")

    @property
    def elapsed(self) -> float:
        """Measured seconds since the fit began."""
        return time.monotonic() - self.started


def log_epoch(path: Path | None, record: dict) -> None:
    """Write one structured record and emit human-readable progress."""
    logging.getLogger(__name__).info("epoch=%s loss=%s val_loss=%s", record["epoch"], record["loss"], record.get("val_loss"))
    if path is not None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(clean_json(record), allow_nan=False) + "\n")
