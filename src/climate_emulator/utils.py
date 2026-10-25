"""Serialization, hashing, logging and reproducibility utilities."""
from __future__ import annotations

import hashlib
import json
import logging
import os
import random
from pathlib import Path
from typing import Any

import numpy as np


def file_hash(path: str | Path, algorithm: str = "sha256") -> str:
    """Hash a file without loading it into memory."""
    digest = hashlib.new(algorithm)
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def clean_json(value: Any) -> Any:
    """Convert NumPy/path values, representing undefined values explicitly."""
    if isinstance(value, dict):
        return {str(k): clean_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [clean_json(v) for v in value]
    if isinstance(value, np.ndarray):
        return clean_json(value.tolist())
    if isinstance(value, np.generic):
        return clean_json(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return "infinity" if np.isposinf(value) else "-infinity" if np.isneginf(value) else None
    if isinstance(value, Path):
        return str(value)
    return value


def object_hash(value: Any) -> str:
    """Stable SHA-256 of JSON-compatible structured state."""
    encoded = json.dumps(clean_json(value), sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def write_json(path: str | Path, value: Any) -> None:
    """Atomically replace a JSON artifact with strict standard JSON."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(clean_json(value), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def read_json(path: str | Path) -> Any:
    """Read a JSON artifact."""
    return json.loads(Path(path).read_text(encoding="utf-8"))


def source_hash(root: Path) -> str:
    """Fingerprint implementation and tests, excluding generated artifacts."""
    files = sorted([*root.glob("src/**/*.py"), *root.glob("scripts/*.py"), *root.glob("tests/*.py"), root / "requirements.txt", root / "pyproject.toml"])
    return object_hash({str(p.relative_to(root)): file_hash(p) for p in files if p.is_file()})


def set_seed(seed: int, tensorflow: bool = False, deterministic: bool = True) -> None:
    """Seed numeric backends; TensorFlow is imported only when requested."""
    random.seed(seed)
    np.random.seed(seed)
    if tensorflow:
        import tensorflow as tf
        tf.keras.utils.set_random_seed(seed)
        if deterministic:
            tf.config.experimental.enable_op_determinism()


def configure_logging() -> None:
    """Enable timestamped application logging when invoked by a CLI."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
