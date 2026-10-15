"""Shared per-channel normalization statistics for archived grids."""

from functools import lru_cache
import json
from pathlib import Path

import numpy as np

CHANNEL_ORDER = (
    "radar_dbz",
    "ir_temp",
    "lightning_density",
    "cape",
    "pwat",
)


@lru_cache(maxsize=1)
def load_normalization_stats() -> tuple[np.ndarray, np.ndarray]:
    """Load and validate the archive-derived channel means and standard deviations."""
    stats_path = Path(__file__).with_name("norm_stats.json")
    with stats_path.open(encoding="utf-8") as stats_file:
        stats = json.load(stats_file)

    if tuple(stats.get("channel_order", ())) != CHANNEL_ORDER:
        raise ValueError(f"Unexpected channel order in {stats_path}")

    means = np.asarray(stats["mean"], dtype=np.float32)
    stds = np.asarray(stats["std"], dtype=np.float32)
    if means.shape != (len(CHANNEL_ORDER),) or stds.shape != means.shape:
        raise ValueError(f"Expected {len(CHANNEL_ORDER)} normalization values in {stats_path}")
    if not np.isfinite(means).all() or not np.isfinite(stds).all() or np.any(stds <= 0):
        raise ValueError(f"Invalid normalization statistics in {stats_path}")

    means.setflags(write=False)
    stds.setflags(write=False)
    return means, stds


def normalize_channels(values: np.ndarray) -> np.ndarray:
    """Normalize channel-last arrays using the shared archive statistics."""
    means, stds = load_normalization_stats()
    return (values - means) / stds