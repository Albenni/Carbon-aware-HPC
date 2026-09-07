"""Paths and argument types every script shares.

These lived in ``run_simulation`` and were imported from there, which made a
download script pull in the whole simulator for two constants. They belong to
no single entry point, so they live here.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data"
DEFAULT_WORKLOAD = DATA_ROOT / "processed" / "pm100_debug_5000.parquet"
DEFAULT_CARBON_CACHE = (
    DATA_ROOT / "carbon_intensity" / "electricity_maps_it_no_04_to_11_2020.json"
)
DEFAULT_OUTPUT_DIR = DATA_ROOT / "simulations"


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp, assuming UTC when the offset is omitted.

    ``fromisoformat`` accepts a trailing ``Z`` directly on Python 3.11+.
    """

    try:
        timestamp = datetime.fromisoformat(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError(f"invalid ISO timestamp: {value}") from error
    if timestamp.utcoffset() is None:
        return timestamp.replace(tzinfo=timezone.utc)
    return timestamp


def display_path(path: Path) -> str:
    """Shorten a path against the project root when it lies inside it."""

    resolved = path.resolve()
    try:
        return str(resolved.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(resolved)
