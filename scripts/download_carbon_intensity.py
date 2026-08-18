"""Download an Electricity Maps historical range into the local JSON cache."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from carbon_intensity import DEFAULT_ZONE, ElectricityMapsClient


def parse_timestamp(value: str) -> datetime:
    """Parse an ISO timestamp, assuming UTC when timezone is omitted."""

    try:
        timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            f"invalid ISO timestamp: {value}"
        ) from error

    if timestamp.tzinfo is None or timestamp.utcoffset() is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)

    return timestamp

def read_api_key(env_file: Path, variable_name: str) -> str:
    """Read the token from env file."""

    environment_value = os.environ.get(variable_name)
    if environment_value:
        return environment_value.strip()

    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("export "):
                line = line.removeprefix("export ").lstrip()
            key, separator, value = line.partition("=")
            if separator and key.strip() == variable_name:
                return value.strip().strip("'\"")

    raise RuntimeError(
        f"set {variable_name} or add it to {env_file}; the token is never cached"
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Cache 15-minute actual carbon intensity from Electricity Maps.",
    )
    parser.add_argument(
        "--start",
        required=True,
        type=parse_timestamp,
        help="inclusive ISO timestamp (UTC when no offset is supplied)",
    )
    parser.add_argument(
        "--end",
        required=True,
        type=parse_timestamp,
        help="exclusive ISO timestamp (UTC when no offset is supplied)",
    )
    parser.add_argument("--zone", default=DEFAULT_ZONE)
    parser.add_argument(
        "--output",
        type=Path,
        help="cache path; defaults to a filename derived from --zone",
    )
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--api-key-variable", default="ELECTRICITY_MAPS_KEY")
    parser.add_argument(
        "--direct-emissions",
        action="store_true",
        help="request direct rather than lifecycle electricity emission factors",
    )
    parser.add_argument(
        "--exclude-estimated",
        action="store_true",
        help="fail if Electricity Maps would need estimated historical values",
    )
    return parser


def default_output_path(zone: str) -> Path:
    """Build a safe, zone-specific cache filename."""

    zone_slug = "".join(
        character.lower() if character.isalnum() else "_" for character in zone
    ).strip("_")
    if not zone_slug:
        raise ValueError("zone must contain at least one letter or number")
    return (
        PROJECT_ROOT
        / "data"
        / "carbon_intensity"
        / f"electricity_maps_{zone_slug}_15min.json"
    )


def main() -> int:
    arguments = build_parser().parse_args()
    try:
        destination = arguments.output or default_output_path(arguments.zone)
        api_key = read_api_key(arguments.env_file, arguments.api_key_variable)
        provider = ElectricityMapsClient(api_key).fetch_actual_range(
            arguments.start,
            arguments.end,
            zone=arguments.zone,
            emission_factor_type=(
                "direct" if arguments.direct_emissions else "lifecycle"
            ),
            include_estimated=not arguments.exclude_estimated,
        )
    except (RuntimeError, ValueError) as error:
        print(f"Download failed: {error}", file=sys.stderr)
        return 1
    output_path = provider.save(destination)
    print(
        f"Saved {len(provider.samples)} buckets for {arguments.zone} to "
        f"{output_path}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
