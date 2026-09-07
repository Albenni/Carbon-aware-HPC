"""Download an Electricity Maps historical range into the local JSON cache."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys


from carbon_intensity import DEFAULT_ZONE, ElectricityMapsClient
from carbon_intensity.electricity_maps import read_api_key
from common import PROJECT_ROOT, parse_timestamp


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
