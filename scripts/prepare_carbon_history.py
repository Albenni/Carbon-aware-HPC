"""Prepare the historical carbon-intensity dataset and temporal protocol."""

from __future__ import annotations

import argparse
from datetime import timedelta
from hashlib import sha256
import json
from pathlib import Path
import sys


from carbon_intensity import ElectricityMapsClient, TemporalProtocol
from carbon_intensity.electricity_maps import read_api_key
from carbon_intensity.history import build_history
from common import PROJECT_ROOT


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--existing-2020", type=Path, default=(
        PROJECT_ROOT / "data/carbon_intensity/electricity_maps_it_no_04_to_11_2020.json"
    ))
    parser.add_argument("--output-dir", type=Path, default=(
        PROJECT_ROOT / "data/carbon_intensity/actual"
    ))
    parser.add_argument("--workload", type=Path, default=(
        PROJECT_ROOT / "data/processed/pm100_clean.parquet"
    ))
    parser.add_argument("--env-file", type=Path, default=PROJECT_ROOT / ".env")
    parser.add_argument("--api-key-variable", default="ELECTRICITY_MAPS_KEY")
    parser.add_argument("--observation-delay-minutes", type=int, default=0)
    parser.add_argument("--offline", action="store_true", help="rebuild using cached chunks only")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    try:
        protocol = TemporalProtocol.from_workload(
            args.workload, observation_delay=timedelta(minutes=args.observation_delay_minutes),
        )
        client = None if args.offline else ElectricityMapsClient(
            read_api_key(args.env_file, args.api_key_variable)
        )
        history = build_history(args.existing_2020, args.output_dir, client)
        splits = protocol.split(history)
        actual_path = args.output_dir / "actual.json"
        manifest = {
            **protocol.metadata(),
            "workload": str(args.workload),
            "actual_path": str(actual_path),
            "actual_sha256": sha256(actual_path.read_bytes()).hexdigest(),
            "split_buckets": {name: len(samples) for name, samples in splits.items()},
        }
        (args.output_dir / "protocol.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8",
        )
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Historical dataset failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved {len(history.samples)} actual buckets to {actual_path}")
    print(json.dumps(manifest["split_buckets"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
