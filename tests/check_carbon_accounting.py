from __future__ import annotations

from collections.abc import Callable
from datetime import datetime, timezone
from math import isclose
from pathlib import Path
import sys
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
sys.path.insert(0, str(SRC_DIR))

from carbon_accounting import (
    JobPowerProfile,
    PowerModel,
    account_emissions,
    measured_average_power,
)

UTC = timezone.utc
PM100_DEBUG_PATH = PROJECT_ROOT / "data" / "processed" / "pm100_debug_5000.parquet"



class OutputChecks:
    def __init__(self) -> None:
        self.passed = 0
        self.failed = 0

    def close(
        self,
        label: str,
        actual: float,
        expected: float,
        unit: str,
        *,
        relative_tolerance: float = 1e-9,
        absolute_tolerance: float = 1e-12,
    ) -> None:
        passed = isclose(
            actual,
            expected,
            rel_tol=relative_tolerance,
            abs_tol=absolute_tolerance,
        )
        self._record(
            label,
            passed,
            f"expected {expected:.9f} {unit} | actual {actual:.9f} {unit}",
        )

    def true(self, label: str, condition: bool, details: str) -> None:
        self._record(label, condition, details)

    def raises(
        self,
        label: str,
        action: Callable[[], object],
        expected_error: type[Exception],
    ) -> None:
        try:
            action()
        except expected_error as error:
            self._record(label, True, f"rejected with: {error}")
        except Exception as error:  # noqa: BLE001 - the report must show unexpected failures
            self._record(
                label,
                False,
                f"expected {expected_error.__name__}, got {type(error).__name__}: {error}",
            )
        else:
            self._record(
                label,
                False,
                f"expected {expected_error.__name__}, but no error was raised",
            )

    def _record(self, label: str, passed: bool, details: str) -> None:
        if passed:
            self.passed += 1
            status = "PASS"
        else:
            self.failed += 1
            status = "FAIL"
        print(f"[{status}] {label}")
        print(f"       {details}")

    def exit_code(self) -> int:
        total = self.passed + self.failed
        print("\n" + "=" * 72)
        print(f"Summary: {self.passed}/{total} checks passed")
        if self.failed:
            print(f"Result: FAILED ({self.failed} inconsistent output(s))")
            return 1
        print("Result: all outputs are consistent")
        return 0


def print_section(title: str) -> None:
    print(f"\n{title}")
    print("-" * len(title))

def check_temporal_shift(checks: OutputChecks) -> None:
    print_section("1. Same job shifted between cleaner and dirtier hours")
    clean_start = datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    dirty_start = datetime(2026, 1, 1, 1, 0, tzinfo=UTC)
    job = JobPowerProfile(
        job_id="shift-example",
        duration_seconds=3_600,
        average_power_watts=1_000,
    )

    def hourly_signal(timestamp: datetime) -> float:
        return 100.0 if timestamp.hour == 0 else 400.0

    clean = account_emissions(job, clean_start, hourly_signal)
    dirty = account_emissions(job, dirty_start, hourly_signal)

    checks.close("energy is independent of start time", dirty.energy_kwh, clean.energy_kwh, "kWh")
    checks.close("clean-hour emissions", clean.emissions_gco2, 100.0, "gCO2")
    checks.close("dirty-hour emissions", dirty.emissions_gco2, 400.0, "gCO2")
    checks.true(
        "time shift changes emissions",
        dirty.emissions_gco2 > clean.emissions_gco2,
        f"clean {clean.emissions_gco2:.3f} gCO2 | dirty {dirty.emissions_gco2:.3f} gCO2",
    )

    try:
        rome_timezone = ZoneInfo("Europe/Rome")
    except ZoneInfoNotFoundError:
        print("[SKIP] daylight-saving check: IANA timezone data is unavailable")
    else:
        rome_start = datetime(2026, 3, 29, 1, 30, tzinfo=rome_timezone)
        dst_job = JobPowerProfile(
            job_id="dst-example",
            duration_seconds=7_200,
            average_power_watts=1_000,
        )
        dst_result = account_emissions(dst_job, rome_start, 100.0)
        elapsed_seconds = (
            dst_result.end_time.astimezone(UTC) - rome_start.astimezone(UTC)
        ).total_seconds()
        checks.close(
            "elapsed time remains correct across a daylight-saving change",
            elapsed_seconds,
            7_200.0,
            "s",
        )

    requested_timestamps: list[datetime] = []
    fractional_job = JobPowerProfile(
        job_id="fractional-segments",
        duration_seconds=1.0,
        average_power_watts=1.0,
        sample_interval_seconds=0.2,
    )
    account_emissions(
        fractional_job,
        clean_start,
        lambda timestamp: requested_timestamps.append(timestamp) or 100.0,
    )
    last_offset_seconds = (
        (requested_timestamps[-1] - clean_start).total_seconds()
        if requested_timestamps
        else float("nan")
    )
    checks.true(
        "segment arithmetic never queries intensity at the job end",
        len(requested_timestamps) == 5
        and isclose(last_offset_seconds, 0.8, abs_tol=1e-12),
        f"5 expected lookups | {len(requested_timestamps)} actual lookups | "
        f"last offset {last_offset_seconds:.1f} s",
    )


def check_power_model_comparison(checks: OutputChecks) -> None:
    print_section("2. Constant-average power versus measured power")
    start = datetime(2026, 1, 1, tzinfo=UTC)
    job = JobPowerProfile(
        job_id="shape-example",
        duration_seconds=40,
        average_power_watts=1_500,
        power_profile_watts=(1_000, 2_000),
        sample_interval_seconds=20,
    )

    def changing_signal(timestamp: datetime) -> float:
        elapsed_seconds = (timestamp - start).total_seconds()
        return 100.0 if elapsed_seconds < 20.0 else 500.0

    average = account_emissions(
        job,
        start,
        changing_signal,
        power_model=PowerModel.AVERAGE,
    )
    measured = account_emissions(
        job,
        start,
        changing_signal,
        power_model=PowerModel.MEASURED,
    )

    checks.close("average-model energy", average.energy_kwh, 1.0 / 60.0, "kWh")
    checks.close("measured-profile energy", measured.energy_kwh, 1.0 / 60.0, "kWh")
    checks.close("average-model emissions", average.emissions_gco2, 5.0, "gCO2")
    checks.close(
        "measured-profile emissions",
        measured.emissions_gco2,
        55.0 / 9.0,
        "gCO2",
    )
    checks.true(
        "power shape matters when grid intensity changes",
        measured.emissions_gco2 > average.emissions_gco2,
        f"average {average.emissions_gco2:.6f} gCO2 | measured {measured.emissions_gco2:.6f} gCO2",
    )

    average_constant = account_emissions(job, start, 300.0)
    measured_constant = account_emissions(
        job,
        start,
        300.0,
        power_model=PowerModel.MEASURED,
    )
    checks.close(
        "models agree under constant intensity and equal energy",
        measured_constant.emissions_gco2,
        average_constant.emissions_gco2,
        "gCO2",
    )


def check_profile_boundaries(checks: OutputChecks) -> None:
    print_section("3. PM100 profile-boundary policy")
    start = datetime(2026, 1, 1, tzinfo=UTC)

    partial = JobPowerProfile(
        job_id="partial",
        duration_seconds=30,
        average_power_watts=4_000 / 3,
        power_profile_watts=(1_000, 2_000),
    )
    partial_result = account_emissions(
        partial,
        start,
        0.0,
        power_model=PowerModel.MEASURED,
    )
    checks.close(
        "terminal excess is trimmed to runtime",
        partial_result.energy_kwh,
        40_000.0 / 3_600_000.0,
        "kWh",
    )
    checks.close(
        "runtime-weighted measured average",
        measured_average_power(partial),
        4_000.0 / 3.0,
        "W",
    )

    held = JobPowerProfile(
        job_id="held-tail",
        duration_seconds=30,
        average_power_watts=1_000,
        power_profile_watts=(1_000,),
    )
    held_result = account_emissions(
        held,
        start,
        0.0,
        power_model=PowerModel.MEASURED,
    )
    checks.close(
        "last value is held over a <=20 s missing tail",
        held_result.energy_kwh,
        30_000.0 / 3_600_000.0,
        "kWh",
    )

    invalid = JobPowerProfile(
        job_id="invalid-gap",
        duration_seconds=41,
        average_power_watts=1_000,
        power_profile_watts=(1_000,),
    )
    checks.raises(
        "a gap larger than one sample is rejected",
        lambda: account_emissions(
            invalid,
            start,
            0.0,
            power_model=PowerModel.MEASURED,
        ),
        ValueError,
    )
    checks.raises(
        "text is rejected as a measured power profile",
        lambda: JobPowerProfile(
            duration_seconds=60,
            average_power_watts=1,
            power_profile_watts="123",  # type: ignore[arg-type]
        ),
        TypeError,
    )


def show_pm100_example(checks: OutputChecks) -> None:
    print_section("4. Processed PM100 example")
    if not PM100_DEBUG_PATH.exists():
        print(f"[SKIP] {PM100_DEBUG_PATH.relative_to(PROJECT_ROOT)} is not present")
        return

    try:
        import pyarrow.parquet as parquet
    except ImportError:
        print("[SKIP] pyarrow is not installed; deterministic checks remain complete")
        return

    columns = [
        "job_id",
        "run_time",
        "start_time",
        "node_power_consumption",
        "node_power_mean_W",
        "num_nodes_alloc",
    ]
    parquet_file = parquet.ParquetFile(PM100_DEBUG_PATH)
    first_batch = next(parquet_file.iter_batches(batch_size=1, columns=columns))
    row = {name: first_batch[name][0].as_py() for name in columns}

    job = JobPowerProfile(
        job_id=row["job_id"],
        duration_seconds=row["run_time"],
        average_power_watts=row["node_power_mean_W"],
        power_profile_watts=tuple(row["node_power_consumption"]),
        sample_interval_seconds=20,
    )
    average = account_emissions(job, row["start_time"], 233.0)
    measured = account_emissions(
        job,
        row["start_time"],
        233.0,
        power_model=PowerModel.MEASURED,
    )

    expected_average_kwh = (
        row["node_power_mean_W"] * row["run_time"] / 3_600_000.0
    )
    remaining_seconds = row["run_time"]
    measured_watt_seconds = 0.0
    for power_watts in row["node_power_consumption"]:
        if remaining_seconds <= 0:
            break
        covered_seconds = min(20, remaining_seconds)
        measured_watt_seconds += power_watts * covered_seconds
        remaining_seconds -= covered_seconds
    if remaining_seconds > 0:
        measured_watt_seconds += (
            row["node_power_consumption"][-1] * remaining_seconds
        )
    expected_measured_kwh = measured_watt_seconds / 3_600_000.0

    checks.close(
        "real PM100 average-model energy",
        average.energy_kwh,
        expected_average_kwh,
        "kWh",
    )
    checks.close(
        "real PM100 measured-profile energy",
        measured.energy_kwh,
        expected_measured_kwh,
        "kWh",
    )
    print(
        f"       job {row['job_id']} | runtime {row['run_time']} s | "
        f"allocated nodes {row['num_nodes_alloc']}"
    )
    print(
        f"       whole-job trace used directly | {len(row['node_power_consumption'])} samples | "
        "no node-count multiplier"
    )
    print(
        f"       at 233 gCO2/kWh: average {average.emissions_gco2:.6f} gCO2 | "
        f"measured {measured.emissions_gco2:.6f} gCO2"
    )


def main() -> int:
    print("Carbon accounting output check")
    print("=" * 72)
    checks = OutputChecks()

    check_temporal_shift(checks)
    check_power_model_comparison(checks)
    check_profile_boundaries(checks)
    show_pm100_example(checks)
    return checks.exit_code()


if __name__ == "__main__":
    raise SystemExit(main())
