"""Small offline checks for the delay-aware experiment: python tests/check_experiments_improved_scheduler.py."""

import contextlib
from datetime import datetime, timedelta, timezone
import io
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import pandas as pd

from carbon_accounting import JobPowerProfile
from carbon_intensity.series import CarbonIntensitySample as Sample
from carbon_intensity.series import FIFTEEN_MINUTES as STEP, TimeSeriesCarbonIntensityProvider as Series
import experiments.improved_scheduler as experiment
from experiments.milp import replay, solve
from hpc_sim import CarbonSignal, Job, account_schedule


BASE = datetime(2020, 5, 6, tzinfo=timezone.utc)
HOUR = timedelta(hours=1)


def at(hours: float) -> datetime:
    return BASE + hours * HOUR


def job(job_id: str, watts: float, *, profile=None, release: float = 0.0, nodes: int = 1) -> Job:
    sampled = {"power_profile_watts": profile, "sample_interval_seconds": 20.0} if profile else {}
    return Job(
        job_id=job_id, submit_time=at(release), release_time=at(release), nodes_required=nodes,
        actual_duration_seconds=3_600.0,
        power=JobPowerProfile(job_id=job_id, duration_seconds=3_600.0, average_power_watts=watts, **sampled),
    )


def main() -> None:
    # Voluntary delay: a target pulled earlier, one pushed later, and a job
    # offered to EASY and then held again. Contention after the target is not
    # the policy's doing and must not count.
    decisions = pd.DataFrame(
        [("a", at(0), at(2)), ("a", at(1), at(1.5)),
         ("b", at(0), at(1)), ("b", at(0.5), at(3)),
         ("c", at(0), at(0)), ("c", at(1), at(2))],
        columns=["job_id", "decision_time", "target"],
    )
    schedule = pd.DataFrame({"job_id": ["a", "b", "c", "d"],
                             "start_time": [at(1 + 40 / 60), at(3), at(2.5), at(0.2)]})
    held = experiment.voluntary_delay(decisions, schedule) / 3600
    assert held.round(9).to_dict() == {"a": 1.5, "b": 3.0, "c": 1.0, "d": 0.0}, held

    value, reason = experiment.ratio(5.0, 0.0)
    assert math.isnan(value) and reason
    assert math.isnan(experiment.ratio(5.0, -2.0)[0]) and experiment.ratio(6.0, 2.0) == (3.0, "")

    # Non-dominance uses the numerical tolerances only.
    assert not experiment.dominates((1, 2, 3, 4, 5), (1, 2, 3, 4, 5 + 1e-12))
    assert experiment.dominates((1, 2, 3, 4, 5), (1, 2, 3, 4, 6))
    points = pd.DataFrame(
        [(name, *vector, saved) for name, vector, saved in (
            ("low-carbon", (10, 9, 9, 9, 9), 5.0),
            ("low-wait", (12, 1, 9, 9, 3), 3.0),
            ("dominated", (13, 9, 9, 9, 9), 1.0),
            ("no-saving", (20, 0, 0, 0, 0), -1.0),
        )],
        columns=["policy_id", *experiment.FRONTIER, "saved_vs_easy_gco2e"],
    )
    # The point saving nothing is non-dominated but not eligible while savers exist.
    assert experiment.representatives(points) == ["low-carbon", "low-wait"]

    # An incomplete join against EASY is refused, not silently averaged.
    with tempfile.TemporaryDirectory() as directory:
        original = experiment.OUTPUT
        experiment.OUTPUT = Path(directory)
        try:
            for key, ids in (("run", ["a", "b"]), ("easy", ["a"])):
                (Path(directory) / "runs" / key).mkdir(parents=True)
                frame = pd.DataFrame({"job_id": ids, "start_time": [at(0)] * len(ids)})
                for column in ("emissions_gco2e", "waiting_s", "bounded_slowdown", "actual_duration_s"):
                    frame[column] = 1.0
                frame.to_parquet(Path(directory) / "runs" / key / "jobs.parquet")
                pd.DataFrame(columns=["job_id", "decision_time", "target"]).to_parquet(
                    Path(directory) / "runs" / key / "decisions.parquet")
            try:
                experiment.paired({"run_key": "run", "stage": "test"}, {"run_key": "easy"})
            except ValueError:
                pass
            else:
                raise AssertionError("an incomplete join was accepted")
        finally:
            experiment.OUTPUT = original

    # MILP: dirty for two hours, clean from 2 h. The job saves 300 g by waiting
    # two hours, 150 g per hour: a 200 g/h price keeps it at release.
    provider = Series(tuple(Sample(BASE + i * STEP, 100.0 if i >= 8 else 400.0) for i in range(96)))
    signal = CarbonSignal(provider)
    single = (job("j", 1_000.0),)
    options = dict(nodes=1, max_delay=3 * HOUR, granularity=HOUR, time_limit=60.0)
    # At zero price every clean start is optimal; nothing breaks the tie.
    assert solve(single, signal, delay_weight=0.0, **options)[0]["j"] in (at(2), at(3))
    assert solve(single, signal, delay_weight=200.0, **options)[0]["j"] == at(0)

    # A non-constant profile straddling the transition (a steep price keeps it
    # at release): the plan minimises the average-power model, and the measured
    # figure is reported beside it, not in its place.
    spiky = (job("s", 1_000.0, profile=(1_800.0,) * 90 + (200.0,) * 90, release=1.5),)
    starts, info = solve(spiky, signal, delay_weight=1e4, **options)
    plan = account_schedule(replay(spiky, starts, 1), spiky, provider)
    record = plan.records[0]
    assert info["solution_status"] == "optimal"
    assert abs(info["objective_gco2e"] - record.emissions_gco2e_average_model) < 1e-6
    assert record.emissions_gco2e != record.emissions_gco2e_average_model

    # A solve stopped by the clock is never labelled optimal.
    crowd = tuple(job(f"c{i}", 1_000.0 + 97 * i, release=i / 8, nodes=1 + i % 3) for i in range(40))
    _, info = solve(crowd, signal, nodes=10, max_delay=6 * HOUR, granularity=STEP,
                    delay_weight=0.0, time_limit=1e-3)
    assert info["solution_status"] != "optimal" and not info["proven_optimal"], info

    # The contention sweep reports and writes after its simulations; it used to
    # die there on a name out of scope. Skipped without the raw trace.
    root = Path(__file__).resolve().parents[1]
    raw = root / "data/job_table.parquet"
    if raw.exists():
        with tempfile.TemporaryDirectory() as directory:
            written = Path(directory) / "sweep.csv"
            subprocess.run(
                [sys.executable, str(root / "scripts/carbon_tradeoff.py"),
                 "--workload", str(root / "data/processed/pm100_debug_5000.parquet"),
                 "--contention-workload", str(raw), "--limit", "40", "--max-delay-hours", "0", "1",
                 "--carbon-cache", str(root / "data/carbon_intensity/electricity_maps_it_no_04_to_11_2020.json"),
                 "--output", str(written)],
                check=True, capture_output=True,
            )
            sweep = pd.read_csv(written)
            assert (sweep["scheduled_jobs"] == 40 + sweep["contention_only_jobs"]).all()

    print("Delay-aware experiment checks passed.")


class ImprovedSchedulerChecks(unittest.TestCase):
    """Make ``unittest discover`` run the script above, which prints its summary."""

    def test_checks_pass(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            main()


if __name__ == "__main__":
    main()
