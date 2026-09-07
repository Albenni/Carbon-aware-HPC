"""Refit the selected forecaster during the replay, on data published by then.

A frozen model answers every day of the PM100 period with the weights it had
before the period started. A walk-forward model is allowed to keep learning: at
a fixed period it is refitted on the observations that, at that instant, would
already have been published. The two differ only in which rows are visible when,
so the question this module answers is whether the extra bookkeeping buys any
accuracy at all.

``period=None`` is the frozen configuration itself, refitted once at the start of
the replay. That is not the same model as the one scored on validation: the model
selected there stops learning at the end of 2019, while a replay starting in May
2020 could legitimately have used the whole validation period as training data.
Both are reported, so the cost of leaving those four months unused is visible.

Whether to update at all is decided on validation, never on the PM100 period:
``python -m carbon_intensity.walkforward`` scores the frozen model against
several retraining frequencies there and records the outcome.
"""

from dataclasses import dataclass, field
from datetime import datetime, timedelta
import json

import numpy as np

from .evaluate import (
    base_parser,
    load_inputs,
    provenance,
    report,
    run_entry_point,
    selected,
    write_evaluation,
)
from .features import LOOKBACK
from .forecasting import (
    HORIZON,
    MODEL_VERSION,
    design,
    fit_direct,
    observable_rows,
    predict,
    trajectory,
)
from .series import CarbonIntensityForecast
from .protocol import TemporalProtocol
from .series import FIFTEEN_MINUTES as STEP, TimeSeriesCarbonIntensityProvider, aware_utc
from .selection import Candidate, predict_validation, score, select


# Retraining frequencies offered to the selection rule, from weekly to quarterly.
UPDATE_CANDIDATES = (
    Candidate("frozen"),
    Candidate("refit_7d", refit=timedelta(days=7)),
    Candidate("refit_14d", refit=timedelta(days=14)),
    Candidate("refit_30d", refit=timedelta(days=30)),
    Candidate("refit_90d", refit=timedelta(days=90)),
)


def refit_cutoffs(start: datetime, end: datetime, period: timedelta | None) -> tuple[datetime, ...]:
    """Instants at which the model is refitted, the first one opening the replay."""
    if start >= end:
        raise ValueError("the replay window must be non-empty")
    if period is None:
        return (start,)
    if not isinstance(period, timedelta) or period <= timedelta(0):
        raise ValueError("the retraining period must be a positive timedelta")
    cutoffs = [start]
    while cutoffs[-1] + period < end:
        cutoffs.append(cutoffs[-1] + period)
    return tuple(cutoffs)


def pooled_design(
    actual: TimeSeriesCarbonIntensityProvider, protocol: TemporalProtocol,
    partitions: tuple[str, ...] = ("train", "validation", "test"),
) -> tuple[np.ndarray, ...]:
    """Stack the partitions a refit schedule may draw rows from, built once.

    A frozen replay never reaches the test rows, but a refitting one does as the
    period advances, so every schedule reads the same matrices and differs only
    in which rows are visible when.
    """
    return tuple(
        np.concatenate(matrices)
        for matrices in zip(*(design(actual, protocol, name) for name in partitions), strict=True)
    )


@dataclass
class WalkForwardForecaster:
    """The selected equations, refitted on a schedule fixed before the replay.

    Every fit is computed up front, so a trajectory is never delayed by training
    and the whole schedule is known before the first forecast is issued.
    """

    actual: TimeSeriesCarbonIntensityProvider
    protocol: TemporalProtocol
    period: timedelta | None
    fits: dict = field(repr=False)
    metadata: dict = field(repr=False)

    @classmethod
    def build(
        cls, actual: TimeSeriesCarbonIntensityProvider, protocol: TemporalProtocol,
        rows_source: tuple[np.ndarray, ...], *, replay: str = "test",
        period: timedelta | None = None, alpha: float, history_span: timedelta | None = None,
    ) -> "WalkForwardForecaster":
        """Fit one model per scheduled cutoff, each on rows published by then."""
        start, end = protocol.intervals[replay]
        shared, seasonal, labels, issued = rows_source
        fits, schedule = {}, []
        for cutoff in refit_cutoffs(start, end, period):
            rows = observable_rows(issued, cutoff, protocol, history_span)
            fits[cutoff] = fit_direct(shared[rows], seasonal[rows], labels[rows], alpha)
            # The latest label any of those rows carries; features are older still.
            training_cutoff = max(issued[rows]) + HORIZON - STEP
            schedule.append({
                "refit_at": cutoff.isoformat(),
                "training_cutoff": training_cutoff.isoformat(),
                "training_available_at": (
                    training_cutoff + STEP + protocol.observation_delay
                ).isoformat(),
                "training_examples": int(rows.sum()),
            })
        return cls(actual, protocol, period, fits, {
            **protocol.metadata(),
            "model_name": "ridge_refit_once" if period is None else f"ridge_refit_{period.days}d",
            "model_version": MODEL_VERSION, "alpha": alpha,
            "history_span_days": None if history_span is None else history_span.days,
            "refit_days": None if period is None else period.days,
            "replay_partition": replay, "replay_start": start.isoformat(), "replay_end": end.isoformat(),
            "refits": schedule,
            "training_cutoff": schedule[0]["training_cutoff"],
            "training_available_at": schedule[0]["training_available_at"],
            "training_examples": schedule[0]["training_examples"],
            "update_rule": "a training row is used only once its whole label window is observable",
            "strategy": "direct: one linear equation per 15-minute target bucket",
        })

    def refit_at(self, issue_time: datetime) -> datetime:
        """The most recent scheduled cutoff at or before ``issue_time``."""
        issue_time = aware_utc(issue_time, "issue_time")
        start = min(self.fits)
        if issue_time < start:
            raise ValueError("the replay had not started at issue_time")
        if self.period is None:
            return start
        return min(start + (issue_time - start) // self.period * self.period, max(self.fits))

    def provenance(self, issue_time: datetime) -> dict:
        """Which fit serves this issue time, so a snapshot can record its own cutoff."""
        refit = self.refit_at(issue_time).isoformat()
        return next(entry for entry in self.metadata["refits"] if entry["refit_at"] == refit)

    def get_forecast(self, issue_time: datetime, horizon: timedelta) -> CarbonIntensityForecast:
        issue_time = aware_utc(issue_time, "issue_time")
        if not isinstance(horizon, timedelta) or not timedelta(0) < horizon <= HORIZON:
            raise ValueError("horizon must be positive and at most 24 hours")
        coefficients, intercept = self.fits[self.refit_at(issue_time)]
        history = self.protocol.history(self.actual, issue_time, LOOKBACK)
        count = (horizon + STEP - timedelta(microseconds=1)) // STEP
        return trajectory(issue_time, predict(
            coefficients, intercept, history, issue_time, self.protocol, count,
        ))


def main() -> int:
    args = base_parser(__doc__).parse_args()

    def work() -> None:
        actual, protocol, actual_hash = load_inputs(args.actual, args.protocol)
        configuration = selected(args.output_dir / "ridge_direct.json")
        candidates = tuple(
            Candidate(item.name, configuration.get("history_span"), item.refit,
                      configuration.get("alpha", item.alpha))
            for item in UPDATE_CANDIDATES
        )
        validation = design(actual, protocol, "validation")
        pooled = pooled_design(actual, protocol, ("train", "validation"))
        rows = []
        for candidate in candidates:
            predicted, used = predict_validation(candidate, protocol, pooled, validation)
            row = score(candidate, predicted, validation[2], used)
            rows.append(row)
            report(candidate.name, row,
                   f", rows={used[0]}..{used[-1]} over {len(used)} fits")
        chosen = select(rows, candidates)
        # Even when frozen wins, the strongest schedule stays on record: it is the
        # arm the replay compares against, so the comparison is not a straw man.
        strongest = min((row for row in rows if row["refit_days"]),
                        key=lambda row: row["mae_gco2e_per_kwh"])
        decision = {
            **chosen,
            "comparison_refit_days": strongest["refit_days"],
            "comparison_candidate": strongest["model"],
            "candidates": [candidate.describe() for candidate in candidates],
            "update_rule": "a training row is used only once its whole label window is observable",
            "replay_note": "the chosen schedule is applied from the start of the PM100 period",
        }
        write_evaluation(args.output_dir, "walkforward", rows, {
            **provenance(args, protocol, actual_hash),
            "selected": decision, "evaluation": rows,
        })
        (args.output_dir / "selected_update.json").write_text(
            json.dumps(decision, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
        )
        print(f"Selected {chosen['model']} "
              f"({'frozen' if decision['frozen'] else 'walk-forward'}); results in {args.output_dir}")

    return run_entry_point("Update-strategy comparison", work)


if __name__ == "__main__":
    raise SystemExit(main())
