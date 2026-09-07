"""Choose the training history and model configuration on validation only.

Every candidate is fitted and scored under the frozen temporal protocol, so the
comparison never reads the PM100 test period. Candidates differ in how much
history they train on, whether the model is refitted during validation, and in
the ridge penalty; the feature pipeline and the direct multi-horizon strategy
are shared, which is what makes the numbers comparable.

Two candidate families answer the same question from opposite ends. A *frozen*
candidate trains once on data ending at the start of validation, the setup the
snapshot archive uses. A *refitting* candidate is retrained at a fixed period
during validation, and may then use the 2020 observations that would already
have been published at that moment: a training row enters only once its whole
24-hour label window has become observable, and it serves exclusively origins
at or after that refit. Both read rows from the same matrices, so the only
difference between them is which rows are visible when.
"""

from dataclasses import dataclass
from datetime import timedelta
import json

import numpy as np

from .evaluate import (
    base_parser,
    error_metrics,
    load_inputs,
    provenance,
    report,
    run_entry_point,
    write_evaluation,
)
from .features import FEATURE_NAMES
from .forecasting import (
    BUCKETS,
    DEFAULT_ALPHA,
    FEATURE_LAYOUT,
    design,
    fit_direct,
    observable_rows,
)
from .protocol import TemporalProtocol
from .series import FIFTEEN_MINUTES as STEP


# A more complex candidate has to beat the simplest one by this relative MAE
# margin. It is declared before the results, so a fractional win never buys
# history bookkeeping or periodic retraining the thesis would have to defend.
MIN_IMPROVEMENT = 0.01
YEAR = timedelta(days=365)


@dataclass(frozen=True, slots=True)
class Candidate:
    """One training-history and hyperparameter choice, scored on validation."""

    name: str
    history_span: timedelta | None = None
    refit: timedelta | None = None
    alpha: float = DEFAULT_ALPHA

    @property
    def complexity(self) -> tuple[bool, bool]:
        """Rank frozen, full-history candidates as the simplest option."""
        return (self.refit is not None, self.history_span is not None)

    def describe(self) -> dict:
        return {
            "model": self.name, "family": "ridge_direct", "alpha": self.alpha,
            "history_span_days": None if self.history_span is None else self.history_span.days,
            "refit_days": None if self.refit is None else self.refit.days,
        }


CANDIDATES = (
    Candidate("full_history_alpha1", alpha=1.0),
    Candidate("full_history_alpha0.1", alpha=0.1),
    Candidate("full_history_alpha0.01", alpha=0.01),
    Candidate("recent_36m", history_span=3 * YEAR),
    Candidate("recent_24m", history_span=2 * YEAR),
    Candidate("recent_12m", history_span=YEAR),
    Candidate("recent_6m", history_span=timedelta(days=182)),
    Candidate("full_history_refit_30d", refit=timedelta(days=30)),
    Candidate("recent_24m_refit_30d", history_span=2 * YEAR, refit=timedelta(days=30)),
    Candidate("recent_12m_refit_30d", history_span=YEAR, refit=timedelta(days=30)),
    Candidate("recent_6m_refit_30d", history_span=timedelta(days=182), refit=timedelta(days=30)),
)


def predict_validation(
    candidate: Candidate, protocol: TemporalProtocol, pooled: tuple, validation: tuple,
) -> tuple[np.ndarray, list[int]]:
    """Predict every validation origin using only rows observable beforehand."""
    shared, seasonal, labels, issued = pooled
    val_shared, val_seasonal, _, val_issued = validation
    edges = [val_issued[0]]
    if candidate.refit is not None:
        while edges[-1] + candidate.refit <= val_issued[-1]:
            edges.append(edges[-1] + candidate.refit)
    predicted = np.empty((len(val_issued), BUCKETS))
    used = []
    for index, cutoff in enumerate(edges):
        rows = observable_rows(issued, cutoff, protocol, candidate.history_span)
        coefficients, intercept = fit_direct(
            shared[rows], seasonal[rows], labels[rows], candidate.alpha,
        )
        served = val_issued >= cutoff
        if index + 1 < len(edges):
            served &= val_issued < edges[index + 1]
        split = len(FEATURE_NAMES)
        predicted[served] = np.maximum(
            intercept
            + val_shared[served] @ coefficients[:, :split].T
            + (val_seasonal[served] * coefficients[:, split:]).sum(axis=2),
            0,
        )
        used.append(int(rows.sum()))
    return predicted, used


def score(candidate: Candidate, predicted: np.ndarray, labels: np.ndarray, used: list[int]) -> dict:
    """Reuse the evaluator's metric definitions; positive bias means overprediction."""
    error = predicted - labels
    def totals(values: np.ndarray, count: int) -> list:
        return [count, float(np.abs(values).sum()), float((values * values).sum()), float(values.sum())]

    return {
        **candidate.describe(), "partition": "validation", "forecasts": error.shape[0],
        "training_rows": used[0], "training_rows_final": used[-1], "fits": len(used),
        **error_metrics(totals(error, error.size)),
        "by_horizon": [
            {"lead_hours": (bucket + 1) * STEP.total_seconds() / 3600,
             **error_metrics(totals(column, error.shape[0]))}
            for bucket, column in enumerate(error.T)
        ],
        "by_month": [], "by_hour": [],
    }


def select(rows: list[dict], candidates: tuple[Candidate, ...] = CANDIDATES) -> dict:
    """Prefer the simplest configuration unless another clears the stated margin."""
    mae = {row["model"]: row["mae_gco2e_per_kwh"] for row in rows}
    plainest = min(item.complexity for item in candidates)
    simplest = min(
        (item for item in candidates if item.complexity == plainest),
        key=lambda item: mae[item.name],
    )
    best = min(candidates, key=lambda item: mae[item.name])
    chosen = best if mae[best.name] <= mae[simplest.name] * (1 - MIN_IMPROVEMENT) else simplest
    return {
        **chosen.describe(), "selected_on": "validation", "frozen": chosen.refit is None,
        "feature_names": list(FEATURE_LAYOUT),
        "selection_rule": "lowest validation MAE, requiring a "
                          f"{MIN_IMPROVEMENT:.0%} relative gain over the simplest configuration",
        "simplest_candidate": simplest.name, "simplest_mae_gco2e_per_kwh": mae[simplest.name],
        "best_candidate": best.name, "best_mae_gco2e_per_kwh": mae[best.name],
        "selected_mae_gco2e_per_kwh": mae[chosen.name],
    }


def main() -> int:
    args = base_parser(__doc__).parse_args()

    def work() -> None:
        actual, protocol, actual_hash = load_inputs(args.actual, args.protocol)
        train = design(actual, protocol, "train")
        validation = design(actual, protocol, "validation")
        pooled = tuple(
            np.concatenate([left, right]) for left, right in zip(train, validation, strict=True)
        )
        rows = []
        for candidate in CANDIDATES:
            predicted, used = predict_validation(candidate, protocol, pooled, validation)
            row = score(candidate, predicted, validation[2], used)
            rows.append(row)
            span = f"rows={used[0]}" + (f"..{used[-1]} over {len(used)} fits" if len(used) > 1 else "")
            report(candidate.name, row, f", {span}")
        selected = select(rows)
        write_evaluation(args.output_dir, "selection", rows, {
            **provenance(args, protocol, actual_hash),
            "selected": selected, "evaluation": rows,
            "candidates": [candidate.describe() for candidate in CANDIDATES],
            "refit_rule": "a training row is used only once its whole label window is observable",
        })
        (args.output_dir / "selected_model.json").write_text(
            json.dumps(selected, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8",
        )
        print(f"Selected {selected['model']} "
              f"({'frozen' if selected['frozen'] else 'refitted'}); results in {args.output_dir}")

    return run_entry_point("Model selection", work)


if __name__ == "__main__":
    raise SystemExit(main())
