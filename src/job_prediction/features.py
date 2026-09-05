"""Submission-time feature groups, including causal per-user history.

Every column produced here is computable at the instant a job is submitted.
The history features are the only ones that read past targets, and they read
them through :func:`causal_group_stats`, which admits a past job only once its
``end_time`` has passed the current job's ``submit_time``. Nothing in this
module touches the current job's own outcome.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from .data import (
    AVERAGE_POWER_WATTS,
    COMPLETION_TIME,
    DURATION_SECONDS,
    ENERGY_KWH,
    SUBMIT_TIME,
)


# Requested walltime is recorded in minutes; the targets are in seconds.
_TIME_LIMIT_SECONDS = 60.0
_RECENT_WINDOW = 8
_ALL_JOBS = "_all"
# Scope name -> (grouping keys, how many recent jobs the short window averages).
# The fingerprint is the exact resource request: users on this trace resubmit
# identical requests constantly, so its history is the sharpest signal.
# The global scope has no key at all and tracks the workload regime, which is
# what shifts between the training period and the held-out one.
_HISTORY_SCOPES: dict[str, tuple[tuple[str, ...], int]] = {
    "user": (("user_id",), _RECENT_WINDOW),
    "signature": (("user_id", "time_limit", "num_nodes_req"), _RECENT_WINDOW),
    "fingerprint": (
        (
            "user_id",
            "time_limit",
            "num_nodes_req",
            "num_cores_req",
            "mem_req",
            "num_gpus_req",
        ),
        _RECENT_WINDOW,
    ),
    "global": ((_ALL_JOBS,), 512),
}
_HISTORY_TARGETS = (DURATION_SECONDS, AVERAGE_POWER_WATTS, ENERGY_KWH)

FEATURE_GROUPS = (
    "base",
    "derived",
    "user_id",
    "user_hist",
    "signature_hist",
    "fingerprint_hist",
    "global_hist",
    "queue",
)
_HISTORY_GROUPS = {
    "user_hist": "user",
    "signature_hist": "signature",
    "fingerprint_hist": "fingerprint",
    "global_hist": "global",
}


@dataclass(frozen=True, slots=True)
class FeatureSpec:
    """Which feature groups a candidate model is allowed to see."""

    groups: tuple[str, ...] = ("base",)
    history_targets: tuple[str, ...] = _HISTORY_TARGETS
    recent_window: int = _RECENT_WINDOW
    name: str = field(default="")

    def __post_init__(self) -> None:
        unknown = set(self.groups).difference(FEATURE_GROUPS)
        if unknown:
            raise ValueError(f"unknown feature groups: {', '.join(sorted(unknown))}")
        if not self.groups:
            raise ValueError("at least one feature group is required")
        if self.recent_window < 1:
            raise ValueError("recent_window must be positive")
        if not self.name:
            object.__setattr__(self, "name", "+".join(self.groups))

    @property
    def history_scopes(self) -> tuple[str, ...]:
        return tuple(
            scope for group, scope in _HISTORY_GROUPS.items() if group in self.groups
        )

    @property
    def uses_history(self) -> bool:
        return bool(self.history_scopes)


def causal_group_stats(
    frame: pd.DataFrame,
    *,
    keys: tuple[str, ...],
    value_column: str,
    prefix: str,
    recent_window: int = _RECENT_WINDOW,
) -> pd.DataFrame:
    """Expanding statistics over same-key jobs already finished at submission.

    For every row the admissible history is the set of rows sharing ``keys``
    whose ``end_time`` is at or before this row's ``submit_time``. A job never
    sees itself, never sees a job that is still running, and never sees the
    future. The statistics are taken on ``log1p`` of the value, so a mean is a
    geometric mean and is robust to the workload's heavy right tail.
    """

    for column in (SUBMIT_TIME, COMPLETION_TIME, value_column, *keys):
        if column not in frame.columns:
            raise ValueError(f"causal statistics need column {column!r}")

    submit = pd.to_datetime(frame[SUBMIT_TIME], utc=True).to_numpy("datetime64[ns]")
    end = pd.to_datetime(frame[COMPLETION_TIME], utc=True).to_numpy("datetime64[ns]")
    values = np.log1p(
        pd.to_numeric(frame[value_column], errors="coerce").to_numpy(dtype=float)
    )

    count = np.zeros(len(frame))
    mean = np.full(len(frame), np.nan)
    std = np.full(len(frame), np.nan)
    last = np.full(len(frame), np.nan)
    recent = np.full(len(frame), np.nan)
    maximum = np.full(len(frame), np.nan)
    age = np.full(len(frame), np.nan)

    grouped = frame.groupby(list(keys), sort=False, dropna=False, observed=True)
    for positions in grouped.indices.values():
        order = positions[np.argsort(end[positions], kind="stable")]
        ordered_end = end[order]
        ordered_values = values[order]
        cumulative = np.concatenate(([0.0], np.cumsum(ordered_values)))
        squares = np.concatenate(([0.0], np.cumsum(np.square(ordered_values))))
        running_max = np.concatenate(
            ([np.nan], np.maximum.accumulate(ordered_values))
        )
        # ``side="right"`` admits a job that finished exactly at this
        # submission instant, matching the split's own availability rule.
        available = np.searchsorted(ordered_end, submit[positions], side="right")

        seen = available.astype(float)
        observed = available > 0
        count[positions] = seen
        totals = cumulative[available]
        with np.errstate(invalid="ignore", divide="ignore"):
            group_mean = np.where(observed, totals / np.maximum(seen, 1.0), np.nan)
            variance = np.where(
                observed,
                squares[available] / np.maximum(seen, 1.0) - np.square(group_mean),
                np.nan,
            )
        mean[positions] = group_mean
        std[positions] = np.sqrt(np.clip(variance, 0.0, None))
        maximum[positions] = running_max[available]
        last[positions] = np.where(
            observed, ordered_values[np.maximum(available - 1, 0)], np.nan
        )
        window = np.minimum(available, recent_window)
        recent[positions] = np.where(
            observed,
            (totals - cumulative[available - window]) / np.maximum(window, 1),
            np.nan,
        )
        elapsed = (
            submit[positions] - ordered_end[np.maximum(available - 1, 0)]
        ) / np.timedelta64(1, "s")
        age[positions] = np.where(observed, elapsed, np.nan)

    return pd.DataFrame(
        {
            f"{prefix}_count": np.log1p(count),
            f"{prefix}_mean": mean,
            f"{prefix}_std": std,
            f"{prefix}_last": last,
            f"{prefix}_recent": recent,
            f"{prefix}_max": maximum,
            f"{prefix}_age": np.log1p(np.clip(age, 0.0, None)),
        },
        index=frame.index,
    )


def causal_inflight(
    frame: pd.DataFrame,
    *,
    keys: tuple[str, ...],
    prefix: str,
) -> pd.DataFrame:
    """How much of the group is already submitted and still unfinished.

    Both counts are observable at the submission instant: a job that has been
    submitted is on the queue, and a job that has not yet ended is visibly
    still there. No outcome value is read.
    """

    submit = pd.to_datetime(frame[SUBMIT_TIME], utc=True).to_numpy("datetime64[ns]")
    end = pd.to_datetime(frame[COMPLETION_TIME], utc=True).to_numpy("datetime64[ns]")
    submitted = np.zeros(len(frame))
    finished = np.zeros(len(frame))

    grouped = frame.groupby(list(keys), sort=False, dropna=False, observed=True)
    for positions in grouped.indices.values():
        group_submit = np.sort(submit[positions])
        group_end = np.sort(end[positions])
        moment = submit[positions]
        # A job does not count itself: "side=left" excludes equal submissions,
        # which is the conservative choice at a tie.
        submitted[positions] = np.searchsorted(group_submit, moment, side="left")
        finished[positions] = np.searchsorted(group_end, moment, side="right")

    return pd.DataFrame(
        {
            f"{prefix}_submitted": np.log1p(submitted),
            f"{prefix}_inflight": np.log1p(np.clip(submitted - finished, 0.0, None)),
        },
        index=frame.index,
    )


def _safe_ratio(numerator: np.ndarray, denominator: np.ndarray) -> np.ndarray:
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(denominator > 0.0, numerator / np.maximum(denominator, 1e-9), np.nan)


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=float)


def _base_columns(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    columns: dict[str, np.ndarray] = {}
    for name in (
        "time_limit",
        "num_nodes_req",
        "num_cores_req",
        "num_tasks",
        "cores_per_task",
        "num_gpus_req",
        "mem_req",
        "priority",
    ):
        columns[f"log1p_{name}"] = np.log1p(np.clip(_numeric(frame, name), 0.0, None))
    columns["num_tasks_missing"] = np.isnan(_numeric(frame, "num_tasks")).astype(float)

    timestamps = pd.to_datetime(frame[SUBMIT_TIME], utc=True)
    hour = (
        timestamps.dt.hour.to_numpy(dtype=float)
        + timestamps.dt.minute.to_numpy(dtype=float) / 60.0
    )
    weekday = timestamps.dt.dayofweek.to_numpy(dtype=float)
    columns["submit_hour"] = hour
    columns["submit_hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
    columns["submit_hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
    columns["submit_weekday"] = weekday
    columns["submit_weekday_sin"] = np.sin(2.0 * np.pi * weekday / 7.0)
    columns["submit_weekday_cos"] = np.cos(2.0 * np.pi * weekday / 7.0)

    # QoS has four recorded values; one indicator each keeps the encoding
    # identical whichever subset a training window happens to contain.
    qos = frame["qos"].astype("string").fillna("<missing>").to_numpy()
    for category in ("1", "4", "8", "11"):
        columns[f"qos_{category}"] = (qos == category).astype(float)
    return columns


def _derived_columns(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    nodes = _numeric(frame, "num_nodes_req")
    cores = _numeric(frame, "num_cores_req")
    memory = _numeric(frame, "mem_req")
    gpus = _numeric(frame, "num_gpus_req")
    tasks = _numeric(frame, "num_tasks")
    cores_per_task = _numeric(frame, "cores_per_task")
    limit = _numeric(frame, "time_limit") * _TIME_LIMIT_SECONDS

    columns = {
        "cores_per_node": _safe_ratio(cores, nodes),
        "memory_per_node": _safe_ratio(memory, nodes),
        "memory_per_core": _safe_ratio(memory, cores),
        "gpus_per_node": _safe_ratio(gpus, nodes),
        "tasks_per_node": _safe_ratio(tasks, nodes),
        "cores_per_task_ratio": _safe_ratio(cores, np.where(tasks > 0, tasks, np.nan)),
        # Requested node-seconds is the size of the reservation the scheduler
        # must find, and is the quantity the carbon policy actually trades.
        "log1p_node_seconds": np.log1p(np.clip(nodes * limit, 0.0, None)),
        "log1p_core_seconds": np.log1p(np.clip(cores * limit, 0.0, None)),
        "task_core_mismatch": (
            np.abs(cores - tasks * cores_per_task) > 0.5
        ).astype(float),
        "shared_ok": (frame["shared"].astype("string") == "OK").to_numpy(dtype=float),
        "req_switch": _numeric(frame, "req_switch"),
        "threads_per_core_present": (
            ~np.isnan(_numeric(frame, "threads_per_core"))
        ).astype(float),
    }
    group = frame["group_id"].astype("string").fillna("<missing>").to_numpy()
    for category in sorted(set(group)):
        columns[f"group_{category}"] = (group == category).astype(float)

    timestamps = pd.to_datetime(frame[SUBMIT_TIME], utc=True)
    columns["submit_month"] = timestamps.dt.month.to_numpy(dtype=float)
    columns["submit_day"] = timestamps.dt.day.to_numpy(dtype=float)
    return columns


def _history_columns(
    frame: pd.DataFrame,
    spec: FeatureSpec,
) -> dict[str, np.ndarray]:
    columns: dict[str, np.ndarray] = {}
    limit_seconds = _numeric(frame, "time_limit") * _TIME_LIMIT_SECONDS
    working = frame.copy()
    working[_ALL_JOBS] = 0
    # A user's historical walltime utilisation converts the requested limit,
    # which is known now, into an expected duration. It is the single most
    # informative causal statistic on this trace.
    working["_walltime_use"] = _safe_ratio(
        _numeric(frame, DURATION_SECONDS), limit_seconds
    )

    for scope in spec.history_scopes:
        keys, window = _HISTORY_SCOPES[scope]
        for value_column in (*spec.history_targets, "_walltime_use"):
            name = "walltime_use" if value_column == "_walltime_use" else value_column
            stats = causal_group_stats(
                working,
                keys=keys,
                value_column=value_column,
                prefix=f"{scope}_{name}",
                recent_window=window if scope == "global" else spec.recent_window,
            )
            columns.update({column: stats[column].to_numpy() for column in stats})
        # Turning the historical utilisation back into seconds gives the model
        # a ready-made duration estimate it only has to correct.
        for statistic in ("mean", "recent", "last"):
            key = f"{scope}_walltime_use_{statistic}"
            if key not in columns:
                continue
            expected = np.expm1(columns[key]) * limit_seconds
            columns[f"{scope}_expected_duration_{statistic}"] = np.log1p(
                np.clip(expected, 0.0, None)
            )
    return columns


def _queue_columns(frame: pd.DataFrame) -> dict[str, np.ndarray]:
    working = frame.copy()
    working[_ALL_JOBS] = 0
    columns: dict[str, np.ndarray] = {}
    for scope, keys in (("user", ("user_id",)), ("global", (_ALL_JOBS,))):
        counts = causal_inflight(working, keys=keys, prefix=f"{scope}_queue")
        columns.update({column: counts[column].to_numpy() for column in counts})
    return columns


def build_features(frame: pd.DataFrame, spec: FeatureSpec) -> pd.DataFrame:
    """Assemble the requested feature groups into one numeric frame."""

    columns: dict[str, np.ndarray] = {}
    if "base" in spec.groups:
        columns.update(_base_columns(frame))
    if "derived" in spec.groups:
        columns.update(_derived_columns(frame))
    if "user_id" in spec.groups:
        columns["user_id"] = _numeric(frame, "user_id")
    if spec.uses_history:
        columns.update(_history_columns(frame, spec))
    if "queue" in spec.groups:
        columns.update(_queue_columns(frame))
    if not columns:
        raise ValueError("the feature specification produced no columns")
    return pd.DataFrame(columns, index=frame.index)
