"""Exogenous grid signals for IT-NO: raw Electricity Maps history and a clean table.

The carbon forecast only sees its own past. This module fetches what drives that
series - production by source, the flow on every border, load, net load and the
day-ahead price - for the training years, keeps every API answer verbatim, and
aligns the values on the forecast's 15-minute UTC grid.

Nothing here is a feature yet. The table holds values plus what a feature
builder needs to stay causal:

* in these years mix, flows and load are hourly values repeated on the
  15-minute grid, and net load is computed from that load, so a bucket is known
  when its *hour* ends, not when the bucket does: ``measured_available_at``,
  before the protocol's observation delay;
* the day-ahead price of a delivery day is published the day before, so each
  bucket carries its own ``price_available_at``;
* values the API says it estimated are blanked, flags kept: an interpolated
  estimate is computed from values published after the gap it fills.

Run ``python -m carbon_intensity.exogenous`` to fetch what is missing and rebuild
the table and its quality report; ``--offline`` only rebuilds.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

import pandas as pd

from .electricity_maps import (
    DEFAULT_ZONE as ZONE, TEMPORAL_GRANULARITY, ElectricityMapsClient, ElectricityMapsError,
    _parse_api_timestamp, read_api_key,
)
from .history import HISTORY_START, VALIDATION_START
from .series import bucket_start


SIGNALS = ("electricity-mix", "electricity-flows", "total-load", "net-load", "price-day-ahead")
UNITS = {"price-day-ahead": "EUR/MWh"}  # every other signal is in MW
# The API refuses longer past-range requests at 15-minute granularity.
REQUEST_SPAN = timedelta(days=3)
# GME publishes day-ahead (MGP) results by 12:55 local time on the day before
# delivery. No archive of the actual publication times exists, so the price is
# taken as known from 14:00: the extra hour covers delayed market coupling.
PRICE_TIMEZONE = "Europe/Rome"
PRICE_PUBLISHED = timedelta(hours=14)


def chunks(start: datetime, end: datetime) -> list[tuple[datetime, datetime]]:
    bounds = []
    while start < end:
        bounds.append((start, min(start + REQUEST_SPAN, end)))
        start = bounds[-1][1]
    return bounds


def raw_path(raw_dir: Path, signal: str, start: datetime, end: datetime) -> Path:
    return raw_dir / signal / f"{start:%Y%m%d}_{end:%Y%m%d}.json"


def download(
    client: ElectricityMapsClient, raw_dir: Path,
    start: datetime = HISTORY_START, end: datetime = VALIDATION_START,
) -> int:
    """Cache one verbatim response per signal and request; return requests made.

    Existing files are reused, so an interrupted run resumes. A 404 is the API
    saying the interval holds no data: that answer is the raw record, kept so the
    gap is documented instead of retried. Any other error stops the run.
    """
    made = 0
    for signal in SIGNALS:
        for lo, hi in chunks(start, end):
            path = raw_path(raw_dir, signal, lo, hi)
            if path.exists():
                continue
            try:
                payload = client.fetch_past_range(signal, lo, hi)
            except ElectricityMapsError as error:
                if getattr(error.__cause__, "code", None) != 404:
                    raise
                payload = {"error": str(error)}
            path.parent.mkdir(parents=True, exist_ok=True)
            temporary = path.with_suffix(".tmp")
            temporary.write_text(json.dumps(payload) + "\n", encoding="utf-8")
            temporary.replace(path)
            made += 1
            if made % 100 == 0:
                print(f"{made} requests, at {signal} {lo:%Y-%m-%d}", flush=True)
    return made


def _slug(name: str) -> str:
    return name.lower().replace(" ", "_").replace("-", "_")


def _net_imports(row: dict) -> dict:
    """Signed flow on each border, positive when IT-NO imports.

    The API files a neighbour under ``import`` or ``export`` by the sign of the
    flow on the alphabetically ordered pair (``AT->IT-NO``, ``IT-NO->SI``), not
    from IT-NO's side: 132 MW arriving from AT comes back as ``export: {AT: 132}``.
    Read that way the borders add up to the zone's import and export totals in
    the mix; read literally they do not. ``quality`` rechecks it on every row.
    """
    net: dict[str, float | None] = {}
    for label, sign in (("export", 1), ("import", -1)):
        for zone, value in (row.get(label) or {}).items():
            column = f"net_import_{_slug(zone)}_mw"
            if column in net:
                raise ValueError(f"{zone} is both an import and an export at {row['datetime']}")
            net[column] = None if value is None else (sign if zone < ZONE else -sign) * value
    return net


def _flatten(signal: str, row: dict) -> dict:
    if signal == "electricity-mix":
        out = {"mix_estimation_method": row.get("estimationMethod"),
               "mix_breakdown_type": row.get("breakdownType")}
        for source, value in row["mix"].items():
            if isinstance(value, dict):
                out.update({f"mix_{_slug(source)}_{part}_mw": v for part, v in value.items()})
            else:
                out[f"mix_{_slug(source)}_mw"] = value
        return out
    if signal == "electricity-flows":
        return _net_imports(row)
    if signal == "price-day-ahead":
        return {"price_day_ahead_eur_per_mwh": row["value"]}
    # ``source`` stays: on net load it names an estimation on rows whose
    # ``isEstimated`` is false.
    name = _slug(signal)
    return {f"{name}_mw": row["value"], f"{name}_is_estimated": row.get("isEstimated"),
            f"{name}_estimation_method": row.get("estimationMethod"),
            f"{name}_source": row.get("source")}


def _read_signal(raw_dir: Path, signal: str, start: datetime, end: datetime):
    """One signal's rows keyed by bucket, checked against the request that returned them."""
    rows: dict[datetime, dict] = {}
    info = {"requests": 0, "no_data_requests": [], "identical_duplicates": 0,
            "updated_at": [], "sources": {}}
    for lo, hi in chunks(start, end):
        path = raw_path(raw_dir, signal, lo, hi)
        if not path.exists():
            raise ValueError(f"missing raw response {path}; run without --offline")
        payload = json.loads(path.read_text(encoding="utf-8"))
        info["requests"] += 1
        if "error" in payload:
            info["no_data_requests"].append(path.stem)
            continue
        for row in payload["data"]:
            meta = {**payload, **row}
            if (meta.get("zone") != ZONE or meta.get("temporalGranularity") != TEMPORAL_GRANULARITY
                    or meta.get("unit") != UNITS.get(signal, "MW") or meta.get("flowTraced", True) is not True):
                raise ValueError(f"{path}: unexpected zone, granularity, unit or flow tracing")
            timestamp = _parse_api_timestamp(row["datetime"])
            if timestamp != bucket_start(timestamp) or not lo <= timestamp < hi:
                raise ValueError(f"{path}: {row['datetime']} is off the grid or outside its request")
            record = _flatten(signal, row)
            if timestamp in rows:
                if rows[timestamp] != record:
                    raise ValueError(f"{signal}: conflicting values at {row['datetime']}")
                info["identical_duplicates"] += 1
                continue
            rows[timestamp] = record
            info["updated_at"].append(row.get("updatedAt"))
            if "source" in row:
                info["sources"][row["source"]] = info["sources"].get(row["source"], 0) + 1
    updated = sorted(filter(None, info.pop("updated_at")))
    info["updated_at_range"] = [updated[0], updated[-1]] if updated else None
    return pd.DataFrame.from_records(list(rows.values()), index=pd.DatetimeIndex(list(rows))), info


def price_available_at(index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    """When the day-ahead price of each bucket's local delivery day was public."""
    delivery_day = index.tz_convert(PRICE_TIMEZONE).tz_localize(None).normalize()
    published = delivery_day - pd.Timedelta(days=1) + PRICE_PUBLISHED
    return published.tz_localize(PRICE_TIMEZONE).tz_convert("UTC")


def _estimated(frame: pd.DataFrame) -> pd.Series:
    """Rows Electricity Maps calls estimated in any field, not only ``isEstimated``."""
    flagged = pd.Series(False, index=frame.index)
    for column in frame:
        if column.endswith("_is_estimated"):
            flagged |= frame[column].eq(True)
        elif column.endswith("_estimation_method"):
            flagged |= frame[column].notna() & frame[column].ne("MEASURED")
        elif column.endswith("_source"):
            flagged |= frame[column].astype("string").str.contains("estimat", case=False, na=False)
    return flagged.astype(bool)


def build(raw_dir: Path, start: datetime = HISTORY_START, end: datetime = VALIDATION_START):
    """The clean table on the forecast grid, and per-signal download facts.

    Buckets the API did not return stay missing: nothing is filled, carried
    forward or interpolated, and a null the API sent stays a null. Values the
    API says it estimated are blanked and their flags kept: the ones checked
    average the hours either side of the gap, so they only existed after it.
    """
    grid = pd.date_range(start, end, freq="15min", inclusive="left", name="timestamp")
    frames, stats = [], {}
    for signal in SIGNALS:
        frame, info = _read_signal(raw_dir, signal, start, end)
        numeric = [column for column in frame if column.endswith(("_mw", "_per_mwh"))]
        frame[numeric] = frame[numeric].astype(float)
        estimated = _estimated(frame)
        frame.loc[estimated, numeric] = float("nan")
        info.update(columns=list(frame.columns), returned=grid.isin(frame.index),
                    estimated_blanked=int(estimated.sum()))
        frames.append(frame.reindex(grid))
        stats[signal] = info
    table = pd.concat(frames, axis=1)
    for column in table:
        if column.endswith("_is_estimated"):
            table[column] = table[column].astype("boolean")
        elif not column.endswith(("_mw", "_per_mwh")):
            table[column] = table[column].astype("string")
    # Mix, flows and load are hourly values repeated on the 15-minute grid, and
    # net load is computed from that load: a bucket is known once its hour ends.
    table["measured_available_at"] = grid.floor("h") + pd.Timedelta(hours=1)
    table["price_available_at"] = price_available_at(grid)
    return table, stats


def _by_year(series: pd.Series) -> dict:
    counts = series.groupby(series.index.year).value_counts()
    return {str(year): {str(value): int(n) for value, n in counts[year].items()}
            for year in counts.index.get_level_values(0).unique()}


def _gaps(missing: pd.Series) -> dict:
    run = (missing != missing.shift()).cumsum()[missing]
    sizes = run.value_counts()
    return {"gaps": len(sizes), "largest_gaps": [
        {"start": run.index[run == key][0].isoformat(), "buckets": int(size)}
        for key, size in sizes.head(5).items()
    ]}


def _signal_quality(table: pd.DataFrame, info: dict) -> dict:
    returned = pd.Series(info["returned"], index=table.index)
    values = table.loc[returned, info["columns"]]
    numeric = table[info["columns"]].select_dtypes("number")
    nulls = values.isna().groupby(values.index.year).sum()
    partial = [column for column in numeric if 0 < values[column].isna().sum() < len(values)]
    hour = table.index.floor("h")
    # Native resolution: an hour whose four quarter-hours are identical is an
    # hourly value repeated on the 15-minute grid.
    constant = (numeric.groupby(hour).nunique(dropna=False) <= 1).all(axis=1)
    constant = constant[returned.groupby(hour).sum() == 4]
    return {
        "requests": info["requests"],
        "no_data_requests": len(info["no_data_requests"]),
        "no_data_range": info["no_data_requests"][:1] + info["no_data_requests"][-1:],
        "buckets_returned": int(returned.sum()),
        "coverage": round(float(returned.mean()), 6),
        "first_returned": returned.idxmax().isoformat() if returned.any() else None,
        "last_returned": returned[::-1].idxmax().isoformat() if returned.any() else None,
        "missing_buckets_by_year": {
            str(year): int(n) for year, n in (~returned).groupby(table.index.year).sum().items()
        },
        **_gaps(~returned),
        "identical_duplicates": info["identical_duplicates"],
        "hourly_constant_share_by_year": {
            str(year): round(float(share), 4)
            for year, share in constant.groupby(constant.index.year).mean().items()
        },
        "estimated_blanked": info["estimated_blanked"],
        "nulls_in_returned_rows_by_year": {
            column: {str(year): int(n) for year, n in counts.items() if n}
            for column, counts in nulls.items() if counts.any()
        },
        "first_value": {column: values[column].first_valid_index().isoformat() for column in partial},
        "flags_by_year": {
            column: _by_year(values[column].astype("string").fillna("null"))
            for column in values if column.endswith(("_estimated", "_method", "_type", "_source"))
        },
        "sources": info["sources"],
        "updated_at_range": info["updated_at_range"],
    }


def _checks(table: pd.DataFrame) -> dict:
    """Cross-signal identities: how independent, and how consistent, the signals are."""
    def share(condition: pd.Series, rows: pd.Series) -> dict:
        return {"rows": int(rows.sum()), "share": round(float(condition[rows].mean()), 6)}

    net = table.filter(like="net_import_")
    imports, exports = net.clip(lower=0).sum(axis=1), (-net).clip(lower=0).sum(axis=1)
    mix = table.filter(regex=r"^mix_.*_mw$")
    charge, discharge = mix.filter(like="_charge_").sum(axis=1), mix.filter(like="_discharge_").sum(axis=1)
    production = mix.drop(columns=mix.filter(regex="storage|flows").columns).sum(axis=1)
    balance = production + table.mix_flows_imports_mw - table.mix_flows_exports_mw - charge + discharge
    measured = table.mix_flows_imports_mw.notna()
    storage = table[["mix_hydro_storage_charge_mw", "mix_hydro_storage_discharge_mw"]].notna()
    load, net_load = table.total_load_mw, table.net_load_mw
    gap = (net_load - (load - table.mix_wind_mw - table.mix_solar_mw)).dropna()
    price = table.price_day_ahead_eur_per_mwh
    return {
        "border_flows_add_up_to_mix_totals": share(
            ((imports - table.mix_flows_imports_mw).abs() <= 1)
            & ((exports - table.mix_flows_exports_mw).abs() <= 1), measured & net.notna().any(axis=1),
        ),
        "total_load_equals_mix_balance": share((load - balance).abs() <= 1, load.notna() & measured),
        "hydro_storage_one_side_set": share(storage.sum(axis=1) == 1, measured),
        "net_load_above_total_load": share(net_load > load, net_load.notna() & load.notna()),
        "net_load_minus_load_less_local_wind_solar_mw": {
            "median": round(float(gap.median()), 1), "p05": round(float(gap.quantile(0.05)), 1),
            "p95": round(float(gap.quantile(0.95)), 1),
        },
        "net_load_estimated_by_source_not_flag": int((
            table.net_load_source.str.contains("Estimation", na=False) & ~table.net_load_is_estimated.fillna(False)
        ).sum()),
        "price_eur_per_mwh": {
            "min": float(price.min()), "max": float(price.max()), "non_positive": int((price <= 0).sum()),
        },
    }


def quality(table: pd.DataFrame, stats: dict) -> dict:
    """Coverage, gaps, duplicates, resolution and flags per signal, plus cross-checks."""
    return {
        "zone": ZONE,
        "grid": {"start": table.index[0].isoformat(),
                 "end": (table.index[-1] + pd.Timedelta(minutes=15)).isoformat(),
                 "buckets": len(table), "step": "15 minutes, UTC, labelled by bucket start"},
        "availability": {
            "measured_available_at": "end of the UTC hour holding the bucket; add the protocol's observation delay",
            "price_available_at": "14:00 Europe/Rome on the day before the local delivery day",
        },
        "signals": {signal: _signal_quality(table, info) for signal, info in stats.items()},
        "checks": _checks(table),
    }


def main() -> int:
    root = Path(__file__).resolve().parents[2]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=root / "data/carbon_intensity/exogenous")
    parser.add_argument("--env-file", type=Path, default=root / ".env")
    parser.add_argument("--offline", action="store_true", help="rebuild from cached raw responses only")
    args = parser.parse_args()
    raw_dir = args.output_dir / "raw"
    try:
        if not args.offline:
            client = ElectricityMapsClient(read_api_key(args.env_file, "ELECTRICITY_MAPS_KEY"))
            print(f"{download(client, raw_dir)} new requests", flush=True)
        table, stats = build(raw_dir)
        report = quality(table, stats)
        table.to_parquet(args.output_dir / "signals.parquet")
        (args.output_dir / "quality.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    except (OSError, RuntimeError, ValueError) as error:
        print(f"Exogenous signals failed: {error}", file=sys.stderr)
        return 1
    for signal, facts in report["signals"].items():
        print(f"{signal}: coverage {facts['coverage']:.2%}, from {facts['first_returned']}, "
              f"{facts['gaps']} gaps")
    print(f"Saved {len(table)} buckets x {table.shape[1]} columns to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
