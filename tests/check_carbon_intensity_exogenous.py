"""Small offline check: python tests/check_carbon_intensity_exogenous.py."""

import contextlib
import io
import math
import unittest
from datetime import datetime, timedelta, timezone
from pathlib import Path
from tempfile import TemporaryDirectory
from urllib.error import HTTPError
from urllib.parse import parse_qs, urlsplit

from carbon_intensity import ElectricityMapsClient, ElectricityMapsError
from carbon_intensity.exogenous import build, download, quality


STEP = timedelta(minutes=15)
BASE = datetime(2019, 6, 1, tzinfo=timezone.utc)
END = BASE + timedelta(days=3, hours=1)  # two requests: three days, then one hour


def row(signal: str, time: str, index: int) -> dict:
    load = {"zone": "IT-NO", "datetime": time, "value": 1_600, "unit": "MW", "flowTraced": True}
    return {
        "electricity-mix": {"datetime": time, "estimationMethod": "MEASURED", "mix": {
            "gas": 1_000, "hydro": 500, "wind": 0, "solar": 0,
            "hydro storage": {"charge": 30, "discharge": None},
            "flows": {"imports": 150, "exports": 20},
        }},
        # 100 MW in from AT, 50 in from SI, 20 out to IT-CNO, as the API labels them.
        "electricity-flows": {"datetime": time, "export": {"AT": 100}, "import": {"SI": 50, "IT-CNO": 20}},
        "total-load": {**load, "source": "MEASURED", "isEstimated": index == 3},
        "net-load": {**load, "isEstimated": False, "source": (
            "Electricity Maps Estimation" if index == 0 else "entsoe.eu"
        )},
        "price-day-ahead": {**load, "value": 50.0, "unit": "EUR/MWh"},
    }[signal]


def transport(url: str, headers: dict) -> dict:
    signal = urlsplit(url).path.split("/")[-2]
    start = datetime.fromisoformat(parse_qs(urlsplit(url).query)["start"][0])
    if signal == "net-load" and start == BASE:
        raise ElectricityMapsError("no data") from HTTPError(url, 404, "Not Found", None, None)
    rows = [row(signal, f"{start + i * STEP:%Y-%m-%dT%H:%M:%S.000Z}", i) for i in range(4)]
    return {"zone": "IT-NO", "temporalGranularity": "15_minutes", "unit": "MW", "data": rows + rows[:1]}


def denied(url: str, headers: dict) -> dict:
    raise ElectricityMapsError("denied") from HTTPError(url, 401, "Unauthorized", None, None)


def main() -> None:
    with TemporaryDirectory() as directory:
        raw = Path(directory)
        client = ElectricityMapsClient("test", transport=transport)
        assert download(client, raw, BASE, END) == 10
        assert download(client, raw, BASE, END) == 0, "the raw cache must be reused"
        try:
            download(ElectricityMapsClient("test", transport=denied), raw / "denied", BASE, END)
        except ElectricityMapsError:
            pass
        else:
            raise AssertionError("only a 404 may be recorded as an empty interval")
        table, stats = build(raw, BASE, END)
        report = quality(table, stats)

    first = table.iloc[0]
    assert (first.net_import_at_mw, first.net_import_si_mw, first.net_import_it_cno_mw) == (100, 50, -20)
    assert report["checks"]["border_flows_add_up_to_mix_totals"]["share"] == 1.0
    assert report["checks"]["total_load_equals_mix_balance"]["share"] == 1.0
    assert stats["net-load"]["no_data_requests"] == ["20190601_20190604"]
    assert stats["price-day-ahead"]["identical_duplicates"] == 2
    # An estimate is blanked whichever field admits it; its flag stays.
    assert math.isnan(table.total_load_mw.iloc[3]) and table.total_load_is_estimated.iloc[3]
    net_load = table.net_load_mw[END - timedelta(hours=1):]
    assert math.isnan(net_load.iloc[0]) and net_load.iloc[1] == 1_600
    # A bucket carries its whole hour; the price is public at 14:00 Rome the day before.
    assert table.measured_available_at.iloc[1] == BASE + timedelta(hours=1)
    assert table.price_available_at.iloc[0] == datetime(2019, 5, 31, 12, tzinfo=timezone.utc)
    assert report["signals"]["total-load"]["buckets_returned"] == 8 and len(table) == 292
    print("Exogenous signal checks passed.")


class ExogenousSignalChecks(unittest.TestCase):
    """Make ``unittest discover`` run the script above, which prints its summary."""

    def test_checks_pass(self) -> None:
        with contextlib.redirect_stdout(io.StringIO()):
            main()


if __name__ == "__main__":
    main()
