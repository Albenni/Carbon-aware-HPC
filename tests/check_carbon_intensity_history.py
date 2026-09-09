"""Small offline check: python tests/check_carbon_intensity_history.py."""

from dataclasses import replace
from datetime import datetime, timezone
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from carbon_intensity import FIFTEEN_MINUTES as STEP, CarbonIntensitySample as Sample
from carbon_intensity import ElectricityMapsClient, TimeSeriesCarbonIntensityProvider as Series
from carbon_intensity.history import ACTUAL_METADATA, merge_actual_caches
from carbon_intensity.protocol import TemporalProtocol


def rejects(action):
    try:
        action()
    except (ValueError, RuntimeError):
        return
    raise AssertionError("unsafe input was accepted")


def main():
    start = datetime(2020, 1, 1, tzinfo=timezone.utc)
    points = tuple(Sample(start + i * STEP, 100 + i, i == 0) for i in range(12))
    actual = Series(points, metadata=ACTUAL_METADATA)
    protocol = TemporalProtocol(start + 8 * STEP, start + 12 * STEP, start, start + 4 * STEP)
    assert [len(part) for part in protocol.split(actual).values()] == [4, 4, 4]
    training = protocol.example(actual, start + STEP, STEP, 2 * STEP, partition="train")
    assert training.features == points[:1] and training.targets == points[1:3]
    assert protocol.history(actual, start + STEP + STEP / 2, STEP) == points[:1]
    delayed = replace(protocol, observation_delay=STEP)
    assert delayed.history(actual, start + 2 * STEP, STEP) == points[:1]
    for action in (
        lambda: protocol.validate_features(points[1:2], start + STEP),
        lambda: delayed.validate_features(points[:1], start + STEP),
        lambda: protocol.validate_features(points[:1], start.replace(tzinfo=None)),
        lambda: protocol.example(actual, start + 3 * STEP, STEP, 2 * STEP, partition="train"),
        lambda: protocol.example(actual, start + 7 * STEP, STEP, 2 * STEP, partition="validation"),
        lambda: protocol.example(actual, start + 8 * STEP, STEP, STEP, partition="validation"),
    ):
        rejects(action)

    with TemporaryDirectory() as directory:
        path, model = Path(directory) / "actual.json", Path(directory) / "model.json"
        actual.save(path)
        merged = merge_actual_caches([path, path], start, start + 12 * STEP)
        assert merged.samples == points and merged.metadata["quality"]["identical_overlaps"] == 12
        assert merged.metadata["quality"]["by_year"]["2020"]["estimated"] == 1
        assert len(merged.metadata["source_caches"][0]["sha256"]) == 64
        conflict = Series((replace(points[0], intensity_gco2e_per_kwh=0), *points[1:]), metadata=ACTUAL_METADATA)
        other = conflict.save(Path(directory) / "conflict.json")
        rejects(lambda: merge_actual_caches([path, other], start, start + 12 * STEP))
        protocol.save_training_metadata(model, "check", [training])
        saved = json.loads(model.read_text())
        assert saved["training_cutoff"] == points[2].timestamp.isoformat()
        assert saved["training_available_at"] == points[3].timestamp.isoformat()
        test = protocol.example(actual, start + 8 * STEP, STEP, STEP, partition="test")
        rejects(lambda: protocol.save_training_metadata(model, "check", [test]))
        rejects(lambda: protocol.save_training_metadata(model, "check", [replace(training, features=points[2:3])]))
        for bad in (
            Series(points[:1] + points[2:], metadata=ACTUAL_METADATA),
            Series(points, metadata={**ACTUAL_METADATA, "flow_traced": False}),
            Series(points, metadata={**ACTUAL_METADATA, "signal": "forecast carbon intensity"}),
        ):
            bad.save(path)
            rejects(lambda: merge_actual_caches([path], start, start + 12 * STEP))

    row = {"datetime": start.isoformat(), "carbonIntensity": 100, "flowTraced": True}
    client = ElectricityMapsClient("test", transport=lambda *_: {"data": [row, row]})
    assert client.fetch_actual_range(start, start + STEP).metadata["duplicate_sample_count"] == 1
    row["flowTraced"] = False
    rejects(lambda: client.fetch_actual_range(start, start + STEP))
    print("Historical data and temporal leakage checks passed.")


if __name__ == "__main__":
    main()
