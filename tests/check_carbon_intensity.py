"""
Deterministic checks for the carbon-intensity provider.
"""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from urllib.parse import parse_qs, urlsplit
from zoneinfo import ZoneInfo


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from carbon_accounting import JobPowerProfile, account_emissions
from carbon_intensity import (
    FIFTEEN_MINUTES,
    CarbonIntensityForecast,
    CarbonIntensitySample,
    ElectricityMapsClient,
    ElectricityMapsError,
    ForecastUnavailableError,
    MissingCarbonIntensityError,
    TimeSeriesCarbonIntensityProvider,
)
from download_carbon_intensity import default_output_path


UTC = timezone.utc
BASE = datetime(2020, 5, 6, 7, 0, tzinfo=UTC)


def sample(minutes: int, value: float) -> CarbonIntensitySample:
    return CarbonIntensitySample(
        timestamp=BASE + timedelta(minutes=minutes),
        intensity_gco2e_per_kwh=value,
    )


def request_interval(url: str) -> tuple[datetime, datetime]:
    """Read the half-open interval from a synthetic Electricity Maps request."""

    query = parse_qs(urlsplit(url).query)
    return (
        datetime.fromisoformat(query["start"][0].replace("Z", "+00:00")),
        datetime.fromisoformat(query["end"][0].replace("Z", "+00:00")),
    )


def complete_api_payload(start: datetime, end: datetime) -> dict[str, object]:
    """Return one deterministic sample for every bucket in ``[start, end)``."""

    data: list[dict[str, object]] = []
    current = start
    while current < end:
        data.append(
            {
                "zone": "IT-NO",
                "datetime": current.isoformat().replace("+00:00", "Z"),
                "carbonIntensity": 250,
                "emissionFactorType": "lifecycle",
                "isEstimated": False,
                "estimationMethod": None,
            }
        )
        current += FIFTEEN_MINUTES
    return {
        "zone": "IT-NO",
        "unit": "gCO2eq/kWh",
        "temporalGranularity": "15_minutes",
        "emissionFactorType": "lifecycle",
        "data": data,
    }


class CarbonIntensityChecks(unittest.TestCase):
    def setUp(self) -> None:
        # Three consecutive buckets make boundaries and range selection clear.
        self.provider = TimeSeriesCarbonIntensityProvider(
            [sample(30, 200.0), sample(0, 100.0), sample(15, 300.0)],
            metadata={"zone": "synthetic"},
        )

    def test_piecewise_constant_lookup_and_timezones(self) -> None:
        """A lookup floors in UTC and changes value only at a bucket edge."""

        self.assertEqual(self.provider.get_actual(BASE), 100.0)
        self.assertEqual(
            self.provider.get_actual(BASE + timedelta(minutes=14, seconds=59)),
            100.0,
        )
        self.assertEqual(
            self.provider.get_actual(BASE + timedelta(minutes=15)),
            300.0,
        )

        # The same instant expressed in Europe/Rome must select the same bucket.
        rome_instant = BASE.astimezone(ZoneInfo("Europe/Rome"))
        self.assertEqual(self.provider.get_actual(rome_instant), 100.0)

    def test_range_is_half_open_and_includes_the_overlapping_first_bucket(self) -> None:
        """An unaligned start includes its bucket; an exact end is excluded."""

        selected = self.provider.get_actual_range(
            BASE + timedelta(minutes=7),
            BASE + timedelta(minutes=30),
        )
        self.assertEqual(
            [point.timestamp for point in selected],
            [BASE, BASE + timedelta(minutes=15)],
        )
        selected_past_boundary = self.provider.get_actual_range(
            BASE + timedelta(minutes=7),
            BASE + timedelta(minutes=30, seconds=1),
        )
        self.assertEqual(len(selected_past_boundary), 3)

    def test_gaps_and_invalid_times_fail_explicitly(self) -> None:
        """No lookup may silently fill a gap or leave known coverage."""

        gapped = TimeSeriesCarbonIntensityProvider([sample(0, 100), sample(30, 200)])
        with self.assertRaises(MissingCarbonIntensityError):
            gapped.get_actual(BASE + timedelta(minutes=15))
        with self.assertRaises(MissingCarbonIntensityError):
            self.provider.get_actual(BASE - timedelta(seconds=1))
        with self.assertRaises(MissingCarbonIntensityError):
            self.provider.get_actual(self.provider.coverage_end)
        with self.assertRaises(ValueError):
            self.provider.get_actual(datetime(2020, 5, 6, 7, 0))
        with self.assertRaises(ValueError):
            self.provider.get_actual_range(BASE, BASE)

    def test_samples_and_series_are_validated(self) -> None:
        """Invalid physical values, grids, and duplicate buckets are rejected."""

        for bad_value in (-1.0, float("nan"), float("inf"), True):
            with self.subTest(value=bad_value), self.assertRaises(
                (TypeError, ValueError)
            ):
                CarbonIntensitySample(BASE, bad_value)

        with self.assertRaises(ValueError):
            TimeSeriesCarbonIntensityProvider([sample(0, 100), sample(0, 200)])
        with self.assertRaises(ValueError):
            TimeSeriesCarbonIntensityProvider(
                [CarbonIntensitySample(BASE + timedelta(minutes=1), 100)]
            )
        with self.assertRaises(TypeError):
            TimeSeriesCarbonIntensityProvider(["not a sample"])  # type: ignore[list-item]

        # Provenance must itself be stable, strict JSON rather than changing
        # shape or accepting non-standard NaN values during serialization.
        with self.assertRaises(TypeError):
            TimeSeriesCarbonIntensityProvider([sample(0, 100)], metadata=[])
        with self.assertRaises(TypeError):
            TimeSeriesCarbonIntensityProvider([sample(0, 100)], metadata={1: "bad"})
        with self.assertRaises(TypeError):
            TimeSeriesCarbonIntensityProvider(
                [sample(0, 100)],
                metadata={"bad": float("nan")},
            )
        with self.assertRaises(TypeError):
            TimeSeriesCarbonIntensityProvider(
                [sample(0, 100)],
                metadata={"nested": {1: "bad"}},
            )

    def test_actual_data_cannot_masquerade_as_a_forecast(self) -> None:
        """An actual-only provider has an explicit unavailable forecast path."""

        with self.assertRaises(ForecastUnavailableError):
            self.provider.get_forecast(BASE, timedelta(hours=24))
        with self.assertRaises(TypeError):
            CarbonIntensityForecast(BASE, ("not a sample",))  # type: ignore[arg-type]

    def test_cache_round_trip_preserves_values_and_provenance(self) -> None:
        """The local cache can reproduce an offline provider exactly."""

        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "carbon_intensity.json"
            source = TimeSeriesCarbonIntensityProvider(
                [
                    CarbonIntensitySample(
                        BASE,
                        100,
                        is_estimated=True,
                        estimation_method="MODEL",
                    )
                ],
                metadata={"zone": "synthetic"},
            )
            source.save(cache_path)
            restored = TimeSeriesCarbonIntensityProvider.load(cache_path)

        self.assertEqual(restored.samples, source.samples)
        self.assertEqual(restored.metadata["zone"], "synthetic")
        self.assertEqual(restored.granularity, timedelta(minutes=15))

    def test_corrupt_cache_is_rejected(self) -> None:
        """Schema, granularity, and required sample fields are revalidated."""

        valid_sample = {
            "timestamp": "2020-05-06T07:00:00Z",
            "intensity_gco2e_per_kwh": 100,
        }
        invalid_payloads = (
            {
                "schema_version": True,
                "granularity_seconds": 900,
                "samples": [valid_sample],
            },
            {
                "schema_version": 1,
                "granularity_seconds": "900",
                "samples": [valid_sample],
            },
            {
                "schema_version": 1,
                "granularity_seconds": 900,
                "samples": [
                    {
                        "timestamp": "2020-05-06T07:00:00",
                        "intensity_gco2e_per_kwh": 100,
                    }
                ],
            },
            {
                "schema_version": 1,
                "granularity_seconds": 900,
                "samples": [{"timestamp": "2020-05-06T07:00:00Z"}],
            },
        )

        with TemporaryDirectory() as directory:
            cache_path = Path(directory) / "invalid.json"
            for payload in invalid_payloads:
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    cache_path.write_text(json.dumps(payload), encoding="utf-8")
                    TimeSeriesCarbonIntensityProvider.load(cache_path)

    def test_electricity_maps_request_and_response_are_normalized(self) -> None:
        """The client sends explicit semantics and parses API provenance flags."""

        observed_requests: list[tuple[str, dict[str, str]]] = []

        def fake_transport(url: str, headers: dict[str, str]) -> dict[str, object]:
            observed_requests.append((url, headers))
            return complete_api_payload(*request_interval(url))

        downloaded = ElectricityMapsClient(
            "test-token",
            transport=fake_transport,
        ).fetch_actual_range(
            BASE + timedelta(minutes=2),
            BASE + timedelta(minutes=47),
            chunk_size=timedelta(minutes=30),
        )

        # The requested interval is expanded to complete covering buckets.
        self.assertEqual(len(downloaded.samples), 4)
        self.assertEqual(downloaded.get_actual(BASE + timedelta(minutes=46)), 250)
        self.assertEqual(len(observed_requests), 2)
        first_url, first_headers = observed_requests[0]
        first_query = parse_qs(urlsplit(first_url).query)
        self.assertEqual(first_headers, {"auth-token": "test-token"})
        self.assertEqual(first_query["zone"], ["IT-NO"])
        self.assertEqual(first_query["temporalGranularity"], ["15_minutes"])
        self.assertEqual(first_query["emissionFactorType"], ["lifecycle"])
        self.assertEqual(first_query["flowTraced"], ["true"])
        self.assertEqual(first_query["disableEstimations"], ["false"])
        self.assertEqual(downloaded.metadata["gap_policy"], "error")

        # Chunk edges must stay on the same 15-minute grid as the samples.
        with self.assertRaises(ValueError):
            ElectricityMapsClient("test-token", transport=fake_transport).fetch_actual_range(
                BASE,
                BASE + timedelta(hours=1),
                chunk_size=timedelta(minutes=20),
            )

    def test_default_chunking_merges_the_full_inclusive_calendar_range(self) -> None:
        """Inclusive dates are downloaded as contiguous two-day API intervals."""

        start = datetime(2020, 4, 30, tzinfo=UTC)
        end_exclusive = datetime(2020, 11, 2, tzinfo=UTC)
        observed_intervals: list[tuple[datetime, datetime]] = []

        def fake_transport(url: str, headers: dict[str, str]) -> dict[str, object]:
            self.assertEqual(headers, {"auth-token": "test-token"})
            interval = request_interval(url)
            observed_intervals.append(interval)
            return complete_api_payload(*interval)

        downloaded = ElectricityMapsClient(
            "test-token",
            transport=fake_transport,
        ).fetch_actual_range(start, end_exclusive)

        two_days = timedelta(days=2)
        expected_intervals = [
            (start + index * two_days, start + (index + 1) * two_days)
            for index in range(93)
        ]
        self.assertEqual(observed_intervals, expected_intervals)
        self.assertTrue(
            all(end - start <= two_days for start, end in observed_intervals)
        )

        timestamps = [point.timestamp for point in downloaded.samples]
        self.assertEqual(len(timestamps), 17_856)
        self.assertEqual(len(set(timestamps)), 17_856)
        self.assertEqual(timestamps, sorted(timestamps))
        self.assertTrue(
            all(
                later - earlier == FIFTEEN_MINUTES
                for earlier, later in zip(timestamps, timestamps[1:])
            )
        )
        self.assertEqual(downloaded.coverage_start, start)
        self.assertEqual(downloaded.coverage_end, end_exclusive)
        self.assertEqual(downloaded.metadata["request_count"], 93)
        self.assertEqual(
            downloaded.metadata["request_chunk_seconds"],
            int(two_days.total_seconds()),
        )

        with self.assertRaises(ValueError):
            ElectricityMapsClient(
                "test-token",
                transport=fake_transport,
            ).fetch_actual_range(
                start,
                end_exclusive,
                chunk_size=two_days + FIFTEEN_MINUTES,
            )
        self.assertEqual(len(observed_intervals), 93)

    def test_electricity_maps_rejects_forbidden_estimates_and_gaps(self) -> None:
        """Exclusion and completeness settings are enforced on the response."""

        def estimated_transport(
            url: str,
            headers: dict[str, str],
        ) -> dict[str, object]:
            del headers
            query = parse_qs(urlsplit(url).query)
            self.assertEqual(query["disableEstimations"], ["true"])
            self.assertEqual(query["emissionFactorType"], ["direct"])
            self.assertEqual(query["flowTraced"], ["false"])
            return {
                "zone": "IT-NO",
                "temporalGranularity": "15_minutes",
                "emissionFactorType": "direct",
                "data": [
                    {
                        "datetime": BASE.isoformat().replace("+00:00", "Z"),
                        "carbonIntensity": 250,
                        "emissionFactorType": "direct",
                        "isEstimated": True,
                    }
                ],
            }

        with self.assertRaises(ElectricityMapsError):
            ElectricityMapsClient(
                "test-token",
                transport=estimated_transport,
            ).fetch_actual_range(
                BASE,
                BASE + timedelta(minutes=15),
                emission_factor_type="direct",
                flow_traced=False,
                include_estimated=False,
            )

        def empty_transport(
            url: str,
            headers: dict[str, str],
        ) -> dict[str, object]:
            del url, headers
            return {
                "zone": "IT-NO",
                "temporalGranularity": "15_minutes",
                "data": [],
            }

        with self.assertRaises(ElectricityMapsError):
            ElectricityMapsClient(
                "test-token",
                transport=empty_transport,
            ).fetch_actual_range(BASE, BASE + timedelta(minutes=15))

        def gapped_transport(
            url: str,
            headers: dict[str, str],
        ) -> dict[str, object]:
            del url, headers
            return {
                "zone": "IT-NO",
                "temporalGranularity": "15_minutes",
                "emissionFactorType": "lifecycle",
                "data": [
                    {
                        "datetime": BASE.isoformat().replace("+00:00", "Z"),
                        "carbonIntensity": 100,
                        "emissionFactorType": "lifecycle",
                        "isEstimated": False,
                    },
                    {
                        "datetime": (BASE + timedelta(minutes=30))
                        .isoformat()
                        .replace("+00:00", "Z"),
                        "carbonIntensity": 200,
                        "emissionFactorType": "lifecycle",
                        "isEstimated": False,
                    },
                ],
            }

        with self.assertRaises(ElectricityMapsError):
            ElectricityMapsClient(
                "test-token",
                transport=gapped_transport,
            ).fetch_actual_range(BASE, BASE + timedelta(minutes=45))

        def mismatched_factor_transport(
            url: str,
            headers: dict[str, str],
        ) -> dict[str, object]:
            del url, headers
            return {
                "zone": "IT-NO",
                "temporalGranularity": "15_minutes",
                "emissionFactorType": "direct",
                "data": [],
            }

        with self.assertRaises(ElectricityMapsError):
            ElectricityMapsClient(
                "test-token",
                transport=mismatched_factor_transport,
            ).fetch_actual_range(BASE, BASE + timedelta(minutes=15))

    def test_default_cache_filename_tracks_the_requested_zone(self) -> None:
        """Changing --zone cannot silently write under an IT-NO filename."""

        self.assertEqual(
            default_output_path("IT-NO").name,
            "electricity_maps_it_no_15min.json",
        )
        self.assertEqual(
            default_output_path("DE").name,
            "electricity_maps_de_15min.json",
        )

    def test_provider_integrates_with_carbon_accounting(self) -> None:
        """The existing accounting API can consume get_actual directly."""

        job = JobPowerProfile(
            job_id="provider-integration",
            duration_seconds=30 * 60,
            average_power_watts=1_000,
        )
        result = account_emissions(job, BASE, self.provider.get_actual)

        # 0.25 kWh at 100 plus 0.25 kWh at 300 equals 100 gCO2e.
        self.assertAlmostEqual(result.energy_kwh, 0.5)
        self.assertAlmostEqual(result.emissions_gco2, 100.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
