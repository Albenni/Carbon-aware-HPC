from __future__ import annotations

from collections.abc import Callable, Iterator
from datetime import datetime, timedelta, timezone
from math import ceil, frexp, fsum, isclose, isfinite, ldexp
from typing import TypeAlias

from .models import AccountingResult, JobPowerProfile, PowerModel


WATT_SECONDS_PER_KILOWATT_HOUR = 3_600_000.0

CarbonIntensity: TypeAlias = float | Callable[[datetime], float]
PowerSegment: TypeAlias = tuple[float, float, float]


def _non_negative_finite(value: object, name: str) -> float:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be a number")

    try:
        converted = float(value)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{name} must be a real number") from error

    if not isfinite(converted):
        raise ValueError(f"{name} must be finite")
    if converted < 0.0:
        raise ValueError(f"{name} cannot be negative")
    return converted


def _aware_start_time(start_time: datetime) -> datetime:
    if not isinstance(start_time, datetime):
        raise TypeError("start_time must be a datetime")
    if start_time.tzinfo is None or start_time.utcoffset() is None:
        raise ValueError("start_time must include timezone information")
    return start_time


def _power_model(power_model: PowerModel | str) -> PowerModel:
    if isinstance(power_model, PowerModel):
        return power_model
    try:
        return PowerModel(power_model)
    except (TypeError, ValueError) as error:
        choices = ", ".join(model.value for model in PowerModel)
        raise ValueError(f"power_model must be one of: {choices}") from error


def _segment_energy_kwh(power_watts: float, duration_seconds: float) -> float:
    energy_kwh = _multiply_then_divide(
        power_watts,
        duration_seconds,
        WATT_SECONDS_PER_KILOWATT_HOUR,
    )
    if not isfinite(energy_kwh):
        raise OverflowError("calculated energy exceeds the supported numeric range")
    return energy_kwh


def _multiply_then_divide(
    first: float,
    second: float,
    divisor: float,
) -> float:
    """Evaluate ``first * second / divisor`` without avoidable range loss."""

    if first == 0.0 or second == 0.0:
        return 0.0

    direct_product = first * second
    if isfinite(direct_product) and direct_product != 0.0:
        direct_result = direct_product / divisor
        if isfinite(direct_result) and direct_result != 0.0:
            return direct_result

    first_mantissa, first_exponent = frexp(first)
    second_mantissa, second_exponent = frexp(second)
    divisor_mantissa, divisor_exponent = frexp(divisor)
    scaled_mantissa = (
        first_mantissa * second_mantissa / divisor_mantissa
    )
    try:
        return ldexp(
            scaled_mantissa,
            first_exponent + second_exponent - divisor_exponent,
        )
    except OverflowError as error:
        raise OverflowError(
            "calculated value exceeds the supported numeric range"
        ) from error


def energy_from_constant_power(
    power_watts: float,
    duration_seconds: float,
) -> float:
    """Convert constant power in W over seconds to energy in kWh."""

    power = _non_negative_finite(power_watts, "power_watts")
    duration = _non_negative_finite(duration_seconds, "duration_seconds")
    return _segment_energy_kwh(power, duration)


def _time_segments(
    duration_seconds: float,
    interval_seconds: float,
) -> Iterator[tuple[float, float]]:
    """Yield stable ``(offset, duration)`` pairs that exactly cover a runtime."""

    quotient = duration_seconds / interval_seconds
    nearest_integer = round(quotient)
    if isclose(quotient, nearest_integer, rel_tol=1e-12, abs_tol=1e-12):
        segment_count = max(1, int(nearest_integer))
    else:
        segment_count = max(1, ceil(quotient))

    for index in range(segment_count):
        offset_seconds = index * interval_seconds
        segment_end = (
            duration_seconds
            if index == segment_count - 1
            else (index + 1) * interval_seconds
        )
        yield offset_seconds, segment_end - offset_seconds


def _average_power_segments(job: JobPowerProfile) -> Iterator[PowerSegment]:
    for offset_seconds, segment_seconds in _time_segments(
        job.duration_seconds,
        job.sample_interval_seconds,
    ):
        yield offset_seconds, segment_seconds, job.average_power_watts


def _validate_measured_coverage(job: JobPowerProfile) -> tuple[float, ...]:
    profile = job.power_profile_watts
    if profile is None:
        raise ValueError(
            "power_model='measured' requires power_profile_watts"
        )

    nominal_coverage = len(profile) * job.sample_interval_seconds
    coverage_difference = abs(nominal_coverage - job.duration_seconds)
    numerical_tolerance = max(1e-9, job.sample_interval_seconds * 1e-12)
    if coverage_difference > job.sample_interval_seconds + numerical_tolerance:
        raise ValueError(
            "measured profile coverage differs from duration by more than one "
            f"sample interval: duration={job.duration_seconds:g}s, "
            f"coverage={nominal_coverage:g}s, "
            f"allowed={job.sample_interval_seconds:g}s"
        )
    return profile


def _measured_power_segments(job: JobPowerProfile) -> Iterator[PowerSegment]:
    profile = _validate_measured_coverage(job)
    for index, (offset_seconds, segment_seconds) in enumerate(
        _time_segments(job.duration_seconds, job.sample_interval_seconds)
    ):
        # A profile may contain one unused terminal sample, or end one sample
        # early. The former is naturally ignored; the latter holds the final
        # observed value over the one permitted missing segment.
        power_watts = profile[index] if index < len(profile) else profile[-1]
        yield offset_seconds, segment_seconds, power_watts


def _segments(
    job: JobPowerProfile,
    power_model: PowerModel,
) -> Iterator[PowerSegment]:
    if power_model is PowerModel.AVERAGE:
        return _average_power_segments(job)
    return _measured_power_segments(job)


def energy_from_measured_profile(job: JobPowerProfile) -> float:
    """Integrate a measured profile, returning whole-job energy in kWh.

    A terminal excess is trimmed. A missing tail no longer than one sampling
    interval is filled by holding the final measured value, matching the PM100
    cleaning contract.
    """

    try:
        return fsum(
            _segment_energy_kwh(power_watts, segment_seconds)
            for _, segment_seconds, power_watts in _measured_power_segments(job)
        )
    except OverflowError as error:
        raise OverflowError(
            "calculated measured-profile energy exceeds the supported numeric range"
        ) from error


def measured_average_power(job: JobPowerProfile) -> float:
    """Return the runtime-weighted mean of the normalized measured profile."""

    maximum_power = max(
        power_watts
        for _, _, power_watts in _measured_power_segments(job)
    )
    if maximum_power == 0.0:
        return 0.0

    normalized_average = fsum(
        (power_watts / maximum_power)
        * (segment_seconds / job.duration_seconds)
        for _, segment_seconds, power_watts in _measured_power_segments(job)
    )
    # The physical weighted mean cannot exceed the largest sample. Clamp the
    # last-bit overshoot that can arise when many floating-point weights sum.
    return maximum_power * min(normalized_average, 1.0)


def _intensity_at(
    carbon_intensity: Callable[[datetime], float],
    timestamp: datetime,
) -> float:
    return _non_negative_finite(
        carbon_intensity(timestamp),
        f"carbon intensity at {timestamp.isoformat()}",
    )


def _execution_end(
    start_time: datetime,
    duration_seconds: float,
) -> tuple[datetime, datetime]:
    """Return UTC start and local-zone end using elapsed, not wall-clock, time."""

    try:
        start_utc = start_time.astimezone(timezone.utc)
        end_utc = start_utc + timedelta(seconds=duration_seconds)
        end_time = end_utc.astimezone(start_time.tzinfo)
    except OverflowError as error:
        raise OverflowError(
            "job duration places end_time outside datetime's supported range"
        ) from error
    return start_utc, end_time


def _emissions_for_segment(energy_kwh: float, intensity: float) -> float:
    emissions_gco2 = energy_kwh * intensity
    if not isfinite(emissions_gco2):
        raise OverflowError("calculated emissions exceed the supported numeric range")
    return emissions_gco2


def account_emissions(
    job: JobPowerProfile,
    start_time: datetime,
    carbon_intensity: CarbonIntensity,
    *,
    power_model: PowerModel | str = PowerModel.AVERAGE,
) -> AccountingResult:
    """Calculate a job's energy and operational emissions.

    ``carbon_intensity`` is either a constant in gCO2/kWh or a callable that
    returns that unit for a timezone-aware timestamp. A callable is sampled at
    the start of every power segment and treated as constant over that segment.
    """

    execution_start = _aware_start_time(start_time)
    execution_start_utc, execution_end = _execution_end(
        execution_start,
        job.duration_seconds,
    )
    selected_model = _power_model(power_model)

    if selected_model is PowerModel.AVERAGE:
        energy_kwh = _segment_energy_kwh(
            job.average_power_watts,
            job.duration_seconds,
        )
    else:
        energy_kwh = energy_from_measured_profile(job)

    if callable(carbon_intensity):
        emissions_gco2 = fsum(
            _emissions_for_segment(
                _segment_energy_kwh(power_watts, segment_seconds),
                _intensity_at(
                    carbon_intensity,
                    (
                        execution_start_utc
                        + timedelta(seconds=offset_seconds)
                    ).astimezone(execution_start.tzinfo),
                ),
            )
            for offset_seconds, segment_seconds, power_watts in _segments(
                job,
                selected_model,
            )
        )
    else:
        constant_intensity = _non_negative_finite(
            carbon_intensity,
            "carbon_intensity",
        )
        emissions_gco2 = _emissions_for_segment(energy_kwh, constant_intensity)

    return AccountingResult(
        job_id=job.job_id,
        power_model=selected_model,
        start_time=execution_start,
        end_time=execution_end,
        energy_kwh=energy_kwh,
        emissions_gco2=emissions_gco2,
    )


def carbon_emissions(
    job: JobPowerProfile,
    start_time: datetime,
    carbon_intensity: CarbonIntensity,
    *,
    power_model: PowerModel | str = PowerModel.AVERAGE,
) -> float:
    """Return only the job emissions in grams of CO2.

    Use :func:`account_emissions` when both energy and emissions are needed.
    """

    return account_emissions(
        job,
        start_time,
        carbon_intensity,
        power_model=power_model,
    ).emissions_gco2
