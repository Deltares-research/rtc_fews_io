from __future__ import annotations
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from math import isnan
from pathlib import Path
import fewsxml as fx
import numpy as np


class TimeStepError(ValueError):
    """Raised when PI time step metadata cannot be normalized."""


@dataclass(frozen=True, order=True)
class PiSeriesKey:
    """FEWS PI identifier for a time series.
    The `id` property intentionally mirrors RTC-Tools' fallback identifier style:
    `locationId:parameterId[:qualifierId...]`. Qualifiers are sorted for stable
    lookup keys, while `qualifier_ids` preserves the order read from or supplied
    to the XML model for writing.
    """

    location_id: str
    parameter_id: str
    qualifier_ids: tuple[str, ...] = ()

    @property
    def id(self) -> str:
        parts = [self.location_id, self.parameter_id, *sorted(self.qualifier_ids)]
        return ":".join(parts)

    @classmethod
    def from_header(cls, header: fx.PIHeader) -> PiSeriesKey:
        return cls(
            location_id=header.locationId,
            parameter_id=header.parameterId,
            qualifier_ids=tuple(header.qualifierId or ()),
        )


@dataclass
class FewsTimeSeries:
    """Normalized XML-only FEWS PI TimeSeries data.
    `values` and `units` are indexed first by ensemble member and then by the
    normalized series identifier (`PiSeriesKey.id`). Missing values are stored as
    `numpy.nan`.
    """

    times: list[datetime]
    values: dict[int, dict[str, np.ndarray]] = field(default_factory=dict)
    units: dict[int, dict[str, str]] = field(default_factory=dict)
    series_keys: dict[str, PiSeriesKey] = field(default_factory=dict)
    timezone: float | None = None
    forecast_datetime: datetime | None = None
    dt: timedelta | None = None
    contains_ensemble: bool = False
    ensemble_size: int = 1
    version: str | None = None
    miss_value: float = -999.0

    @classmethod
    def read(cls, path: str | Path) -> FewsTimeSeries:
        """Read a PI TimeSeries XML file and normalize it."""
        return cls.from_pi_timeseries(fx.read(str(path)))

    @classmethod
    def from_file(cls, path: str | Path) -> FewsTimeSeries:
        """Alias for `read`."""
        return cls.read(path)

    @classmethod
    def from_pi_timeseries(cls, pi: fx.PITimeSeries) -> FewsTimeSeries:
        """Normalize a `fewsxml.PITimeSeries` model."""
        if not pi.series:
            raise ValueError("PI TimeSeries contains no series.")
        series_infos = [_SeriesInfo.from_series(series) for series in pi.series]
        dt = _common_time_step(series_infos)
        start_datetime = min(info.start_datetime for info in series_infos)
        end_datetime = max(info.end_datetime for info in series_infos)
        times = _global_times(series_infos, start_datetime, end_datetime, dt)
        forecast_datetime = _forecast_datetime(series_infos)
        forecast_index = (
            _index_or_none(times, forecast_datetime) if forecast_datetime else None
        )
        ensemble_indexes = {
            info.ensemble_member for info in series_infos if info.has_ensemble
        }
        contains_ensemble, ensemble_size = _ensemble_metadata(ensemble_indexes)
        normalized = cls(
            times=times,
            timezone=pi.timeZone,
            forecast_datetime=forecast_datetime,
            dt=dt,
            contains_ensemble=contains_ensemble,
            ensemble_size=ensemble_size,
            version=pi.version,
        )
        normalized._forecast_index = forecast_index
        for ensemble_member in range(ensemble_size):
            normalized.values.setdefault(ensemble_member, {})
            normalized.units.setdefault(ensemble_member, {})
        time_to_index = {time: i for i, time in enumerate(times)}
        for info in series_infos:
            target_members: Iterable[int]
            if contains_ensemble:
                if info.has_ensemble:
                    target_members = (info.ensemble_member,)
                else:
                    target_members = range(ensemble_size)
            else:
                target_members = (0,)
            values = _values_on_global_axis(info, times, time_to_index)
            unit = info.unit or "unit_unknown"
            variable = info.key.id
            normalized.series_keys[variable] = info.key
            for target_member in target_members:
                if variable in normalized.values[target_member]:
                    raise ValueError(
                        "Duplicate PI TimeSeries for "
                        f"{variable!r} in ensemble member {target_member}."
                    )
                normalized.values[target_member][variable] = values.copy()
                normalized.units[target_member][variable] = unit
        return normalized

    @property
    def forecast_index(self) -> int | None:
        """Index of `forecast_datetime` in `times`, or `None` if outside the range."""
        return getattr(
            self, "_forecast_index", _index_or_none(self.times, self.forecast_datetime)
        )

    @property
    def start_datetime(self) -> datetime:
        return self.times[0]

    @property
    def end_datetime(self) -> datetime:
        return self.times[-1]

    def variable_ids(self, ensemble_member: int = 0) -> tuple[str, ...]:
        """Return normalized variable IDs for an ensemble member."""
        self._require_ensemble_member(ensemble_member)
        return tuple(self.values[ensemble_member].keys())

    def items(self, ensemble_member: int = 0) -> Iterator[tuple[str, np.ndarray]]:
        """Iterate over `(variable_id, values)` for an ensemble member."""
        self._require_ensemble_member(ensemble_member)
        yield from self.values[ensemble_member].items()

    def get(self, variable: str, ensemble_member: int = 0) -> np.ndarray:
        """Return values for a variable and ensemble member."""
        self._require_ensemble_member(ensemble_member)
        return self.values[ensemble_member][variable]

    def set(
        self,
        variable: str,
        values: Iterable[float],
        *,
        key: PiSeriesKey | None = None,
        unit: str | None = None,
        ensemble_member: int = 0,
    ) -> None:
        """Set or replace values for a variable.
        `values` must have the same length as `times`.
        """
        self._require_ensemble_member(ensemble_member, allow_create=True)
        array = np.asarray(list(values), dtype=float)
        if len(array) != len(self.times):
            raise ValueError(
                f"Length of values ({len(array)}) must match length of times ({len(self.times)})."
            )
        if key is not None:
            self.series_keys[variable] = key
        elif variable not in self.series_keys:
            self.series_keys[variable] = _key_from_variable_id(variable)
        self.values.setdefault(ensemble_member, {})[variable] = array
        self.units.setdefault(ensemble_member, {})[variable] = unit or self.get_unit(
            variable, ensemble_member
        )

    def get_unit(self, variable: str, ensemble_member: int = 0) -> str:
        """Return the unit for a variable, or `unit_unknown`."""
        self._require_ensemble_member(ensemble_member)
        return self.units.get(ensemble_member, {}).get(variable, "unit_unknown")

    def set_unit(self, variable: str, unit: str, ensemble_member: int = 0) -> None:
        """Set a variable unit."""
        self._require_ensemble_member(ensemble_member)
        self.units.setdefault(ensemble_member, {})[variable] = unit

    def to_pi_timeseries(self) -> fx.PITimeSeries:
        """Convert normalized data back to a `fewsxml.PITimeSeries` model."""
        series = []
        members = range(self.ensemble_size) if self.contains_ensemble else (0,)
        for ensemble_member in members:
            for variable in sorted(self.values.get(ensemble_member, {})):
                key = self.series_keys.get(variable) or _key_from_variable_id(variable)
                values = self.values[ensemble_member][variable]
                unit = self.get_unit(variable, ensemble_member)
                header_kwargs: dict[str, object] = {
                    "timeStep": _time_step_model(self.dt),
                    "missVal": str(self.miss_value),
                    "stationName": key.location_id,
                    "units": unit,
                }
                if key.qualifier_ids:
                    header_kwargs["qualifierId"] = list(key.qualifier_ids)
                if self.forecast_datetime is not None:
                    header_kwargs["forecastDate"] = _pi_datetime(self.forecast_datetime)
                if self.contains_ensemble:
                    header_kwargs["ensembleMemberIndex"] = ensemble_member
                header = fx.create_pi_header(
                    type="instantaneous",
                    location_id=key.location_id,
                    parameter_id=key.parameter_id,
                    start_date=self.start_datetime,
                    end_date=self.end_datetime,
                    **header_kwargs,
                )
                events = [
                    {
                        "date": time,
                        "value": (
                            self.miss_value
                            if _is_missing_value(value)
                            else float(value)
                        ),
                    }
                    for time, value in zip(self.times, values, strict=True)
                ]
                series.append(fx.create_pi_series(header=header, events=events))
        return fx.create_pi_timeseries(
            series=series,
            time_zone=self.timezone,
            version=self.version or "1.2",
        )

    def write(self, path: str | Path) -> None:
        """Write normalized data as PI TimeSeries XML."""
        fx.write(self.to_pi_timeseries(), str(path))

    def _require_ensemble_member(
        self, ensemble_member: int, *, allow_create: bool = False
    ) -> None:
        if ensemble_member < 0:
            raise KeyError(f"ensemble_member {ensemble_member} does not exist")
        if ensemble_member >= self.ensemble_size:
            if not allow_create:
                raise KeyError(f"ensemble_member {ensemble_member} does not exist")
            if ensemble_member > 0:
                self.contains_ensemble = True
            self.ensemble_size = ensemble_member + 1


@dataclass(frozen=True)
class _SeriesInfo:
    series: fx.PISeries
    key: PiSeriesKey
    start_datetime: datetime
    end_datetime: datetime
    forecast_datetime: datetime | None
    time_step: timedelta | None
    has_ensemble: bool
    ensemble_member: int
    miss_value: float
    unit: str | None

    @classmethod
    def from_series(cls, series: fx.PISeries) -> _SeriesInfo:
        header = series.header
        return cls(
            series=series,
            key=PiSeriesKey.from_header(header),
            start_datetime=_parse_pi_datetime(header.startDate),
            end_datetime=_parse_pi_datetime(header.endDate),
            forecast_datetime=(
                _parse_pi_datetime(header.forecastDate)
                if header.forecastDate is not None
                else None
            ),
            time_step=_parse_time_step(header.timeStep),
            has_ensemble=header.ensembleMemberIndex is not None,
            ensemble_member=(
                header.ensembleMemberIndex
                if header.ensembleMemberIndex is not None
                else 0
            ),
            miss_value=float(header.missVal) if header.missVal is not None else -999.0,
            unit=header.units,
        )


def _parse_pi_datetime(value: fx.PIDateTime) -> datetime:
    return datetime.strptime(value.date + " " + value.time, "%Y-%m-%d %H:%M:%S")


def _pi_datetime(value: datetime) -> fx.PIDateTime:
    return fx.PIDateTime(
        date=value.strftime("%Y-%m-%d"), time=value.strftime("%H:%M:%S")
    )


def _event_datetime(event: fx.PIEvent) -> datetime:
    if event.date is None or event.time is None:
        raise ValueError("Only PI events with date/time attributes are supported.")
    return datetime.strptime(event.date + " " + event.time, "%Y-%m-%d %H:%M:%S")


def _parse_time_step(time_step: fx.PITimeStep | None) -> timedelta | None:
    if time_step is None:
        raise TimeStepError("Missing PI timeStep element.")
    unit = time_step.unit
    if unit == "nonequidistant":
        return None
    multiplier = (
        time_step.multiplier if time_step.multiplier is not None else time_step.minutes
    )
    if multiplier is None:
        multiplier = 1
    multiplier = int(multiplier)
    if multiplier <= 0:
        raise TimeStepError("PI timeStep multiplier must be a positive integer.")
    if unit == "second":
        return timedelta(seconds=multiplier)
    if unit == "minute":
        return timedelta(minutes=multiplier)
    if unit == "hour":
        return timedelta(hours=multiplier)
    if unit == "day":
        return timedelta(days=multiplier)
    raise TimeStepError(f"Unsupported PI timeStep unit: {unit!r}.")


def _common_time_step(series_infos: list[_SeriesInfo]) -> timedelta | None:
    first = series_infos[0].time_step
    for info in series_infos[1:]:
        if info.time_step != first:
            raise TimeStepError("Not all PI time series have the same time step size.")
    return first


def _global_times(
    series_infos: list[_SeriesInfo],
    start_datetime: datetime,
    end_datetime: datetime,
    dt: timedelta | None,
) -> list[datetime]:
    if dt is None:
        times = {
            _event_datetime(event)
            for info in series_infos
            for event in info.series.event
        }
        return [
            time for time in sorted(times) if start_datetime <= time <= end_datetime
        ]
    n_steps = int(
        round((end_datetime - start_datetime).total_seconds() / dt.total_seconds())
    )
    return [start_datetime + i * dt for i in range(n_steps + 1)]


def _forecast_datetime(series_infos: list[_SeriesInfo]) -> datetime:
    explicit = [
        info.forecast_datetime
        for info in series_infos
        if info.forecast_datetime is not None
    ]
    if not explicit:
        return max(info.start_datetime for info in series_infos)
    first = explicit[0]
    if any(value != first for value in explicit[1:]):
        raise ValueError("Not all PI time series share the same forecastDate.")
    return first


def _ensemble_metadata(ensemble_indexes: set[int]) -> tuple[bool, int]:
    if len(ensemble_indexes) <= 1:
        return False, 1
    sorted_indexes = sorted(ensemble_indexes)
    expected = list(range(len(sorted_indexes)))
    if sorted_indexes != expected:
        raise ValueError("PI ensemble ids must be zero-based and increasing by 1.")
    return True, len(sorted_indexes)


def _values_on_global_axis(
    info: _SeriesInfo,
    times: list[datetime],
    time_to_index: Mapping[datetime, int],
) -> np.ndarray:
    values = np.full(len(times), np.nan, dtype=float)
    for event in info.series.event:
        event_time = _event_datetime(event)
        try:
            index = time_to_index[event_time]
        except KeyError as exc:
            raise ValueError(
                f"Event time {event_time} is outside the global time axis."
            ) from exc
        value = _event_value(event)
        if value == info.miss_value:
            value = np.nan
        values[index] = value
    return values


def _event_value(event: fx.PIEvent) -> float:
    if event.value is None:
        return np.nan
    try:
        return float(event.value)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"Unsupported non-numeric PI event value: {event.value!r}."
        ) from exc


def _index_or_none(times: list[datetime], value: datetime | None) -> int | None:
    if value is None:
        return None
    try:
        return times.index(value)
    except ValueError:
        return None


def _time_step_model(dt: timedelta | None) -> fx.PITimeStep:
    if dt is None:
        return fx.PITimeStep(unit="nonequidistant")
    seconds = dt.total_seconds()
    if not seconds.is_integer():
        raise TimeStepError("Only whole-second time steps can be written to PI XML.")
    return fx.PITimeStep(unit="second", multiplier=int(seconds))


def _key_from_variable_id(variable: str) -> PiSeriesKey:
    parts = variable.split(":")
    if len(parts) < 2:
        return PiSeriesKey(location_id=variable, parameter_id=variable)
    return PiSeriesKey(
        location_id=parts[0], parameter_id=parts[1], qualifier_ids=tuple(parts[2:])
    )


def _is_missing_value(value: float) -> bool:
    try:
        return isnan(float(value))
    except (TypeError, ValueError):
        return False
