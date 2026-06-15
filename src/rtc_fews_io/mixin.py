from __future__ import annotations

import logging
import bisect
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import numpy as np

from .mapping import DataConfig, FewsSeriesId
from .parameters import ParameterConfig
from .timeseries import FewsTimeSeries, PiSeriesKey

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimeseriesValues:
    """Small fallback timeseries container used when RTC-Tools is not importable."""

    times: np.ndarray
    values: np.ndarray


class FewsIOMixin:
    """FEWS PI XML I/O mixin for RTC-Tools-style optimization and simulation problems.

    The mixin intentionally uses fresh code and the lightweight helpers in this package. It
    expects the host problem to expose RTC-Tools' usual ``io`` datastore and folder attributes.
    Small behavior differences are selected with ``fews_io_mode``:

    - ``"optimization"``: read all ensemble members and broadcast parameters to each member.
    - ``"simulation"``: read one ensemble member selected by ``pi_ensemble_member``.
    - ``"auto"``: infer the mode from common RTC-Tools simulation attributes.

    Binary PI files are not supported by design.
    """

    fews_io_mode = "auto"

    timeseries_import_basename = "timeseries_import"
    timeseries_export_basename = "timeseries_export"

    pi_binary_timeseries = False
    pi_parameter_config_basenames = ["rtcParameterConfig"]
    pi_parameter_config_numerical_basename = "rtcParameterConfig_Numerical"
    pi_check_for_duplicate_parameters = True
    pi_validate_timeseries = True
    pi_ensemble_member = 0

    def __init__(self, **kwargs: Any):
        super().__init__(**kwargs)
        self._input_folder = getattr(
            self, "_input_folder", kwargs.get("input_folder", "input")
        )
        self._output_folder = getattr(
            self, "_output_folder", kwargs.get("output_folder", "output")
        )
        self.__data_config = DataConfig(self._input_folder)
        self.__parameter_config: list[ParameterConfig] = []
        self.__parameter_config_numerical: ParameterConfig | None = None
        self.__timeseries_import: FewsTimeSeries | None = None
        self.__timeseries_export: FewsTimeSeries | None = None

    def pre(self) -> None:
        _call_super_if_present(super(), "pre")
        self.read()

    def post(self) -> None:
        _call_super_if_present(super(), "post")
        self.write()

    def read(self) -> None:
        """Read FEWS input files and populate the RTC-Tools-style datastore."""
        if self.pi_binary_timeseries:
            raise NotImplementedError(
                "FewsIOMixin intentionally supports XML PI TimeSeries only."
            )

        _call_super_if_present(super(), "read")

        self.__parameter_config = [
            ParameterConfig(self._input_folder, basename)
            for basename in self.pi_parameter_config_basenames
        ]
        try:
            self.__parameter_config_numerical = ParameterConfig(
                self._input_folder, self.pi_parameter_config_numerical_basename
            )
        except FileNotFoundError:
            self.__parameter_config_numerical = None

        import_path = (
            Path(self._input_folder) / f"{self.timeseries_import_basename}.xml"
        )
        if not import_path.exists():
            raise FileNotFoundError(
                f"FewsIOMixin: {import_path.name} not found in {self._input_folder}."
            )

        raw_timeseries = FewsTimeSeries.read(import_path)
        self.__timeseries_import = _map_import_timeseries(
            raw_timeseries, self.__data_config
        )
        self.__timeseries_export = _new_output_timeseries(self.__timeseries_import)

        if self.pi_validate_timeseries:
            _validate_times(
                self.__timeseries_import, require_equidistant=self._is_simulation_mode()
            )

        self.io.reference_datetime = self.__timeseries_import.forecast_datetime

        if self._is_simulation_mode():
            self._read_simulation_inputs()
        else:
            self._read_optimization_inputs()

    def solver_options(self) -> dict[str, Any]:
        """Return solver options extended with rtcParameterConfig_Numerical values."""
        options = _call_super_if_present(super(), "solver_options", default={})
        if options is None:
            options = {}
        else:
            options = dict(options)

        if self.__parameter_config_numerical is not None:
            for (
                _location_id,
                _model_id,
                option,
                value,
            ) in self.__parameter_config_numerical:
                options[option] = value
        return options

    def times(self, variable: str | None = None) -> np.ndarray:
        """Return imported time stamps in seconds from forecast time onward."""
        del variable
        seconds = np.asarray(self.io.times_sec, dtype=float)
        start = bisect.bisect_left(seconds, 0.0)
        return seconds[start:]

    def parameters(self, ensemble_member: int = 0) -> dict[str, Any]:
        """Return model parameters merged with parameters read from FEWS files."""
        try:
            parameters = _call_super_if_present(
                super(), "parameters", ensemble_member, default={}
            )
        except TypeError:
            parameters = _call_super_if_present(super(), "parameters", default={})
        if parameters is None:
            parameters = {}
        else:
            parameters = dict(parameters)
        parameters.update(dict(self.io.parameters(ensemble_member)))
        return parameters

    def constant_inputs(self, ensemble_member: int) -> dict[str, Any]:
        """Return constant-input time series augmented with imported FEWS values."""
        constant_inputs = (
            _call_super_if_present(
                super(), "constant_inputs", ensemble_member, default={}
            )
            or {}
        )
        constant_inputs = dict(constant_inputs)
        start = bisect.bisect_left(
            self.io.times_sec, getattr(self, "initial_time", 0.0)
        )
        for symbol in getattr(self, "dae_variables", {}).get("constant_inputs", ()):
            variable = symbol.name() if hasattr(symbol, "name") else str(symbol)
            try:
                times, values = self.io.get_timeseries_sec(variable, ensemble_member)
            except KeyError:
                continue
            selected = np.asarray(values[start:], dtype=float)
            if np.any(np.isnan(selected)):
                raise ValueError(
                    f"FewsIOMixin: constant input {variable!r} contains NaN."
                )
            constant_inputs[variable] = _timeseries_container(times[start:], selected)
        return constant_inputs

    def history(self, ensemble_member: int) -> dict[str, Any]:
        """Return history time series up to and including the initial time."""
        history = (
            _call_super_if_present(super(), "history", ensemble_member, default={})
            or {}
        )
        history = dict(history)
        end = (
            bisect.bisect_left(self.io.times_sec, getattr(self, "initial_time", 0.0))
            + 1
        )
        for variable in _dae_variable_names(
            self,
            "states",
            "algebraics",
            "control_inputs",
            "constant_inputs",
        ):
            try:
                times, values = self.io.get_timeseries_sec(variable, ensemble_member)
            except KeyError:
                continue
            history[variable] = _timeseries_container(times[:end], values[:end])
        return history

    def seed(self, ensemble_member: int) -> dict[str, Any]:
        """Return seed values augmented with imported free-variable time series."""
        seed = (
            _call_super_if_present(super(), "seed", ensemble_member, default={}) or {}
        )
        seed = dict(seed)
        for variable in _dae_variable_names(self, "free_variables"):
            try:
                times, values = self.io.get_timeseries_sec(variable, ensemble_member)
            except KeyError:
                continue
            seed_values = np.asarray(values, dtype=float).copy()
            seed_values[np.isnan(seed_values)] = 0.0
            seed[variable] = _timeseries_container(times, seed_values)
        return seed

    def bounds(self, ensemble_member: int | None = None) -> dict[str, Any]:
        """Return bounds augmented with ``<variable>_Min``/``<variable>_Max`` FEWS series."""
        try:
            bounds = _call_super_if_present(
                super(), "bounds", ensemble_member, default={}
            )
        except TypeError:
            bounds = _call_super_if_present(super(), "bounds", default={})
        bounds = dict(bounds or {})
        member = 0 if ensemble_member is None else ensemble_member
        start = bisect.bisect_left(
            self.io.times_sec, getattr(self, "initial_time", 0.0)
        )
        for variable in _dae_variable_names(self, "free_variables"):
            lower = _bound_series(self, f"{variable}_Min", member, start, lower=True)
            upper = _bound_series(self, f"{variable}_Max", member, start, lower=False)
            if lower is not None or upper is not None:
                bounds[variable] = (lower, upper)
        return bounds

    def initialize(self, config_file: str | None = None) -> None:
        """Simulation initialization helper compatible with RTC-Tools-style models."""
        if not self._is_simulation_mode():
            _call_super_if_present(super(), "initialize", config_file)
            return

        times = np.asarray(self.io.times_sec, dtype=float)
        dt = float(times[1] - times[0]) if len(times) > 1 else 0.0
        self.__simulation_dt = dt
        self.setup_experiment(0.0, float(times[-1]), dt)

        parameter_variables = set(self.get_parameter_variables())
        for parameter, value in self.io.parameters().items():
            if parameter in parameter_variables:
                self.set_var(parameter, value)

        self.__simulation_input_variables = set(self.get_input_variables().keys())
        self.__set_simulation_inputs(bisect.bisect_left(times, 0.0))
        self._simulation_times = [self.get_current_time()]
        self._io_output_variables = list(self.get_output_variables())
        self._io_output = _alias_dict(self)

        _call_super_if_present(super(), "initialize", config_file)

        for variable in self._io_output_variables:
            self._io_output[variable] = [self.get_var(variable)]

    def update(self, dt: float) -> None:
        """Simulation update helper that advances imported inputs and stores outputs."""
        if not self._is_simulation_mode():
            _call_super_if_present(super(), "update", dt)
            return
        if dt < 0:
            dt = self.__simulation_dt
        current_time = self.get_current_time()
        next_time = current_time + dt
        self._simulation_times.append(next_time)
        self.__set_simulation_inputs(bisect.bisect_left(self.io.times_sec, next_time))
        _call_super_if_present(super(), "update", dt)
        for variable, values in self._io_output.items():
            values.append(self.get_var(variable))

    def extract_results(self, *args: Any, **kwargs: Any) -> Any:
        """Return simulation outputs, or delegate to the host optimization problem."""
        if not self._is_simulation_mode() or not hasattr(self, "_io_output"):
            return _call_super_if_present(super(), "extract_results", *args, **kwargs)
        return {
            variable: np.asarray(values) for variable, values in self._io_output.items()
        }

    def timeseries_at(self, variable: str, t: float, ensemble_member: int = 0) -> float:
        """Interpolate an imported time series at seconds ``t``."""
        times, values = self.io.get_timeseries_sec(variable, ensemble_member)
        return float(np.interp(t, times, values))

    def write(self) -> None:
        """Write mapped output series to ``timeseries_export.xml``."""
        _call_super_if_present(super(), "write")
        if self.__timeseries_import is None:
            raise RuntimeError("FewsIOMixin.write() called before read().")

        output = (
            self._collect_simulation_output()
            if self._is_simulation_mode()
            else self._collect_optimization_output()
        )
        output.write(
            Path(self._output_folder) / f"{self.timeseries_export_basename}.xml"
        )
        self.__timeseries_export = output

    def set_timeseries(
        self,
        variable: str,
        values: Any,
        *args: Any,
        unit: str | None = None,
        output: bool = True,
        check_consistency: bool = True,
        ensemble_member: int = 0,
        **kwargs: Any,
    ) -> None:
        """Set a time series and optionally include it in the FEWS export file.

        If a superclass provides ``set_timeseries`` it is used first, preserving RTC-Tools
        behavior. The local FEWS buffers are then updated so that units and export values stay in
        sync for simulation-like use cases.
        """
        if unit is not None and self.__timeseries_import is not None:
            self.__timeseries_import.set_unit(
                variable,
                unit,
                min(ensemble_member, self.__timeseries_import.ensemble_size - 1),
            )

        if self._is_simulation_mode():
            called_super = _call_super_if_present(
                super(),
                "set_timeseries",
                variable,
                values,
                *args,
                output=output,
                check_consistency=check_consistency,
                unit=unit,
                default=_NO_SUPER,
                **kwargs,
            )
        else:
            called_super = _call_super_if_present(
                super(),
                "set_timeseries",
                variable,
                values,
                *args,
                ensemble_member=ensemble_member,
                output=output,
                check_consistency=check_consistency,
                default=_NO_SUPER,
                **kwargs,
            )
        if called_super is not _NO_SUPER:
            return

        if self.__timeseries_import is None:
            raise RuntimeError("set_timeseries() can only be used after read().")

        array = _values_from_argument(values)
        if check_consistency and len(array) != len(self.times()):
            raise ValueError(
                (
                    f"FewsIOMixin: values for {variable!r} have length {len(array)}, "
                    f"expected {len(self.times())}."
                )
            )
        datetimes = self.__timeseries_import.times
        padded = _pad_forecast_values(
            datetimes, self.__timeseries_import.forecast_datetime, array
        )
        self.__timeseries_import.set(
            variable, padded, unit=unit, ensemble_member=ensemble_member
        )
        self.io.set_timeseries(variable, datetimes, padded, ensemble_member)

        if output and self.__timeseries_export is not None:
            try:
                key = _pi_key_for_variable(self.__data_config, variable)
            except KeyError:
                logger.debug(
                    "FewsIOMixin: variable %s has no rtcDataConfig mapping; not exporting.",
                    variable,
                )
            else:
                self.__timeseries_export.set(
                    variable,
                    padded,
                    key=key,
                    unit=unit
                    or self.__timeseries_import.get_unit(variable, ensemble_member),
                    ensemble_member=ensemble_member,
                )

    def get_timeseries(self, variable: str, ensemble_member: int = 0) -> Any:
        """Return a RTC-Tools Timeseries object when available, otherwise a small fallback."""
        try:
            super_timeseries = _call_super_if_present(
                super(), "get_timeseries", variable, ensemble_member, default=_NO_SUPER
            )
        except TypeError:
            super_timeseries = _NO_SUPER
        if super_timeseries is not _NO_SUPER:
            return super_timeseries
        times, values = self.io.get_timeseries_sec(variable, ensemble_member)
        cls = _rtctools_timeseries_class()
        return (
            cls(times, values) if cls is not None else TimeseriesValues(times, values)
        )

    @property
    def timeseries_import(self) -> FewsTimeSeries:
        if self.__timeseries_import is None:
            raise RuntimeError("timeseries_import is available after read().")
        return self.__timeseries_import

    @property
    def timeseries_export(self) -> FewsTimeSeries:
        if self.__timeseries_export is None:
            raise RuntimeError("timeseries_export is available after read().")
        return self.__timeseries_export

    @property
    def timeseries_import_times(self) -> np.ndarray:
        return self.io.times_sec

    @property
    def ensemble_size(self) -> int:
        return int(
            getattr(self.io, "ensemble_size", self.timeseries_import.ensemble_size)
        )

    def set_unit(self, variable: str, unit: str) -> None:
        """Set a unit on import and export buffers."""
        self.timeseries_import.set_unit(variable, unit, 0)
        self.timeseries_export.set_unit(variable, unit, 0)

    def _read_optimization_inputs(self) -> None:
        assert self.__timeseries_import is not None
        for ensemble_member in range(self.__timeseries_import.ensemble_size):
            for variable, values in self.__timeseries_import.items(ensemble_member):
                self.io.set_timeseries(
                    variable, self.__timeseries_import.times, values, ensemble_member
                )
            for parameter_config in self.__parameter_config:
                for location_id, model_id, parameter_id, value in parameter_config:
                    parameter = _map_parameter(
                        self.__data_config, parameter_id, location_id, model_id
                    )
                    self.io.set_parameter(
                        parameter,
                        value,
                        ensemble_member,
                        check_duplicates=self.pi_check_for_duplicate_parameters,
                    )

    def _read_simulation_inputs(self) -> None:
        assert self.__timeseries_import is not None
        member = (
            self.pi_ensemble_member if self.__timeseries_import.contains_ensemble else 0
        )
        for parameter_config in self.__parameter_config:
            for location_id, model_id, parameter_id, value in parameter_config:
                self.io.set_parameter(
                    _map_parameter(
                        self.__data_config, parameter_id, location_id, model_id
                    ),
                    value,
                )
        for variable, values in self.__timeseries_import.items(member):
            self.io.set_timeseries(variable, self.__timeseries_import.times, values)

    def _collect_optimization_output(self) -> FewsTimeSeries:
        assert self.__timeseries_import is not None
        seconds = np.asarray(self.times(), dtype=float)
        output = _new_output_timeseries(
            self.__timeseries_import,
            times=[
                self.__timeseries_import.forecast_datetime + timedelta(seconds=float(s))
                for s in seconds
            ],
            dt=_seconds_step(seconds),
            ensemble_size=self.ensemble_size,
            contains_ensemble=self.ensemble_size > 1,
        )
        for ensemble_member in range(self.ensemble_size):
            results = self.extract_results(ensemble_member)
            for variable in _output_variable_names(
                getattr(self, "output_variables", ())
            ):
                for alias in _aliases(self, variable):
                    values = _result_values(results, alias)
                    if values is None:
                        values = _stored_values(self, alias, ensemble_member)
                    if values is None:
                        logger.error(
                            "FewsIOMixin: output requested for missing alias %s.", alias
                        )
                        continue
                    self._add_output_series(output, alias, values, ensemble_member)
        return output

    def _collect_simulation_output(self) -> FewsTimeSeries:
        assert self.__timeseries_import is not None
        seconds = np.asarray(
            getattr(self, "_simulation_times", self.times()), dtype=float
        )
        output = _new_output_timeseries(
            self.__timeseries_import,
            times=[
                self.io.reference_datetime + timedelta(seconds=float(s))
                for s in seconds
            ],
            dt=_seconds_step(seconds),
            ensemble_size=1,
            contains_ensemble=self.__timeseries_import.contains_ensemble,
        )
        for variable in getattr(self, "_io_output_variables", ()):
            values = np.asarray(getattr(self, "_io_output", {})[variable], dtype=float)
            self._add_output_series(output, variable, values, 0)
        return output

    def _add_output_series(
        self, output: FewsTimeSeries, variable: str, values: Any, ensemble_member: int
    ) -> None:
        assert self.__timeseries_import is not None
        try:
            key = _pi_key_for_variable(self.__data_config, variable)
        except KeyError:
            logger.debug(
                "FewsIOMixin: variable %s has no rtcDataConfig mapping; not exporting.",
                variable,
            )
            return

        array = np.asarray(values, dtype=float)
        if len(array) != len(output.times):
            array = _interpolate_if_possible(self, variable, output.times, array)
        output.set(
            variable,
            array,
            key=key,
            unit=self.__timeseries_import.get_unit(variable, 0),
            ensemble_member=ensemble_member,
        )

    def _is_simulation_mode(self) -> bool:
        if self.fews_io_mode == "simulation":
            return True
        if self.fews_io_mode == "optimization":
            return False
        modules = {cls.__module__ for cls in type(self).mro()}
        if any("rtctools.simulation" in module for module in modules):
            return True
        if any("rtctools.optimization" in module for module in modules):
            return False
        return hasattr(self, "_simulation_times") or hasattr(
            self, "_io_output_variables"
        )

    def __set_simulation_inputs(self, index: int) -> None:
        available = set(_timeseries_names(self.io))
        for variable in getattr(self, "_FewsIOMixin__simulation_input_variables", ()):
            if variable not in available:
                continue
            _times, values = self.io.get_timeseries_sec(variable)
            value = values[index]
            if np.isfinite(value):
                self.set_var(variable, value)


_NO_SUPER = object()


def _call_super_if_present(
    super_proxy: Any, name: str, *args: Any, default: Any = None, **kwargs: Any
) -> Any:
    method = getattr(super_proxy, name, None)
    if method is None:
        return default
    try:
        return method(*args, **kwargs)
    except NotImplementedError:
        return default


def _map_import_timeseries(
    timeseries: FewsTimeSeries, data_config: DataConfig
) -> FewsTimeSeries:
    mapped = FewsTimeSeries(
        times=list(timeseries.times),
        timezone=timeseries.timezone,
        forecast_datetime=timeseries.forecast_datetime,
        dt=timeseries.dt,
        contains_ensemble=timeseries.contains_ensemble,
        ensemble_size=timeseries.ensemble_size,
        version=timeseries.version,
        miss_value=timeseries.miss_value,
    )
    for ensemble_member, series in timeseries.values.items():
        for fallback_id, values in series.items():
            key = timeseries.series_keys[fallback_id]
            variable = data_config.variable(key)
            mapped.set(
                variable,
                values,
                key=key,
                unit=timeseries.get_unit(fallback_id, ensemble_member),
                ensemble_member=ensemble_member,
            )
    mapped._forecast_index = timeseries.forecast_index
    return mapped


def _new_output_timeseries(
    source: FewsTimeSeries,
    *,
    times: list[datetime] | None = None,
    dt: timedelta | None | object = _NO_SUPER,
    ensemble_size: int | None = None,
    contains_ensemble: bool | None = None,
) -> FewsTimeSeries:
    return FewsTimeSeries(
        times=list(times if times is not None else source.times),
        timezone=source.timezone,
        forecast_datetime=source.forecast_datetime,
        dt=source.dt if dt is _NO_SUPER else dt,
        contains_ensemble=(
            source.contains_ensemble if contains_ensemble is None else contains_ensemble
        ),
        ensemble_size=source.ensemble_size if ensemble_size is None else ensemble_size,
        version=source.version,
        miss_value=source.miss_value,
    )


def _validate_times(timeseries: FewsTimeSeries, *, require_equidistant: bool) -> None:
    if any(
        left >= right
        for left, right in zip(timeseries.times, timeseries.times[1:], strict=False)
    ):
        raise ValueError("FewsIOMixin: Time stamps must be strictly increasing.")
    if require_equidistant and len(timeseries.times) > 2:
        step = timeseries.times[1] - timeseries.times[0]
        if any(
            right - left != step
            for left, right in zip(timeseries.times, timeseries.times[1:], strict=False)
        ):
            raise ValueError("FewsIOMixin: Expecting equidistant timeseries.")


def _map_parameter(
    data_config: DataConfig,
    parameter_id: str,
    location_id: str | None,
    model_id: str | None,
) -> str:
    try:
        return data_config.parameter(parameter_id, location_id, model_id)
    except KeyError:
        return parameter_id


def _pi_key_for_variable(data_config: DataConfig, variable: str) -> PiSeriesKey:
    ids = data_config.pi_variable_ids(variable)
    return _series_ids_to_key(ids)


def _series_ids_to_key(ids: FewsSeriesId) -> PiSeriesKey:
    return PiSeriesKey(ids.location_id, ids.parameter_id, ids.qualifier_id)


def _output_variable_names(output_variables: Iterable[Any]) -> list[str]:
    names = []
    for variable in output_variables:
        names.append(variable.name() if hasattr(variable, "name") else str(variable))
    return names


def _aliases(problem: Any, variable: str) -> Iterable[str]:
    relation = getattr(problem, "alias_relation", None)
    aliases = getattr(relation, "aliases", None)
    if aliases is None:
        return (variable,)
    return aliases(variable)


def _result_values(results: Mapping[str, Any], variable: str) -> np.ndarray | None:
    try:
        return np.asarray(results[variable], dtype=float)
    except KeyError:
        return None


def _stored_values(
    problem: Any, variable: str, ensemble_member: int
) -> np.ndarray | None:
    try:
        ts = problem.get_timeseries(variable, ensemble_member)
    except KeyError:
        return None
    if hasattr(ts, "values"):
        return np.asarray(ts.values, dtype=float)
    return np.asarray(ts[1], dtype=float)


def _dae_variable_names(problem: Any, *groups: str) -> list[str]:
    names: list[str] = []
    dae_variables = getattr(problem, "dae_variables", {})
    for group in groups:
        for variable in dae_variables.get(group, ()):
            names.append(
                variable.name() if hasattr(variable, "name") else str(variable)
            )
    return names


def _bound_series(
    problem: Any, variable: str, ensemble_member: int, start: int, *, lower: bool
) -> Any:
    try:
        times, values = problem.io.get_timeseries_sec(variable, ensemble_member)
    except KeyError:
        return None
    values = np.asarray(values[start:], dtype=float).copy()
    values[np.isnan(values)] = (
        np.finfo(values.dtype).min if lower else np.finfo(values.dtype).max
    )
    return _timeseries_container(times[start:], values)


def _timeseries_container(times: Iterable[float], values: Iterable[float]) -> Any:
    cls = _rtctools_timeseries_class()
    times_array = np.asarray(times, dtype=float)
    values_array = np.asarray(values, dtype=float)
    return (
        cls(times_array, values_array)
        if cls is not None
        else TimeseriesValues(times_array, values_array)
    )


def _timeseries_names(io: Any) -> Iterable[str]:
    names = getattr(io, "get_timeseries_names", None)
    if names is not None:
        return names(0)
    series = getattr(io, "_series", None)
    if series:
        return series[0].keys()
    return ()


def _alias_dict(problem: Any) -> dict[str, Any]:
    try:
        from rtctools._internal.alias_tools import AliasDict  # type: ignore
    except Exception:
        return {}
    return AliasDict(problem.alias_relation)


def _values_from_argument(values: Any) -> np.ndarray:
    if hasattr(values, "values"):
        return np.asarray(values.values, dtype=float)
    return np.asarray(values, dtype=float)


def _pad_forecast_values(
    datetimes: list[datetime], forecast_datetime: datetime | None, values: np.ndarray
) -> np.ndarray:
    if len(values) == len(datetimes):
        return values
    if forecast_datetime is None:
        raise ValueError("Cannot pad forecast values without a forecast datetime.")
    start = datetimes.index(forecast_datetime)
    padded = np.full(len(datetimes), np.nan, dtype=float)
    padded[start:start + len(values)] = values
    return padded


def _seconds_step(seconds: np.ndarray) -> timedelta | None:
    if len(seconds) < 2:
        return None
    diffs = np.diff(seconds)
    if len(set(float(value) for value in diffs)) == 1:
        return timedelta(seconds=float(diffs[0]))
    return None


def _interpolate_if_possible(
    problem: Any, variable: str, output_times: list[datetime], values: np.ndarray
) -> np.ndarray:
    if not hasattr(problem, "interpolate"):
        raise ValueError(
            f"Output values for {variable!r} do not match the export time axis length."
        )
    target_seconds = np.asarray(
        problem.io.datetime_to_sec(output_times, problem.io.reference_datetime),
        dtype=float,
    )
    try:
        source_seconds = np.asarray(problem.times(variable), dtype=float)
    except TypeError:
        source_seconds = np.asarray(problem.times(), dtype=float)
    return np.asarray(
        problem.interpolate(target_seconds, source_seconds, values), dtype=float
    )


def _rtctools_timeseries_class() -> type[Any] | None:
    try:
        from rtctools.optimization.timeseries import Timeseries  # type: ignore
    except Exception:
        return None
    return Timeseries
