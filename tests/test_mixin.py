from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from rtc_fews_io import FewsIOMixin, FewsTimeSeries, PiSeriesKey
from rtc_fews_io.mixin import (
    TimeseriesValues,
    _interpolate_if_possible,
    _values_on_import_axis,
)


class _AliasRelation:
    def aliases(self, variable: str):
        return (variable,)


class _Symbol:
    def __init__(self, name: str):
        self._name = name

    def name(self) -> str:
        return self._name


class _Store:
    def __init__(self):
        self.reference_datetime = None
        self._datetimes = None
        self._series = [{}]
        self._parameters = [{}]

    @property
    def datetimes(self):
        return list(self._datetimes)

    @property
    def ensemble_size(self):
        return len(self._series)

    @property
    def times_sec(self):
        return self.datetime_to_sec(self._datetimes, self.reference_datetime)

    def set_timeseries(
        self, variable, datetimes, values, ensemble_member=0, check_duplicates=False
    ):
        del check_duplicates
        self._ensure_member(ensemble_member)
        self._datetimes = list(datetimes)
        self._series[ensemble_member][variable] = np.asarray(values, dtype=float)

    def get_timeseries_sec(self, variable, ensemble_member=0):
        return self.times_sec, self._series[ensemble_member][variable]

    def set_parameter(self, name, value, ensemble_member=0, check_duplicates=False):
        del check_duplicates
        self._ensure_member(ensemble_member)
        self._parameters[ensemble_member][name] = value

    def parameters(self, ensemble_member=0):
        return self._parameters[ensemble_member]

    def _ensure_member(self, ensemble_member):
        while len(self._series) <= ensemble_member:
            self._series.append({})
            self._parameters.append({})

    @staticmethod
    def datetime_to_sec(datetimes, reference):
        return np.asarray(
            [(value - reference).total_seconds() for value in datetimes], dtype=float
        )


class _BaseProblem:
    alias_relation = _AliasRelation()

    def __init__(self, **kwargs):
        self._input_folder = kwargs["input_folder"]
        self._output_folder = kwargs["output_folder"]
        self.io = _Store()


class _AbstractTimeseriesBase(_BaseProblem):
    def get_timeseries(self, variable, ensemble_member=0):
        del variable, ensemble_member
        raise NotImplementedError

    def set_timeseries(
        self, variable, values, ensemble_member=0, output=True, check_consistency=True
    ):
        del variable, values, ensemble_member, output, check_consistency
        raise NotImplementedError


class _CopyPreservingMapping(dict):
    def copy(self):
        return type(self)(self)


class _AliasPreservingMapping(_CopyPreservingMapping):
    def __getitem__(self, key):
        if key == "VZM_storage":
            key = "storage_VZM.V"
        return super().__getitem__(key)


class _BoundsBase(_BaseProblem):
    def bounds(self, ensemble_member=None):
        del ensemble_member
        return _CopyPreservingMapping({"x": (0.0, 1.0)})


class _OptimizationProblem(FewsIOMixin, _BaseProblem):
    fews_io_mode = "optimization"
    pi_parameter_config_basenames = ["rtcParameterConfig"]

    @property
    def output_variables(self):
        return [_Symbol("x")]

    def times(self, variable=None):
        del variable
        return np.asarray([0.0, 3600.0, 7200.0])

    def extract_results(self, ensemble_member):
        return {
            "x": np.asarray(
                [ensemble_member + 10.0, ensemble_member + 11.0, ensemble_member + 12.0]
            )
        }


class _DefaultTimesOptimizationProblem(FewsIOMixin, _BaseProblem):
    fews_io_mode = "optimization"
    pi_parameter_config_basenames = ["rtcParameterConfig"]

    @property
    def output_variables(self):
        return [_Symbol("x")]

    def extract_results(self, ensemble_member):
        return {
            "x": np.asarray(
                [ensemble_member + 10.0 + i for i in range(len(self.times()))]
            )
        }


class _IntegerOutputOptimizationProblem(_OptimizationProblem):
    def extract_results(self, ensemble_member):
        return {"x": np.asarray([10, 11, 12], dtype=int)}


class _ChangingOptimizationProblem(FewsIOMixin, _BaseProblem):
    fews_io_mode = "optimization"
    pi_parameter_config_basenames = ["rtcParameterConfig"]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.result_offset = 10.0

    @property
    def output_variables(self):
        return [_Symbol("x")]

    def times(self, variable=None):
        del variable
        return np.asarray([0.0, 3600.0, 7200.0])

    def extract_results(self, ensemble_member):
        return {
            "x": np.asarray(
                [
                    self.result_offset + ensemble_member,
                    self.result_offset + ensemble_member + 1.0,
                    self.result_offset + ensemble_member + 2.0,
                ]
            )
        }


class _OptimizationProblemWithAbstractTimeseriesBase(
    FewsIOMixin, _AbstractTimeseriesBase
):
    fews_io_mode = "optimization"


class _OptimizationProblemWithCustomBounds(FewsIOMixin, _BoundsBase):
    fews_io_mode = "optimization"
    dae_variables = {"free_variables": ()}


class _SimulationProblem(FewsIOMixin, _BaseProblem):
    fews_io_mode = "simulation"
    pi_ensemble_member = 1

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._simulation_times = [0.0, 3600.0, 7200.0]
        self._io_output_variables = ["x"]
        self._io_output = {"x": [20.0, 21.0, 22.0]}

    def times(self, variable=None):
        del variable
        return np.asarray(self._simulation_times)


def test_fews_io_mixin_reads_optimization_inputs_and_writes_mapped_output(tmp_path):
    _write_case(tmp_path)
    problem = _OptimizationProblem(input_folder=tmp_path, output_folder=tmp_path)

    problem.read()

    assert problem.timeseries_import.path == str(tmp_path / "timeseries_import.xml")
    assert problem.timeseries_export.path == str(tmp_path / "timeseries_export.xml")
    assert problem.equidistant is True
    assert problem.io.reference_datetime == datetime(2024, 1, 1)
    np.testing.assert_allclose(
        problem.io.get_timeseries_sec("x", 0)[1], [1.0, 2.0, 3.0]
    )
    np.testing.assert_allclose(
        problem.io.get_timeseries_sec("x", 1)[1], [4.0, 5.0, 6.0]
    )
    assert problem.io.parameters(0)["mapped_k"] == 3.5
    assert problem.io.parameters(1)["mapped_k"] == 3.5
    assert problem.solver_options()["max_iter"] == 12

    problem.write()

    assert problem.timeseries_export.path == str(tmp_path / "timeseries_export.xml")
    xml = (tmp_path / "timeseries_export.xml").read_text(encoding="utf-8")
    assert 'value="10.0"' in xml
    exported = FewsTimeSeries.read(tmp_path / "timeseries_export.xml")
    assert exported.contains_ensemble is True
    assert exported.ensemble_size == 2
    assert exported.series_keys["Loc:X"] == PiSeriesKey("Loc", "X")
    assert exported.get_unit("Loc:X", 0) == "m"
    np.testing.assert_allclose(exported.get("Loc:X", 0), [10.0, 11.0, 12.0])
    np.testing.assert_allclose(exported.get("Loc:X", 1), [11.0, 12.0, 13.0])


def test_fews_io_mixin_preserves_integer_output_values(tmp_path):
    _write_case(tmp_path)
    problem = _IntegerOutputOptimizationProblem(
        input_folder=tmp_path, output_folder=tmp_path
    )

    problem.read()
    problem.write()

    xml = (tmp_path / "timeseries_export.xml").read_text(encoding="utf-8")
    assert 'value="10"' in xml
    assert 'value="10.0"' not in xml


def test_fews_io_mixin_reads_selected_simulation_ensemble_and_writes_single_output(
    tmp_path,
):
    _write_case(tmp_path)
    problem = _SimulationProblem(input_folder=tmp_path, output_folder=tmp_path)

    problem.read()

    np.testing.assert_allclose(problem.io.get_timeseries_sec("x")[1], [4.0, 5.0, 6.0])
    assert problem.io.parameters()["mapped_k"] == 3.5

    # Simulation mode should return numpy array directly (subscriptable)
    timeseries_values = problem.get_timeseries("x")
    assert isinstance(timeseries_values, np.ndarray)
    np.testing.assert_allclose(timeseries_values, [4.0, 5.0, 6.0])
    assert timeseries_values[0] == 4.0

    # Ensure methods callable without arguments in simulation mode
    assert problem.seed() == {}
    assert problem.constant_inputs() == {}
    assert problem.history() == {}
    assert problem.bounds() == {}
    assert problem.parameters()["mapped_k"] == 3.5

    problem.write()

    exported = FewsTimeSeries.read(tmp_path / "timeseries_export.xml")
    assert exported.ensemble_size == 1
    np.testing.assert_allclose(exported.get("Loc:X"), [20.0, 21.0, 22.0])


def test_fews_io_mixin_extract_results_preserves_alias_lookup(tmp_path):
    _write_case(tmp_path)
    problem = _SimulationProblem(input_folder=tmp_path, output_folder=tmp_path)
    problem._io_output = _AliasPreservingMapping(
        {"storage_VZM.V": [10.0, 11.0, 12.0]}
    )

    results = problem.extract_results()

    assert isinstance(results, _AliasPreservingMapping)
    np.testing.assert_allclose(results["VZM_storage"], [10.0, 11.0, 12.0])


def test_fews_io_mixin_get_timeseries_ignores_abstract_super_placeholder(tmp_path):
    _write_case(tmp_path)
    problem = _OptimizationProblemWithAbstractTimeseriesBase(
        input_folder=tmp_path, output_folder=tmp_path
    )

    problem.read()

    imported = problem.get_timeseries("x")
    np.testing.assert_allclose(imported.values, [1.0, 2.0, 3.0])

    problem.set_timeseries("x", [7.0, 8.0, 9.0], unit="m")
    updated = problem.get_timeseries("x")
    np.testing.assert_allclose(updated.values, [7.0, 8.0, 9.0])


def test_fews_io_mixin_accepts_positional_ensemble_member(tmp_path):
    _write_case(tmp_path)
    problem = _OptimizationProblemWithAbstractTimeseriesBase(
        input_folder=tmp_path, output_folder=tmp_path
    )
    problem.read()

    values = TimeseriesValues(problem.times(), np.asarray([7.0, 8.0, 9.0]))
    problem.set_timeseries("x", values, 1, False)

    np.testing.assert_allclose(problem.get_timeseries("x", 0).values, [1.0, 2.0, 3.0])
    np.testing.assert_allclose(problem.get_timeseries("x", 1).values, [7.0, 8.0, 9.0])


def test_fews_io_mixin_does_not_add_generated_optimization_series_to_imports(tmp_path):
    """
    Expected behavior:
    When solving optimization problems, internally generated timeseries stored with
    `set_timeseries(variable, values, output=False)` (such as day/night masks, helper
    indicators, or reference curves) must NOT be added to `self.timeseries_import`.

    In RTC-Tools `PIMixin`, `self.__timeseries_import.set(...)` is only called when running in
    simulation mode (`if self._is_simulation_mode():`). In optimization mode, these non-output
    series are stored only in the internal store (`self.io` / local buffer) so that they can be
    retrieved via `get_timeseries(...)` during optimization, without contaminating the original
    imported FEWS timeseries collection.
    """
    _write_case(tmp_path)
    problem = _OptimizationProblemWithAbstractTimeseriesBase(
        input_folder=tmp_path, output_folder=tmp_path
    )
    problem.read()

    # Store a non-output internal helper series (e.g. daynight flag)
    problem.set_timeseries("daynight", [1.0, -1.0, 1.0], output=False)

    # Verify that the internal store holds the series for problem retrieval
    np.testing.assert_allclose(
        problem.get_timeseries("daynight").values, [1.0, -1.0, 1.0]
    )
    # Verify that the imported FEWS timeseries collection remains pristine and untouched
    assert "daynight" not in problem.timeseries_import.variable_ids()


def test_fews_io_mixin_reports_non_equidistant_optimization_input(tmp_path):
    _write_case(tmp_path, dt=None)
    problem = _DefaultTimesOptimizationProblem(
        input_folder=tmp_path, output_folder=tmp_path
    )

    problem.read()

    assert problem.equidistant is False
    np.testing.assert_allclose(problem.times(), [0.0, 3600.0, 10800.0])

    problem.write()

    exported = FewsTimeSeries.read(tmp_path / "timeseries_export.xml")
    assert exported.dt is None
    assert exported.times == [
        datetime(2024, 1, 1),
        datetime(2024, 1, 1, 1),
        datetime(2024, 1, 1, 3),
    ]


def test_fews_io_mixin_repeated_write_uses_current_results(tmp_path):
    _write_case(tmp_path)
    problem = _ChangingOptimizationProblem(input_folder=tmp_path, output_folder=tmp_path)

    problem.read()
    problem.write()
    problem.result_offset = 100.0
    problem.write()

    exported = FewsTimeSeries.read(tmp_path / "timeseries_export.xml")
    np.testing.assert_allclose(exported.get("Loc:X", 0), [100.0, 101.0, 102.0])
    np.testing.assert_allclose(exported.get("Loc:X", 1), [101.0, 102.0, 103.0])


def test_fews_io_mixin_optimization_results_override_buffered_series(tmp_path):
    _write_case(tmp_path)
    problem = _OptimizationProblem(input_folder=tmp_path, output_folder=tmp_path)

    problem.read()
    problem.set_timeseries("x", [0.0, np.nan, np.nan])
    problem.write()

    exported = FewsTimeSeries.read(tmp_path / "timeseries_export.xml")
    np.testing.assert_allclose(exported.get("Loc:X", 0), [10.0, 11.0, 12.0])
    np.testing.assert_allclose(exported.get("Loc:X", 1), [11.0, 12.0, 13.0])


def test_fews_io_mixin_writes_series_staged_in_timeseries_export(tmp_path):
    """
    Expected behavior:
    When a timeseries is staged directly into `problem.timeseries_export` during optimization
    mode (e.g. custom post-processed metrics or supplementary export series), `FewsIOMixin.write()`
    must preserve and merge these staged series into the final exported FEWS PI XML file
    alongside the optimization results.

    The exported series must maintain its mapped locationId, parameterId, and qualifierId
    (e.g. Loc:X:Maximum).
    """
    _write_case(tmp_path)
    data_config = """<?xml version="1.0" encoding="UTF-8"?>
<rtcDataConfig xmlns="http://www.wldelft.nl/fews">
    <timeSeries id="x">
        <PITimeSeries>
            <locationId>Loc</locationId>
            <parameterId>X</parameterId>
        </PITimeSeries>
    </timeSeries>
    <timeSeries id="x_max">
        <PITimeSeries>
            <locationId>Loc</locationId>
            <parameterId>X</parameterId>
            <qualifierId>Maximum</qualifierId>
        </PITimeSeries>
    </timeSeries>
</rtcDataConfig>
"""
    (tmp_path / "rtcDataConfig.xml").write_text(data_config, encoding="utf-8")
    problem = _OptimizationProblem(input_folder=tmp_path, output_folder=tmp_path)

    problem.read()
    # Stage an extra output series directly on timeseries_export
    problem.timeseries_export.set("x_max", [20.0, 21.0, 22.0])
    problem.write()

    # Verify that the staged series is included in the exported XML with correct key and values
    exported = FewsTimeSeries.read(tmp_path / "timeseries_export.xml")
    assert exported.series_keys["Loc:X:Maximum"] == PiSeriesKey(
        "Loc", "X", ("Maximum",)
    )
    np.testing.assert_allclose(
        exported.get("Loc:X:Maximum"), [20.0, 21.0, 22.0]
    )


def test_fews_io_mixin_simulation_writes_series_staged_in_timeseries_export(tmp_path):
    """
    Expected behavior:
    In simulation mode, when a timeseries is staged directly into `problem.timeseries_export`
    with custom metadata such as `unit="m"`, `FewsIOMixin.write()` must preserve both the series
    values and its custom unit in the final export, rather than omitting the series or losing
    the unit metadata.
    """
    _write_case(tmp_path)
    data_config = """<?xml version="1.0" encoding="UTF-8"?>
<rtcDataConfig xmlns="http://www.wldelft.nl/fews">
    <timeSeries id="x">
        <PITimeSeries>
            <locationId>Loc</locationId>
            <parameterId>X</parameterId>
        </PITimeSeries>
    </timeSeries>
    <timeSeries id="x_postprocessed">
        <PITimeSeries>
            <locationId>Loc</locationId>
            <parameterId>X</parameterId>
            <qualifierId>Postprocessed</qualifierId>
        </PITimeSeries>
    </timeSeries>
</rtcDataConfig>
"""
    (tmp_path / "rtcDataConfig.xml").write_text(data_config, encoding="utf-8")
    problem = _SimulationProblem(input_folder=tmp_path, output_folder=tmp_path)

    problem.read()
    # Stage a post-processed series directly with an explicit unit
    problem.timeseries_export.set(
        "x_postprocessed", [30.0, 31.0, 32.0], unit="m"
    )
    problem.write()

    # Verify that the series and its custom unit are faithfully exported to XML
    exported = FewsTimeSeries.read(tmp_path / "timeseries_export.xml")
    assert exported.get_unit("Loc:X:Postprocessed") == "m"
    np.testing.assert_allclose(
        exported.get("Loc:X:Postprocessed"), [30.0, 31.0, 32.0]
    )


def test_fews_io_mixin_simulation_writes_series_with_full_import_axis_length(tmp_path):
    """
    Expected behavior:
    In simulation mode, post-processing calculations may generate an output timeseries
    (such as advice levels or warm-up-extended series) that spans the full imported timeline
    (history + forecast horizon, e.g. 3 time steps), which is longer than the simulation
    forecast horizon (e.g. 2 time steps). When such a series is stored via
    `set_timeseries(variable, values, output=True, check_consistency=False)`, `FewsIOMixin`
    must write the entire series into the exported FEWS PI XML file starting at `start_datetime`
    with step `dt`, exactly replicating RTC-Tools `PIMixin`.

    Specifically:
    1. The series must not be forcibly truncated or sliced to the shorter simulation forecast window
       when `output.dt` is defined.
    2. Leading uncalculated / missing values (`np.nan`) must be serialized as FEWS missing values
       (`-999`).
    3. All subsequent calculated values across the timeline must be accurately preserved in the XML.
    """
    _write_case(tmp_path)
    data_config = """<?xml version="1.0" encoding="UTF-8"?>
<rtcDataConfig xmlns="http://www.wldelft.nl/fews">
    <timeSeries id="x">
        <PITimeSeries>
            <locationId>Loc</locationId>
            <parameterId>X</parameterId>
        </PITimeSeries>
    </timeSeries>
    <timeSeries id="x_advies">
        <PITimeSeries>
            <locationId>Loc</locationId>
            <parameterId>X</parameterId>
            <qualifierId>Advies</qualifierId>
        </PITimeSeries>
    </timeSeries>
</rtcDataConfig>
"""
    (tmp_path / "rtcDataConfig.xml").write_text(data_config, encoding="utf-8")
    problem = _SimulationProblem(input_folder=tmp_path, output_folder=tmp_path)
    # The imported times have 3 steps: [0.0, 3600.0, 7200.0].
    # The simulation forecast covers only the last 2 steps: [3600.0, 7200.0].
    problem._simulation_times = [3600.0, 7200.0]

    problem.read()
    # A full-length series (3 elements) is provided spanning history (NaN) and forecast (20.0, 30.0)
    full_vals = np.asarray([np.nan, 20.0, 30.0])
    problem.set_timeseries("x_advies", full_vals, output=True, check_consistency=False)
    problem.write()

    # Verify that the full 3-element series is written to the exported XML with missing value (-999)
    # and calculated values (20.0, 30.0) preserved without truncation
    exported_xml = (tmp_path / "timeseries_export.xml").read_text(encoding="utf-8")
    assert 'value="-999"' in exported_xml
    assert 'value="20.0"' in exported_xml
    assert 'value="30.0"' in exported_xml


def test_fews_io_mixin_bounds_preserves_super_mapping_type(tmp_path):
    _write_case(tmp_path)
    problem = _OptimizationProblemWithCustomBounds(
        input_folder=tmp_path, output_folder=tmp_path
    )
    problem.read()

    bounds = problem.bounds()

    assert isinstance(bounds, _CopyPreservingMapping)
    assert bounds == {"x": (0.0, 1.0)}


def test_values_on_import_axis_aligns_gapped_subset():
    """
    Expected behavior:
    When input `TimeseriesValues` contains timestamps that are a valid subset of the
    imported timeline but contain gaps (e.g. points at t=0 and t=7200, omitting t=3600),
    `_values_on_import_axis` with `check_consistency=True` must align each provided value
    to its exact matching index on the imported time axis and insert `np.nan` at the missing
    intermediate indices (resulting in `[1.0, np.nan, 3.0]`).
    """
    values = TimeseriesValues(
        times=np.asarray([0.0, 7200.0]), values=np.asarray([1.0, 3.0])
    )

    padded = _values_on_import_axis(
        values,
        variable="x",
        imported_times=np.asarray([0.0, 3600.0, 7200.0]),
        forecast_times=np.asarray([0.0, 3600.0, 7200.0]),
        initial_time=0.0,
        check_consistency=True,
    )

    # Values at matching timestamps are preserved; gaps receive NaN
    np.testing.assert_allclose(padded, [1.0, np.nan, 3.0], equal_nan=True)


def test_values_on_import_axis_allows_unaligned_times_when_check_is_disabled():
    """
    Expected behavior:
    When `check_consistency=False`, `_values_on_import_axis` should allow timeseries whose
    timestamps do not align strictly with the imported grid points (e.g. an intermediate
    timestamp like t=5400 when the grid is at 0, 3600, 7200). Instead of raising a consistency
    error, it accepts the values and pads any remaining positions with `np.nan` to match the
    imported axis length.
    """
    values = TimeseriesValues(
        times=np.asarray([0.0, 5400.0]), values=np.asarray([1.0, 2.0])
    )

    padded = _values_on_import_axis(
        values,
        variable="x",
        imported_times=np.asarray([0.0, 3600.0, 7200.0]),
        forecast_times=np.asarray([0.0, 3600.0, 7200.0]),
        initial_time=0.0,
        check_consistency=False,
    )

    # Values are accepted sequentially and trailing positions are padded with NaN
    np.testing.assert_allclose(padded, [1.0, 2.0, np.nan], equal_nan=True)


def test_values_on_import_axis_accepts_full_import_axis():
    """
    Expected behavior:
    When an input array already matches the length of the full imported time axis
    (`len(values) == len(imported_times)`), `_values_on_import_axis` with `check_consistency=False`
    must accept and return the array unchanged without misinterpreting it as covering only the
    forecast horizon and without corrupting it with spurious padding.

    This ensures that timeseries spanning the full history plus forecast horizon (such as
    post-processed advice levels spanning historical warm-up and simulation steps) can be set
    cleanly on the problem.
    """
    values = np.asarray([1.0, 2.0, 3.0, 4.0])

    padded = _values_on_import_axis(
        values,
        variable="x",
        # Imported times has length 4 (starts at t=-3600s), while forecast
        # times has length 3 (starts at t=0s)
        imported_times=np.asarray([-3600.0, 0.0, 3600.0, 7200.0]),
        forecast_times=np.asarray([0.0, 3600.0, 7200.0]),
        initial_time=0.0,
        check_consistency=False,
    )

    # The 4-element array already spans the imported axis and should be returned unchanged
    np.testing.assert_allclose(padded, values)


def test_output_values_select_exact_times_from_full_import_axis():
    """
    Expected behavior:
    When mapping an array of values defined over the full imported time axis onto a subset
    target output timeline (`output.times`), `_interpolate_if_possible` must detect that the
    source series spans `timeseries_import.times` and extract the exact values corresponding
    to the matching timestamps without numerical interpolation artifacts.

    Here, imported times have 4 hourly points (hours 0, 1, 2, 3) and the requested target times
    are hours 1, 2, 3 (length 3). The function must extract `[2.0, 3.0, 4.0]`.
    """
    imported_times = [datetime(2024, 1, 1, hour) for hour in range(4)]

    class Problem:
        timeseries_import = FewsTimeSeries(times=imported_times)

    values = _interpolate_if_possible(
        Problem(), "x", imported_times[1:], np.asarray([1.0, 2.0, 3.0, 4.0])
    )

    # Exactly selects values at hours 1, 2, 3 matching imported_times[1:]
    np.testing.assert_allclose(values, [2.0, 3.0, 4.0])


def _write_case(folder: Path, *, dt: timedelta | None = timedelta(hours=1)) -> None:
    (folder / "rtcDataConfig.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<rtcDataConfig xmlns="http://www.wldelft.nl/fews">
  <timeSeries id="x">
    <PITimeSeries>
      <locationId>Loc</locationId>
      <parameterId>X</parameterId>
    </PITimeSeries>
  </timeSeries>
  <parameter id="mapped_k">
    <PIParameter>
      <parameterId>K</parameterId>
    </PIParameter>
  </parameter>
</rtcDataConfig>
""",
        encoding="utf-8",
    )
    (folder / "rtcParameterConfig.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<pi:parameters xmlns:pi="http://www.wldelft.nl/fews/PI">
  <pi:group id="parameters">
    <pi:parameter id="K">
      <pi:dblValue>3.5</pi:dblValue>
    </pi:parameter>
  </pi:group>
</pi:parameters>
""",
        encoding="utf-8",
    )
    (folder / "rtcParameterConfig_Numerical.xml").write_text(
        """<?xml version="1.0" encoding="UTF-8"?>
<pi:parameters xmlns:pi="http://www.wldelft.nl/fews/PI">
  <pi:group id="numerics">
    <pi:parameter id="max_iter">
      <pi:intValue>12</pi:intValue>
    </pi:parameter>
  </pi:group>
</pi:parameters>
""",
        encoding="utf-8",
    )

    times = [
        datetime(2024, 1, 1),
        datetime(2024, 1, 1, 1),
        datetime(2024, 1, 1, 2) if dt is not None else datetime(2024, 1, 1, 3),
    ]
    timeseries = FewsTimeSeries(
        times=times,
        timezone=0.0,
        forecast_datetime=datetime(2024, 1, 1),
        dt=dt,
        contains_ensemble=True,
        ensemble_size=2,
        version="1.2",
    )
    timeseries.set(
        "Loc:X",
        [1.0, 2.0, 3.0],
        key=PiSeriesKey("Loc", "X"),
        unit="m",
        ensemble_member=0,
    )
    timeseries.set(
        "Loc:X",
        [4.0, 5.0, 6.0],
        key=PiSeriesKey("Loc", "X"),
        unit="m",
        ensemble_member=1,
    )
    timeseries.write(folder / "timeseries_import.xml")
