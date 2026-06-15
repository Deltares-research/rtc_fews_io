from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from rtc_fews_io import FewsIOMixin, FewsTimeSeries, PiSeriesKey


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


class _OptimizationProblemWithAbstractTimeseriesBase(
    FewsIOMixin, _AbstractTimeseriesBase
):
    fews_io_mode = "optimization"


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

    exported = FewsTimeSeries.read(tmp_path / "timeseries_export.xml")
    assert exported.contains_ensemble is True
    assert exported.ensemble_size == 2
    assert exported.series_keys["Loc:X"] == PiSeriesKey("Loc", "X")
    assert exported.get_unit("Loc:X", 0) == "m"
    np.testing.assert_allclose(exported.get("Loc:X", 0), [10.0, 11.0, 12.0])
    np.testing.assert_allclose(exported.get("Loc:X", 1), [11.0, 12.0, 13.0])


def test_fews_io_mixin_reads_selected_simulation_ensemble_and_writes_single_output(
    tmp_path,
):
    _write_case(tmp_path)
    problem = _SimulationProblem(input_folder=tmp_path, output_folder=tmp_path)

    problem.read()

    np.testing.assert_allclose(problem.io.get_timeseries_sec("x")[1], [4.0, 5.0, 6.0])
    assert problem.io.parameters()["mapped_k"] == 3.5

    problem.write()

    exported = FewsTimeSeries.read(tmp_path / "timeseries_export.xml")
    assert exported.ensemble_size == 1
    np.testing.assert_allclose(exported.get("Loc:X"), [20.0, 21.0, 22.0])


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


def _write_case(folder: Path) -> None:
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

    timeseries = FewsTimeSeries(
        times=[datetime(2024, 1, 1) + timedelta(hours=i) for i in range(3)],
        timezone=0.0,
        forecast_datetime=datetime(2024, 1, 1),
        dt=timedelta(hours=1),
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
