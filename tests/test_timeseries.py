from __future__ import annotations
from datetime import datetime, timedelta
import os
from pathlib import Path
import fewsxml as fx
import numpy as np
import pytest
from rtc_fews_io import FewsTimeSeries, PiSeriesKey

RTC_TOOLS_ROOT = Path(
    os.environ.get("RTC_TOOLS_ROOT", Path(__file__).resolve().parents[2] / "rtc-tools")
)
RTC_TOOLS_DATA = RTC_TOOLS_ROOT / "tests" / "data" / "data"
requires_rtc_tools_fixtures = pytest.mark.skipif(
    not RTC_TOOLS_DATA.exists(),
    reason="RTC-Tools test fixtures are not available; set RTC_TOOLS_ROOT to enable them.",
)


@requires_rtc_tools_fixtures
def test_reads_rtc_tools_basic_fixture():
    path = RTC_TOOLS_DATA / "timeseries_import.xml"
    ts = FewsTimeSeries.read(path)
    assert ts.timezone == 0.0
    assert ts.dt == timedelta(hours=1)
    assert ts.forecast_datetime == datetime(2013, 5, 19, 22)
    assert ts.forecast_index == 0
    assert ts.variable_ids() == ("Reservoir:QI:TEST",)
    assert ts.get_unit("Reservoir:QI:TEST") == "m3/s"
    np.testing.assert_allclose(ts.get("Reservoir:QI:TEST")[:3], [0.0, 1.0, 2.0])
    assert np.isnan(ts.get("Reservoir:QI:TEST")[3])


@requires_rtc_tools_fixtures
def test_reads_rtc_tools_ensemble_fixture():
    path = RTC_TOOLS_DATA / "timeseries_import_ensemble.xml"
    ts = FewsTimeSeries.read(path)
    assert ts.contains_ensemble is True
    assert ts.ensemble_size == 2
    variable = "Reservoir:QI:TEST"
    assert ts.get(variable, 0)[2] == 2.0
    assert ts.get(variable, 1)[2] == 3.0


@requires_rtc_tools_fixtures
def test_reads_rtc_tools_nonequidistant_fixture():
    path = RTC_TOOLS_DATA / "timeseries_import_neq.xml"
    ts = FewsTimeSeries.read(path)
    assert ts.dt is None
    assert len(ts.times) == 8
    assert ts.times[1] == datetime(2013, 5, 20, 0)
    assert np.isnan(ts.get("Reservoir:QI:TEST")[-2])


def test_pads_series_with_different_ranges():
    pi = _make_two_range_model()
    ts = FewsTimeSeries.from_pi_timeseries(pi)
    assert ts.times == [datetime(2024, 1, 1, h) for h in range(4)]
    assert ts.forecast_datetime == datetime(2024, 1, 1, 1)
    assert ts.forecast_index == 1
    np.testing.assert_allclose(
        ts.get("A:x"), [1.0, 2.0, np.nan, np.nan], equal_nan=True
    )
    np.testing.assert_allclose(ts.get("B:y"), [np.nan, 3.0, 4.0, 5.0], equal_nan=True)


def test_common_series_expands_to_all_ensemble_members():
    pi = _make_mixed_ensemble_model()
    ts = FewsTimeSeries.from_pi_timeseries(pi)
    assert ts.contains_ensemble is True
    assert ts.ensemble_size == 2
    np.testing.assert_allclose(ts.get("Common:p", 0), [10.0, 11.0])
    np.testing.assert_allclose(ts.get("Common:p", 1), [10.0, 11.0])
    np.testing.assert_allclose(ts.get("Ens:q", 1), [2.0, 3.0])


def test_rejects_non_contiguous_multi_ensemble_indexes():
    pi = _make_bad_ensemble_model()
    with pytest.raises(ValueError, match="zero-based"):
        FewsTimeSeries.from_pi_timeseries(pi)


def test_roundtrip_to_pi_timeseries_and_xml(tmp_path):
    ts = FewsTimeSeries(
        times=[datetime(2024, 1, 1, 0), datetime(2024, 1, 1, 1)],
        timezone=None,
        forecast_datetime=datetime(2024, 1, 1, 0),
        dt=timedelta(hours=1),
        version="1.2",
    )
    ts.set(
        "Loc:Param:Q1",
        [1.0, np.nan],
        key=PiSeriesKey("Loc", "Param", ("Q1",)),
        unit="m",
    )
    output_path = tmp_path / "timeseries_export.xml"
    ts.write(output_path)
    reparsed = FewsTimeSeries.read(output_path)
    assert reparsed.timezone is None
    assert reparsed.series_keys["Loc:Param:Q1"] == PiSeriesKey("Loc", "Param", ("Q1",))
    assert reparsed.get_unit("Loc:Param:Q1") == "m"
    np.testing.assert_allclose(
        reparsed.get("Loc:Param:Q1"), [1.0, np.nan], equal_nan=True
    )


def _header(location, parameter, start, end, **kwargs):
    return fx.create_pi_header(
        type="instantaneous",
        location_id=location,
        parameter_id=parameter,
        start_date=start,
        end_date=end,
        timeStep=kwargs.pop("timeStep", fx.PITimeStep(unit="second", multiplier=3600)),
        missVal="-999.0",
        units="unit",
        **kwargs,
    )


def _series(location, parameter, start, values, **kwargs):
    end = start + timedelta(hours=len(values) - 1)
    header = _header(location, parameter, start, end, **kwargs)
    events = [
        {"date": start + timedelta(hours=i), "value": value}
        for i, value in enumerate(values)
    ]
    return fx.create_pi_series(header, events)


def _make_two_range_model():
    a = _series("A", "x", datetime(2024, 1, 1, 0), [1.0, 2.0])
    b = _series("B", "y", datetime(2024, 1, 1, 1), [3.0, 4.0, 5.0])
    return fx.create_pi_timeseries([a, b], time_zone=0.0, version="1.2")


def _make_mixed_ensemble_model():
    common = _series("Common", "p", datetime(2024, 1, 1), [10.0, 11.0])
    e0 = _series(
        "Ens",
        "q",
        datetime(2024, 1, 1),
        [0.0, 1.0],
        ensembleMemberIndex=0,
    )
    e1 = _series(
        "Ens",
        "q",
        datetime(2024, 1, 1),
        [2.0, 3.0],
        ensembleMemberIndex=1,
    )
    return fx.create_pi_timeseries([common, e0, e1], time_zone=0.0, version="1.2")


def _make_bad_ensemble_model():
    e1 = _series(
        "Ens",
        "q",
        datetime(2024, 1, 1),
        [0.0, 1.0],
        ensembleMemberIndex=1,
    )
    e2 = _series(
        "Ens",
        "q",
        datetime(2024, 1, 1),
        [2.0, 3.0],
        ensembleMemberIndex=2,
    )
    return fx.create_pi_timeseries([e1, e2], time_zone=0.0, version="1.2")
