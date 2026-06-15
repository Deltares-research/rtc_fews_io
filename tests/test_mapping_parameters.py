from __future__ import annotations
import os
from datetime import datetime
from pathlib import Path
import fewsxml as fx
import numpy as np
import pytest
from rtc_fews_io import DataConfig, FewsSeriesId, ParameterConfig, PiSeriesKey

RTC_TOOLS_ROOT = Path(
    os.environ.get("RTC_TOOLS_ROOT", Path(__file__).resolve().parents[2] / "rtc-tools")
)
OPT_DATA = RTC_TOOLS_ROOT / "tests" / "optimization" / "data"
DATA_DATA = RTC_TOOLS_ROOT / "tests" / "data" / "data"
requires_rtc_tools_fixtures = pytest.mark.skipif(
    not OPT_DATA.exists() or not DATA_DATA.exists(),
    reason="RTC-Tools test fixtures are not available; set RTC_TOOLS_ROOT to enable them.",
)


@requires_rtc_tools_fixtures
def test_data_config_reads_rtc_tools_mappings():
    config = DataConfig(OPT_DATA)
    ids = config.pi_variable_ids("x")
    assert ids.location_id == "States"
    assert ids.parameter_id == "X"
    assert ids.qualifier_id == ()
    header = fx.create_pi_header(
        type="instantaneous",
        location_id="States",
        parameter_id="X",
        start_date=datetime(2024, 1, 1),
        end_date=datetime(2024, 1, 1),
    )
    assert config.variable(header) == "x"
    assert config.variable(PiSeriesKey("States", "X")) == "x"
    assert (
        config.variable(PiSeriesKey("unknown", "parameter", ("q",)))
        == "unknown:parameter:q"
    )
    assert config.parameter("y", location_id="H", model_id="SV") == "SV_H_y"
    pi_parameter = config.pi_parameter_ids("SV_V_y")
    assert pi_parameter.model_id == "SV"
    assert pi_parameter.location_id == "V"
    assert pi_parameter.parameter_id == "y"


@requires_rtc_tools_fixtures
def test_data_config_reads_qualifier_fixture():
    config = DataConfig(DATA_DATA)
    ids = config.pi_variable_ids("S")
    assert ids == FewsSeriesId("Reservoir", "QI", ("TEST",))
    assert config.variable(PiSeriesKey("Reservoir", "QI", ("TEST",))) == "S"


@requires_rtc_tools_fixtures
def test_parameter_config_reads_rtc_tools_fixture():
    config = ParameterConfig(OPT_DATA, "rtcParameterConfig")
    assert config.get("parameters", "k") == 1.01
    assert config.get("nested", "y", location_id="V", model="SV") == 22.02
    values = {(loc, model, parameter): value for loc, model, parameter, value in config}
    assert values[(None, None, "k")] == 1.01
    assert values[("V", "SV", "y")] == 22.02


def test_data_config_reads_basenames_and_rejects_duplicates(tmp_path):
    _write(
        tmp_path / "rtcDataConfig.xml",
        """<?xml version="1.0" encoding="UTF-8"?>
<rtcDataConfig xmlns="http://www.wldelft.nl/fews">
  <import>
    <PITimeSeriesFile>
      <timeSeriesFile>custom_import.xml</timeSeriesFile>
    </PITimeSeriesFile>
  </import>
  <export>
    <PITimeSeriesFile>
      <timeSeriesFile>custom_export.xml</timeSeriesFile>
    </PITimeSeriesFile>
  </export>
  <timeSeries id="a">
    <PITimeSeries>
      <locationId>L</locationId>
      <parameterId>P</parameterId>
      <qualifierId>Q</qualifierId>
    </PITimeSeries>
  </timeSeries>
</rtcDataConfig>
""",
    )
    config = DataConfig(tmp_path)
    assert config.basename_import == "custom_import"
    assert config.basename_export == "custom_export"
    assert config.variable(PiSeriesKey("L", "P", ("Q",))) == "a"
    _write(
        tmp_path / "rtcDataConfig.xml",
        """<rtcDataConfig xmlns="http://www.wldelft.nl/fews">
  <timeSeries id="a">
    <PITimeSeries>
      <locationId>L</locationId>
      <parameterId>P</parameterId>
    </PITimeSeries>
  </timeSeries>
  <timeSeries id="b">
    <PITimeSeries>
      <locationId>L</locationId>
      <parameterId>P</parameterId>
    </PITimeSeries>
  </timeSeries>
</rtcDataConfig>
""",
    )
    with pytest.raises(ValueError, match="external id"):
        DataConfig(tmp_path)


def test_parameter_config_get_set_write_and_table(tmp_path):
    source = tmp_path / "rtcParameterConfig.xml"
    _write(
        source,
        """<?xml version="1.0" encoding="UTF-8"?>
<pi:parameters xmlns:pi="http://www.wldelft.nl/fews/PI" version="1.5">
  <pi:group id="g" name="g">
    <pi:locationId>L</pi:locationId>
    <pi:model>M</pi:model>
    <pi:parameter id="flag"><pi:boolValue>true</pi:boolValue></pi:parameter>
    <pi:parameter id="count"><pi:intValue>7</pi:intValue></pi:parameter>
    <pi:parameter id="scale"><pi:dblValue>1.5</pi:dblValue></pi:parameter>
    <pi:parameter id="name"><pi:stringValue>alpha</pi:stringValue></pi:parameter>
    <pi:parameter id="curve">
      <pi:table>
        <pi:columnIds c1="x" c2="label" />
        <pi:columnTypes c1="double" c2="string" />
        <pi:row c1="1.25" c2="a" />
        <pi:row c1="2.50" c2="b" />
      </pi:table>
    </pi:parameter>
  </pi:group>
</pi:parameters>
""",
    )
    config = ParameterConfig(tmp_path)
    assert config.get("g", "flag", location_id="L", model="M") is True
    assert config.get("g", "count") == 7
    assert config.get("g", "scale") == 1.5
    assert config.get("g", "name") == "alpha"
    table = config.get("g", "curve")
    np.testing.assert_allclose(table["x"], [1.25, 2.5])
    assert list(table["label"]) == [b"a", b"b"]
    config.set("g", "flag", False)
    config.set("g", "count", 8)
    config.set("g", "scale", 2.75)
    config.set("g", "name", "beta")
    config.write(tmp_path, "roundtrip")
    reread = ParameterConfig(tmp_path, "roundtrip")
    assert reread.get("g", "flag") is False
    assert reread.get("g", "count") == 8
    assert reread.get("g", "scale") == 2.75
    assert reread.get("g", "name") == "beta"


def _write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
