# rtc-fews-io

`rtc_fews_io` provides lightweight XML-based FEWS PI I/O helpers for
RTC-Tools-style workflows. It reads and writes FEWS PI TimeSeries XML files,
maps FEWS identifiers to internal model names, reads FEWS parameter
configuration XML, and offers a mixin for integrating these pieces into
RTC-Tools optimization or simulation problems.

## Table of contents

- [Features](#features)
- [Installation](#installation)
- [Quick start](#quick-start)
- [Working with PI TimeSeries XML](#working-with-pi-timeseries-xml)
- [Mapping FEWS identifiers with `DataConfig`](#mapping-fews-identifiers-with-dataconfig)
- [Reading and writing parameters with `ParameterConfig`](#reading-and-writing-parameters-with-parameterconfig)
- [Using `FewsIOMixin` in RTC-Tools-style problems](#using-fewsiomixin-in-rtc-tools-style-problems)
- [Supported file types and conventions](#supported-file-types-and-conventions)
- [Development](#development)

## Features

- **FEWS PI TimeSeries XML adapter**
  - Read FEWS PI TimeSeries XML into a normalized in-memory representation.
  - Write normalized time series back to PI TimeSeries XML.
  - Store values as NumPy arrays indexed by ensemble member and variable ID.
  - Preserve units, timezone, forecast datetime, time step metadata, FEWS
    location IDs, parameter IDs, and qualifier IDs.
  - Convert FEWS missing values to `numpy.nan` when reading and back to the PI
    missing value when writing.
  - Support equidistant and non-equidistant time steps.
  - Support ensemble time series and expansion of common series to ensemble
    members.

- **FEWS/RTC mapping support**
  - Read `rtcDataConfig.xml` files.
  - Map FEWS PI time series identifiers to internal RTC-Tools variable names.
  - Map FEWS PI model parameter identifiers to internal RTC-Tools parameter
    names.
  - Read configured import/export time series basenames.

- **Parameter configuration support**
  - Read `rtcParameterConfig.xml`-style PI parameter files.
  - Read boolean, integer, floating point, string, and table parameter values.
  - Update scalar parameters and write the modified configuration back to XML.
  - Iterate over parameters as `(location_id, model_id, parameter_id, value)`
    tuples.

- **RTC-Tools-style integration**
  - `FewsIOMixin` reads mapped FEWS inputs into an RTC-Tools-style datastore.
  - Supports optimization and simulation behavior through `fews_io_mode`.
  - Reads one or more parameter configuration files.
  - Adds numerical parameter configuration values as solver options.
  - Writes mapped FEWS PI TimeSeries XML output.

## Installation

Install from PyPI:

```powershell
python -m pip install rtc-fews-io
```

For development, install the package in editable mode with test and development
dependencies:

```powershell
python -m pip install -e .[test,dev]
```

The package requires Python 3.11 or newer.

## Quick start

```python
from rtc_fews_io import FewsTimeSeries

series = FewsTimeSeries.read("timeseries_import.xml")

print(series.times)
print(series.variable_ids())

values = series.get("Reservoir:QI:TEST")
unit = series.get_unit("Reservoir:QI:TEST")

print(values, unit)
```

## Working with PI TimeSeries XML

### Read a FEWS PI TimeSeries file

```python
from rtc_fews_io import FewsTimeSeries

ts = FewsTimeSeries.read("timeseries_import.xml")

print(ts.start_datetime)
print(ts.end_datetime)
print(ts.forecast_datetime)
print(ts.dt)
print(ts.timezone)

for variable_id, values in ts.items():
    print(variable_id, values)
```

`FewsTimeSeries` stores values on a single global datetime axis. If different
series cover different time ranges, missing values on the global axis are stored
as `numpy.nan`.

### Read ensemble members

```python
from rtc_fews_io import FewsTimeSeries

ts = FewsTimeSeries.read("timeseries_import_ensemble.xml")

if ts.contains_ensemble:
    for ensemble_member in range(ts.ensemble_size):
        for variable_id, values in ts.items(ensemble_member):
            print(ensemble_member, variable_id, values)
```

### Create and write a PI TimeSeries file

```python
from datetime import datetime, timedelta

import numpy as np

from rtc_fews_io import FewsTimeSeries, PiSeriesKey

times = [datetime(2024, 1, 1, hour) for hour in range(3)]

ts = FewsTimeSeries(
    times=times,
    timezone=0.0,
    forecast_datetime=times[0],
    dt=timedelta(hours=1),
    version="1.2",
)

ts.set(
    "Reservoir:QI:TEST",
    [1.0, np.nan, 3.0],
    key=PiSeriesKey("Reservoir", "QI", ("TEST",)),
    unit="m3/s",
)

ts.write("timeseries_export.xml")
```

## Mapping FEWS identifiers with `DataConfig`

`DataConfig` reads `rtcDataConfig.xml` and maps between FEWS PI identifiers and
internal model names.

```python
from rtc_fews_io import DataConfig, PiSeriesKey

config = DataConfig("input")  # Reads input/rtcDataConfig.xml

# FEWS PI identifiers -> internal variable name
variable = config.variable(PiSeriesKey("Reservoir", "QI", ("TEST",)))

# Internal variable name -> FEWS PI identifiers
ids = config.pi_variable_ids(variable)

print(variable)
print(ids.location_id, ids.parameter_id, ids.qualifier_id)
```

Parameter mappings can be read from `<parameter>` entries in the same data
configuration file:

```python
from rtc_fews_io import DataConfig

config = DataConfig("input")

internal_name = config.parameter("K", location_id="Loc", model_id="Model")
external_ids = config.pi_parameter_ids(internal_name)

print(internal_name)
print(external_ids.model_id, external_ids.location_id, external_ids.parameter_id)
```

If a time series mapping is missing, `DataConfig.variable(...)` returns a stable
fallback FEWS identifier of the form `locationId:parameterId[:qualifierId...]`.

## Reading and writing parameters with `ParameterConfig`

`ParameterConfig` reads PI model parameter files such as
`rtcParameterConfig.xml`.

```python
from rtc_fews_io import ParameterConfig

config = ParameterConfig("input")  # Reads input/rtcParameterConfig.xml

flag = config.get("parameters", "enabled")
count = config.get("parameters", "count")
scale = config.get("parameters", "scale")
name = config.get("parameters", "name")

print(flag, count, scale, name)
```

You can select parameter groups by location and model when those fields are
present in the XML:

```python
from rtc_fews_io import ParameterConfig

config = ParameterConfig("input", "rtcParameterConfig")
value = config.get("nested", "y", location_id="V", model="SV")
```

Scalar values can be updated and written back to XML:

```python
from rtc_fews_io import ParameterConfig

config = ParameterConfig("input")

config.set("parameters", "scale", 2.75)
config.set("parameters", "enabled", False)

output_path = config.write("output", "rtcParameterConfig")
print(output_path)
```

Table parameters are returned as dictionaries of NumPy arrays:

```python
from rtc_fews_io import ParameterConfig

config = ParameterConfig("input")

table = config.get("parameters", "curve")

print(table.keys())
print(table["x"])
```

To inspect all parameters in a file:

```python
from rtc_fews_io import ParameterConfig

config = ParameterConfig("input")

for location_id, model_id, parameter_id, value in config:
    print(location_id, model_id, parameter_id, value)
```

## Using `FewsIOMixin` in RTC-Tools-style problems

`FewsIOMixin` integrates the XML helpers with an RTC-Tools-style problem class.
It expects the host problem to expose the usual RTC-Tools `io` datastore and
input/output folder attributes.

```python
from rtc_fews_io import FewsIOMixin


class MyOptimizationProblem(FewsIOMixin, BaseOptimizationProblem):
    fews_io_mode = "optimization"

    # Defaults shown explicitly for clarity.
    timeseries_import_basename = "timeseries_import"
    timeseries_export_basename = "timeseries_export"
    pi_parameter_config_basenames = ["rtcParameterConfig"]
    pi_parameter_config_numerical_basename = "rtcParameterConfig_Numerical"
```

Typical behavior:

- `pre()` calls `read()` and imports FEWS XML inputs.
- `post()` calls `write()` and exports FEWS XML outputs.
- `read()` loads:
  - `rtcDataConfig.xml`
  - `timeseries_import.xml`
  - one or more `rtcParameterConfig.xml`-style files
  - optional `rtcParameterConfig_Numerical.xml`
- `solver_options()` is extended with values from
  `rtcParameterConfig_Numerical.xml`.

The mixin supports these modes:

| Mode | Behavior |
| --- | --- |
| `"optimization"` | Reads all ensemble members and broadcasts parameters to each member. |
| `"simulation"` | Reads one ensemble member selected by `pi_ensemble_member`. |
| `"auto"` | Infers behavior from common RTC-Tools simulation attributes. |

For simulation workflows, select an ensemble member with:

```python
from rtc_fews_io import FewsIOMixin


class MySimulationProblem(FewsIOMixin, BaseSimulationProblem):
    fews_io_mode = "simulation"
    pi_ensemble_member = 1
```

## Supported file types and conventions

| File | Purpose |
| --- | --- |
| `rtcDataConfig.xml` | Maps FEWS PI identifiers to internal variable and parameter names. |
| `timeseries_import.xml` | Default FEWS PI TimeSeries XML input file. |
| `timeseries_export.xml` | Default FEWS PI TimeSeries XML output file. |
| `rtcParameterConfig.xml` | Default FEWS PI parameter configuration input file. |
| `rtcParameterConfig_Numerical.xml` | Optional numerical solver options file. |

The default basenames can be overridden on `FewsIOMixin` subclasses.

Important conventions:

- PI TimeSeries XML is supported; binary PI files are not supported.
- Missing time series values are represented as `numpy.nan` in Python.
- FEWS time series IDs are normalized as
  `locationId:parameterId[:qualifierId...]`.
- Qualifier IDs are preserved for writing and sorted when constructing stable
  lookup IDs.
- Ensemble member indexes must be zero-based and contiguous when multiple
  ensemble members are present.

## Development

Install development dependencies and enable pre-commit checks:

```powershell
python -m pip install -e .[test,dev]
pre-commit install
```

Run the test suite:

```powershell
python -m pytest
```

Run linting and pre-commit checks manually:

```powershell
flake8 .
pre-commit run --all-files
```

Some tests can optionally read RTC-Tools fixture data when `RTC_TOOLS_ROOT`
points to a local RTC-Tools checkout. The test suite also includes synthetic
FEWS/RTC-style XML fixtures so it can run without copying RTC-Tools fixture
files into this repository.
