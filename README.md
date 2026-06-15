# rtc-fews-io
`rtc_fews_io` will provide FEWS I/O support for RTC-Tools.
Phase 2 contains the XML-only PI TimeSeries adapter. It converts between
`fewsxml.PITimeSeries` models and a normalized in-memory representation with:
- a global datetime axis;
- per-ensemble NumPy value arrays;
- FEWS location/parameter/qualifier identifiers;
- units, timezone, forecast datetime, and time step metadata;
- missing values represented as `numpy.nan`.
Binary PI files are intentionally out of scope.

Phase 3 adds lightweight configuration readers for common RTC-Tools/FEWS XML
files:
- `DataConfig` maps between internal RTC-Tools variable/parameter names and
  FEWS PI identifiers from `rtcDataConfig.xml`.
- `ParameterConfig` reads and writes scalar PI model parameters and reads table
  parameters from `rtcParameterConfig.xml`-style files.

Phase 4/5 add the public `FewsIOMixin` integration layer. The mixin uses the
new parsers and XML-only `FewsTimeSeries` adapter to:
- read mapped PI TimeSeries XML into an RTC-Tools-style datastore;
- read one or more parameter configuration files;
- apply numerical parameter configuration values as solver options;
- write mapped PI TimeSeries XML output;
- dispatch small optimization/simulation behavior differences through one
  public mixin (`fews_io_mode="auto"`, `"optimization"`, or `"simulation"`).

The test suite uses small synthetic FEWS/RTC-style XML fixtures for mixin
behavior instead of copying RTC-Tools' LGPL test files. Some adapter tests can
optionally read RTC-Tools fixture data when `RTC_TOOLS_ROOT` points to a local
RTC-Tools checkout.

## Development
```powershell
python -m pip install -e .[test,dev]
pre-commit install
python -m pytest
```

We use `pre-commit` to run linting and code quality checks automatically on commit. Running `pre-commit install` installs the git hook. To run pre-commit checks manually on all files, use:
```powershell
pre-commit run --all-files
```
