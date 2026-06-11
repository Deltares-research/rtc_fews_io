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

## Development
```powershell
python -m pip install -e .[test]
python -m pytest
```
