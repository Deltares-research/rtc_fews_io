# rtc-fews-io
`rtc_fews_io` will provide FEWS PI-XML I/O support for RTC-Tools.
Phase 2 contains the XML-only PI TimeSeries adapter. It converts between
`fewsxml.PITimeSeries` models and a normalized in-memory representation with:
- a global datetime axis;
- per-ensemble NumPy value arrays;
- FEWS location/parameter/qualifier identifiers;
- units, timezone, forecast datetime, and time step metadata;
- missing values represented as `numpy.nan`.
Binary PI files are intentionally out of scope.
## Development
```powershell
python -m pip install -e .[test]
python -m pytest
ruff check src tests
ruff format src tests
```
