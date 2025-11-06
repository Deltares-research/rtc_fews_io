# RTC FEWS IO

`rtcfewsio` is a library that facilitates the input/output data exchange between [RTC-Tools](https://github.com/Deltares/rtc-tools) and [Delft FEWS](https://oss.deltares.nl/web/delft-fews).


## List of APIs

### Reading input data

`rf_read`

Returns `RFData` structure based on the files `timeseries_import.xml` and `rtcDataConfig.xml` in the input folder. This function should be called before running the simulation and the returned data structure should be retained for later use.

---
### Writing output data

`rf_write`

---
### Getting the timeseries values of a variable

`rf_getValues(rf_data: RFData, var_name: str) -> np.ndarray`

Returns a numpy array, representing the values of the given variable. Returns `None` if the given variable is not found in `RFData`.
