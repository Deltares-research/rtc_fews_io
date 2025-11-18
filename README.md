# RTC FEWS IO

`rtcfewsio` is a library that facilitates the input/output data exchange between [RTC-Tools](https://github.com/Deltares/rtc-tools) and [Delft FEWS](https://oss.deltares.nl/web/delft-fews).


## List of APIs
- [rf_getCurrentDatetime](#rf_getcurrentdatetime)
- [rf_getPredictionHorizon](#rf_getpredictionhorizon)
- [rf_getTimestepSize](#rf_gettimestepsize)
- [rf_getValues](#rf_getvalues)
- [rf_read](#rf_read)
- [rf_write](#rf_write)


--------------------------------------------------------------------------------
### `rf_getCurrentDatetime`
Get the current datetime of the simulation from an instance of `RFData`.

The function’s signature is:
```python
def rf_getCurrentDatetime(rf_data: RFData) -> datetime:
```

Example of usage:
```python
def pre(self):
    super().pre()
    list_parameter_files = ["../input/rtcParameterConfig1.xml", \
                        "../input/rtcParameterConfig2.xml"]
    rf_data = rf_read(self, "../input", list_parameter_files)
    currentDatetime = rf_getCurrentDatetime(rf_data)
```
In the above example, it is assumed that `pre` is a method of a class inherited 
from `RFMixin` and `CollocatedIntegratedOptimizationProblem`.

--------------------------------------------------------------------------------
### `rf_getPredictionHorizon`
A helper function to get the prediction horizon in seconds from an instance 
of `RFData`.

The function’s signature is:
```python
def rf_getPredictionHorizon(rf_data: RFData) -> int:
```

Example of usage:
```python
def pre(self):
    super().pre()
    list_parameter_files = ["../input/rtcParameterConfig1.xml", \
                        "../input/rtcParameterConfig2.xml"]
    rf_data = rf_read(self, "../input", list_parameter_files)
    Np = rf_getPredictionHorizon(rf_data)
```
In the above example, it is assumed that `pre` is a method of a class inherited from `RFMixin` and `CollocatedIntegratedOptimizationProblem`.


--------------------------------------------------------------------------------
### `rf_getTimestepSize`

A helper function to get the timestep size in seconds from an instance 
of `RFData`.

The function’s signature is:
```python
def rf_getTimestepSize(rf_data: RFData) -> int:
```

Example of usage:

```python
def pre(self):
    super().pre()
    list_parameter_files = ["../input/rtcParameterConfig1.xml", \
                        "../input/rtcParameterConfig2.xml"]
    rf_data = rf_read(self, "../input", list_parameter_files)
    timestep_size = rf_getTimestepSize(rf_data)
```
In the above example, it is assumed that `pre` is a method of a class 
inherited from `RFMixin` and `CollocatedIntegratedOptimizationProblem`.

--------------------------------------------------------------------------------
### `rf_getValues`
Returns a NumPy array representing the values of the given variable. 
Returns `None` if the given variable is not found in the given instance 
of `RFData`.

The function’s signature is:
```python
def rf_getValues(rf_data: RFData, var_name: str, fromDatetime: datetime = None, toDatetime: datetime = None) -> np.ndarray:
```

Example of usage:
```Python
def pre(self):
    super().pre()
    list_parameter_files = ["../input/rtcParameterConfig1.xml", \
                        "../input/rtcParameterConfig2.xml"]
    rf_data = rf_read(self, "../input", list_parameter_files)
    pump_Qmax = rf_getValues(rf_data, 'pump_Qmax')
```
In the above example, it is assumed that `pre` is a method of a class 
inherited from `RFMixin` and `CollocatedIntegratedOptimizationProblem`.

--------------------------------------------------------------------------------
### `rf_read`

Used for reading input data. Returns an `RFData` structure based on the 
files `timeseries_import.xml` and `rtcDataConfig.xml` in the input 
folder. This function should be called before running the simulation 
and the returned data structure should be retained for later use.

The function’s signature is:
```python
def rf_read(rfmixin: RFMixin, input_folder: str, parameter_file_list: list = None) -> RFData:
```

Example of usage:
```Python
def pre(self):
    super().pre()
    list_parameter_files = ["../input/rtcParameterConfig1.xml", \
                        "../input/rtcParameterConfig2.xml"]
    rf_data = rf_read(self, "../input", list_parameter_files)
```
In the above example, it is assumed that `pre` is a method of a class inherited 
from `RFMixin` and `CollocatedIntegratedOptimizationProblem`. 


--------------------------------------------------------------------------------
### `rf_write`

Used for writing output data. This function should be called after 
RTC-Tools has run successfully.

The function’s signature is:
```python
def rf_write(rf_data: RFData, results: list, output_variables: list) -> None:
```

Example of usage:
```Python
def post(self):
    import numpy as np
    results = self.extract_results()
    rf_write(self.rf_data, results, [sym.name() for sym in self.output_variables])
```
In the above example, it is assumed that `post` is a method of a class inherited
from `RFMixin` and `CollocatedIntegratedOptimizationProblem`, and that
`self.rf_data` was assigned in the `pre` method.
