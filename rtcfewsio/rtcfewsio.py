import fewsxml as fx
import logging
import os

from fewsxml import FXData
from rtctools.optimization.timeseries import Timeseries
from rtctools.optimization.collocated_integrated_optimization_problem import CollocatedIntegratedOptimizationProblem
import xml.etree.ElementTree as ET
from typing import TypedDict, Dict
from datetime import datetime, timedelta
import numpy as np

logger = logging.getLogger("rtctools")

class RFData(TypedDict):
    """A dictionary representing RTC FEWS input data.

    Fields:
      - fx_data (FXData): The FXData read from timeseries_import.xml
      - current_timestep (datetime): The current timestep of the simulation
      - timestep_size (int): The time step size in seconds
      - prediction_horizon (int): The prediction horizon in number of timesteps
      - id_to_tsmeta (Dict[str, Tuple[str, str, Optional[str]]]): Mapping id -> (parameterId, locationId, qualifierId or None)
      - tsmeta_to_id (Dict[Tuple[str, str, Optional[str]], str]): Reverse mapping (parameterId, locationId, qualifierId or None) -> id
      - Ts (int): The time step size in seconds
      - T0 (datetime): The current timestep
      - Np (int): The prediction horizon in number of timesteps
    """
    fx_data: fx.FXData
    current_timestep: datetime
    timestep_size: int
    prediction_horizon: int
    id_to_tsmeta: Dict[str, tuple]
    tsmeta_to_id: Dict[tuple, str]
    Ts: int
    T0: datetime
    Np: int


class RFMixin(CollocatedIntegratedOptimizationProblem):
    """
    Mixin to add RTC FEWS input/output functionality to CollocatedIntegratedOptimizationProblem.
    """
    times_arr = None

    def set_times(self, rfdata: RFData):
        """
        Sets the times array from the RFData.

        :param rf_data:
        """
        Ts = rf_getTimestepSize(rfdata)
        Np = rf_getPredictionHorizon(rfdata)
        self.times_arr = np.array([i * Ts for i in range(Np + 1)])

    def times(self, variable=None) -> np.ndarray:
        """
        Returns the times in seconds from the reference datetime onwards.

        :param variable:
        """
        if self.times_arr is None:
            raise ValueError("Times array not set. Call set_times() first.")
        return self.times_arr


def rf_getTimestepSize(rf_data: RFData) -> int:
    """Internal helper to get timestep size in seconds from fx_data."""
    if "Ts" in rf_data:
        return rf_data["Ts"]

    fx_data = rf_data["fx_data"]
    timeseries_list = fx_data.get("timeseries", [])
    if not timeseries_list:
        raise ValueError("No timeseries found in fx_data to determine timestep size.")

    first_ts = timeseries_list[0]
    time_step_size = first_ts.get("timeStepSize")
    if time_step_size is None:
        raise ValueError("timeStepSize not found in the first timeseries.")

    rf_data["Ts"] = int(time_step_size)
    return rf_data["Ts"]  # Assuming timeStepSize is in seconds


def rf_getCurrentDatetime(rf_data: RFData) -> datetime:
    """Internal helper to get the current timestep from fx_data."""
    if "T0" in rf_data:
        return rf_data["T0"]

    fx_data = rf_data["fx_data"]
    timeseries_list = fx_data.get("timeseries", [])
    if not timeseries_list:
        raise ValueError("No timeseries found in fx_data to determine current timestep.")

    first_ts = timeseries_list[0]
    timesteps = first_ts.get("timesteps", [])
    if not timesteps:
        raise ValueError("No timesteps found in the first timeseries.")

    rf_data["T0"] = timesteps[0]
    return rf_data["T0"]  # Assuming the first timestep is the current timestep


def rf_getPredictionHorizon(rf_data: RFData) -> int:
    """Internal helper to get prediction horizon in number of timesteps from fx_data."""
    if "Np" in rf_data:
        return rf_data["Np"]

    fx_data = rf_data["fx_data"]
    timeseries_list = fx_data.get("timeseries", [])
    if not timeseries_list:
        raise ValueError("No timeseries found in fx_data to determine prediction horizon.")

    first_ts = timeseries_list[0]
    timesteps = first_ts.get("timesteps", [])
    if not timesteps:
        raise ValueError("No timesteps found in the first timeseries.")

    rf_data["Np"] = len(timesteps) - 1
    return rf_data["Np"]  # Number of timesteps in the first timeseries


def rf_read(rfmixin:RFMixin, input_folder: str, parameter_file_list: list = None) -> RFData:
    """Read timeseries_import.xml and rtcDataConfig.xml from the input folder.

    Returns a dictionary containing:
      - timeseries data from timeseries_import.xml (fx.read_xml result)
      - id_to_tsmeta: mapping id -> (parameterId, locationId, qualifierId or None)
      - tsmeta_to_id: reverse mapping (parameterId, locationId, qualifierId or None) -> id
    """
    rf_data = {}

    # Reading timeseries_import.xml
    input_folder = os.path.abspath(input_folder)
    ts_import_path = os.path.join(input_folder, "timeseries_import.xml")
    if not os.path.exists(ts_import_path):
        raise FileNotFoundError(f"Required file 'timeseries_import.xml' not found in {input_folder}")
    data_fews = fx.FXData({
        "inputFilePath": ts_import_path
    })
    fx_data = fx.read_xml(data_fews)
    rf_data["fx_data"] = fx_data
    rf_data["Ts"] = rf_getTimestepSize(rf_data)
    rf_data["T0"] = rf_getCurrentDatetime(rf_data)
    rf_data["Np"] = rf_getPredictionHorizon(rf_data)

    # Reading rtcDataConfig.xml
    rtc_config_path = os.path.join(input_folder, "rtcDataConfig.xml")
    if not os.path.exists(rtc_config_path):
        raise FileNotFoundError(f"Required file 'rtcDataConfig.xml' not found in {input_folder}")
    try:
        tree = ET.parse(rtc_config_path)
        root = tree.getroot()
    except Exception as e:
        raise RuntimeError(f"Failed to parse rtcDataConfig.xml: {e}") from e
    ns = {"fews": "http://www.wldelft.nl/fews"}
    id_to_tsmeta = {}
    tsmeta_to_id = {}
    for ts_el in root.findall(".//fews:timeSeries", ns):
        ts_id = ts_el.get("id")
        if not ts_id:
            logger.warning("Encountered <timeSeries> without id attribute; skipping.")
            continue
        pit = ts_el.find("fews:PITimeSeries", ns)
        if pit is None:
            logger.warning(f"<timeSeries id='{ts_id}'> has no <PITimeSeries>; skipping.")
            continue
        location_el = pit.find("fews:locationId", ns)
        parameter_el = pit.find("fews:parameterId", ns)
        qualifier_el = pit.find("fews:qualifierId", ns)

        location_id = location_el.text.strip() if location_el is not None and location_el.text else None
        parameter_id = parameter_el.text.strip() if parameter_el is not None and parameter_el.text else None
        qualifier_id = qualifier_el.text.strip() if qualifier_el is not None and qualifier_el.text else None

        if not location_id or not parameter_id:
            logger.warning(f"<timeSeries id='{ts_id}'> missing required locationId or parameterId; skipping.")
            continue

        key = (location_id, parameter_id, qualifier_id)

        # Handle potential duplicates
        if ts_id in id_to_tsmeta:
            logger.warning(f"Duplicate timeSeries id '{ts_id}' encountered; overwriting previous metadata.")
        if key in tsmeta_to_id and tsmeta_to_id[key] != ts_id:
            logger.warning(
                f"Metadata tuple {key} already mapped to id '{tsmeta_to_id[key]}'; new id '{ts_id}' will overwrite."
            )

        id_to_tsmeta[ts_id] = key
        tsmeta_to_id[key] = ts_id

    rf_data["id_to_tsmeta"] = id_to_tsmeta
    rf_data["tsmeta_to_id"] = tsmeta_to_id

    # Reading parameters if provided
    if parameter_file_list:
        parameters = {}
        pi_ns = {'pi': 'http://www.wldelft.nl/fews/PI'}
        for param_file in parameter_file_list:
            param_path = os.path.join(input_folder, param_file)
            if not os.path.exists(param_path):
                logger.warning(f"Parameter file '{param_file}' not found in {input_folder}; skipping.")
                continue
            try:
                tree = ET.parse(param_path)
                root = tree.getroot()
            except Exception as e:
                logger.warning(f"Failed to parse parameter file '{param_file}': {e}; skipping.")
                continue
            # Find all <parameter> elements under any <group>
            for param_el in root.findall('.//pi:parameter', pi_ns):
                param_id = param_el.get('id')
                dbl_value_el = param_el.find('pi:dblValue', pi_ns)
                int_value_el = param_el.find('pi:intValue', pi_ns)
                str_value_el = param_el.find('pi:stringValue', pi_ns)
                if param_id:
                    if dbl_value_el is not None and dbl_value_el.text:
                        parameters[param_id] = float(dbl_value_el.text.strip())
                    elif int_value_el is not None and int_value_el.text:
                        parameters[param_id] = int(int_value_el.text.strip())
                    elif str_value_el is not None and str_value_el.text:
                        parameters[param_id] = str_value_el.text.strip()
        rf_data["parameters"] = parameters

    rfmixin.set_times(rf_data)

    return rf_data


def rf_write(rf_data: RFData, results: list, output_variables: list) -> None:
    """Write output data to FEWS"""

    timeseries = []
    for output_var in output_variables:
        if output_var in results:
            values = results[output_var]
            location_id = rf_data["id_to_tsmeta"][output_var][0]
            parameter_id = rf_data["id_to_tsmeta"][output_var][1]
            qualifier_id = rf_data["id_to_tsmeta"][output_var][2]
            # timesteps = rf_data["fx_data"].get("timeseries", [])[0].get("timesteps", [])
            Ts = rf_getTimestepSize(rf_data)
            start_datetime = rf_getCurrentDatetime(rf_data)
            end_datetime = start_datetime + timedelta(seconds=Ts * (len(values) - 1))
            timesteps = [start_datetime + timedelta(seconds=Ts * i) \
                         for i in range(rf_getPredictionHorizon(rf_data) + 1)]
            timeseries_dict = fx.FXTimeseries({
                "locationId": location_id,
                "parameterId": parameter_id,
                "timesteps": timesteps,
                "values": values,
                "timeStepSize": Ts,
                "startDateTime": start_datetime,
                "endDateTime": end_datetime,
            })
            if qualifier_id is not None:
                timeseries_dict["qualifierId"] = qualifier_id
            timeseries.append(timeseries_dict)
        else:
            logger.warning(f"Output variable '{output_var}' not found in results; skipping.")
            continue
    if timeseries:
        data = fx.FXData({
            "timeseries": timeseries,
            "outputFilePath": os.path.join("..", "output", "timeseries_export.xml")
        })
        fx.write_xml(data)
        logger.info(f"timeseries_export.xml has been written successfully.")
    else:
        logger.warning("No timeseries to export; skipping write operation.")


def _find_timeseries_dict(rf_data: RFData, var_name: str):
    """Internal helper to find timeseries dictionary by FEWS id (var_name).

    Returns the timeseries dictionary or None if not found.
    """
    id_to_tsmeta = rf_data.get("id_to_tsmeta", {})

    ts_meta = id_to_tsmeta.get(var_name)
    if not ts_meta:
        return None

    location_id, parameter_id, qualifier_id = ts_meta
    timeseries_list = rf_data["fx_data"].get("timeseries", [])

    for ts_dict in timeseries_list:
        ts_location = ts_dict.get("locationId")
        ts_parameter = ts_dict.get("parameterId")
        ts_qualifier = ts_dict.get("qualifierId")

        # Match location and parameter (required)
        if ts_location == location_id and ts_parameter == parameter_id:
            # Check qualifier match (both None or both equal)
            if (qualifier_id is None and ts_qualifier is None) or (qualifier_id == ts_qualifier):
                return ts_dict

    # No matching timeseries found
    logger.warning(f"No timeseries found matching metadata {ts_meta} for variable '{var_name}'")
    return None


def rf_getValues(rf_data: RFData, var_name: str, fromDatetime:datetime = None, toDatetime:datetime = None) -> np.ndarray:
    """Retrieve values as a numpy array by FEWS id (var_name) from parsed fx_data.

    Returns numpy array of values or None if not found.
    """
    import numpy as np

    ts_dict = _find_timeseries_dict(rf_data, var_name)
    if ts_dict is None:
        return None

    datetimes = ts_dict.get("timesteps", [])
    values = ts_dict.get("values", [])
    miss_val = ts_dict.get("missVal")
    miss_val_float = float(miss_val) if miss_val is not None else None
    values = [v if v != miss_val_float else np.nan for v in values]

    filtered_values = []
    for t, v in zip(datetimes, values):
        if (fromDatetime is None or t >= fromDatetime) and (toDatetime is None or t <= toDatetime):
            filtered_values.append(v)

    return np.array(filtered_values)


def rf_getTimesteps(rf_data: RFData, var_name: str, in_seconds:bool = False) -> np.ndarray:
    """Retrieve timesteps as an array by FEWS id (var_name) from parsed fx_data.

    Returns array of timesteps or None if not found.

    If the parameter in_seconds is True, the timesteps are returned as seconds from the current datetime,
    otherwise as datetime objects.
    """
    import numpy as np

    fx_data = rf_data["fx_data"]

    ts_dict = _find_timeseries_dict(fx_data, var_name)
    if ts_dict is None:
        return None

    timesteps = ts_dict.get("timesteps", [])
    if in_seconds:
        current_datetime = rf_getCurrentDatetime(rf_data)
        timestep_size = rf_getTimestepSize(rf_data)
        timesteps = [(t - current_datetime).total_seconds() for t in timesteps]
    return np.array(timesteps)


def rf_getTimeseries(fx_data:FXData, var_name: str, fromDatetime:datetime = None, toDatetime:datetime = None) -> Timeseries:
    """Retrieve a timeseries by FEWS id (var_name) from parsed fx_data.

    Returns the Timeseries object or None if not found.
    """
    ts_dict = _find_timeseries_dict(fx_data, var_name)
    if ts_dict is None:
        return None

    timesteps = ts_dict.get("timesteps", [])
    values = ts_dict.get("values", [])

    filtered_times = []
    filtered_values = []
    for t, v in zip(timesteps, values):
        if (fromDatetime is None or t >= fromDatetime) and (toDatetime is None or t <= toDatetime):
            filtered_times.append(t)
            filtered_values.append(v)

    return Timeseries(times=filtered_times, values=filtered_values)
