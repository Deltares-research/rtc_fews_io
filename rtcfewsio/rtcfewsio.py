import fewsxml as fx
import logging
import os

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
    """
    fx_data: fx.FXData
    current_timestep: datetime
    timestep_size: int
    prediction_horizon: int
    id_to_tsmeta: Dict[str, tuple]
    tsmeta_to_id: Dict[tuple, str]


class RFMixin(CollocatedIntegratedOptimizationProblem):
    """
    Mixin to add RTC FEWS input/output functionality to CollocatedIntegratedOptimizationProblem.
    """
    def times(self, variable=None) -> np.ndarray:
        """
        Returns the times in seconds from the reference datetime onwards.

        :param variable:
        """
        return np.array([0])


def _get_timestep_size(fx_data) -> int:
    """Internal helper to get timestep size in seconds from fx_data."""
    timeseries_list = fx_data.get("timeseries", [])
    if not timeseries_list:
        raise ValueError("No timeseries found in fx_data to determine timestep size.")

    first_ts = timeseries_list[0]
    time_step_size = first_ts.get("timeStepSize")
    if time_step_size is None:
        raise ValueError("timeStepSize not found in the first timeseries.")

    return int(time_step_size)  # Assuming timeStepSize is in seconds


def _get_current_timestep(fx_data) -> datetime:
    """Internal helper to get the current timestep from fx_data."""
    timeseries_list = fx_data.get("timeseries", [])
    if not timeseries_list:
        raise ValueError("No timeseries found in fx_data to determine current timestep.")

    first_ts = timeseries_list[0]
    timesteps = first_ts.get("timesteps", [])
    if not timesteps:
        raise ValueError("No timesteps found in the first timeseries.")

    return timesteps[0]  # Assuming the first timestep is the current timestep


def _get_prediction_horizon(fx_data) -> int:
    """Internal helper to get prediction horizon in number of timesteps from fx_data."""
    timeseries_list = fx_data.get("timeseries", [])
    if not timeseries_list:
        raise ValueError("No timeseries found in fx_data to determine prediction horizon.")

    first_ts = timeseries_list[0]
    timesteps = first_ts.get("timesteps", [])
    if not timesteps:
        raise ValueError("No timesteps found in the first timeseries.")

    return len(timesteps) - 1  # Number of timesteps in the first timeseries


def rf_read(input_folder) -> RFData:
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
    rf_data["timestep_size"] = _get_timestep_size(fx_data)
    rf_data["current_timestep"] = _get_current_timestep(fx_data)
    rf_data["prediction_horizon"] = _get_prediction_horizon(fx_data)

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
    for ts_el in root.findall("fews:timeSeries", ns):
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
            timesteps = rf_data["fx_data"].get("timeseries", [])[0].get("timesteps", [])
            timestep_size = rf_data["timestep_size"]
            start_datetime = rf_data["current_timestep"]
            end_datetime = start_datetime + timedelta(seconds=timestep_size * (len(values) - 1))
            timeseries_dict = fx.FXTimeseries({
                "locationId": location_id,
                "parameterId": parameter_id,
                "timesteps": timesteps,
                "values": values,
                "timeStepSize": timestep_size,
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


def rf_getValues(rf_data: RFData, var_name: str):
    """Retrieve values as a numpy array by FEWS id (var_name) from parsed fx_data.

    Returns numpy array of values or None if not found.
    """
    import numpy as np

    ts_dict = _find_timeseries_dict(rf_data, var_name)
    if ts_dict is None:
        return None

    values = ts_dict.get("values", [])
    return np.array(values)


def rf_getTimesteps(rf_data: RFData, var_name: str):
    """Retrieve timesteps as an array by FEWS id (var_name) from parsed fx_data.

    Returns array of timesteps or None if not found.
    """
    import numpy as np

    fx_data = rf_data["fx_data"]

    ts_dict = _find_timeseries_dict(fx_data, var_name)
    if ts_dict is None:
        return None

    timesteps = ts_dict.get("timesteps", [])
    return np.array(timesteps)


def rf_getTimeseries(fx_data, var_name):
    """Retrieve a timeseries by FEWS id (var_name) from parsed fx_data.

    Returns the Timeseries object or None if not found.
    """
    ts_dict = _find_timeseries_dict(fx_data, var_name)
    if ts_dict is None:
        return None

    timesteps = ts_dict.get("timesteps", [])
    values = ts_dict.get("values", [])

    return Timeseries(times=timesteps, values=values)
