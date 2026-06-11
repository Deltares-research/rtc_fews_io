from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import defusedxml.ElementTree as ET
from .timeseries import PiSeriesKey
@dataclass(frozen=True)
class FewsSeriesId:
    """External FEWS identifiers for a PI time series."""
    location_id: str
    parameter_id: str
    qualifier_id: tuple[str, ...] = ()
    @property
    def key(self) -> str:
        return _series_key(self.location_id, self.parameter_id, self.qualifier_id)
@dataclass(frozen=True)
class FewsParameterId:
    """External FEWS identifiers for a PI model parameter."""
    model_id: str = ""
    location_id: str = ""
    parameter_id: str = ""
    @property
    def key(self) -> tuple[str, str, str]:
        return (self.model_id, self.location_id, self.parameter_id)
class DataConfig:
    """Reader for `rtcDataConfig.xml` mapping files.
    The class maps between FEWS PI identifiers and internal RTC-Tools names. It intentionally
    keeps the small public surface used by RTC-Tools' PI I/O code (`variable`,
    `pi_variable_ids`, `parameter`, and `pi_parameter_ids`) while using a fresh parser.
    """
    def __init__(self, folder_or_file: str | Path):
        path = Path(folder_or_file)
        if path.is_dir():
            path = path / "rtcDataConfig.xml"
        if not path.exists():
            raise FileNotFoundError(f"rtcDataConfig.xml not found: {path}")
        self.path = path
        self.basename_import: str | None = None
        self.basename_export: str | None = None
        self._variable_by_external_id: dict[str, str] = {}
        self._external_by_variable: dict[str, FewsSeriesId] = {}
        self._parameter_by_external_id: dict[tuple[str, str, str], str] = {}
        self._external_by_parameter: dict[str, FewsParameterId] = {}
        root = ET.parse(path).getroot()
        self._read_time_series(root)
        self._read_file_basenames(root)
        self._read_parameters(root)
    def variable(self, pi_header_or_key: Any) -> str:
        """Map a PI header/key to an internal RTC-Tools variable name.
        If no mapping is present, a stable FEWS identifier string is returned.
        """
        series_id = self._series_id_from_input(pi_header_or_key)
        return self._variable_by_external_id.get(series_id, series_id)
    def pi_variable_ids(self, variable: str) -> FewsSeriesId:
        """Map an internal variable name to FEWS PI identifiers."""
        return self._external_by_variable[variable]
    def parameter(
        self,
        parameter_id: str,
        location_id: str | None = None,
        model_id: str | None = None,
    ) -> str:
        """Map FEWS parameter identifiers to an internal RTC-Tools parameter name."""
        return self._parameter_by_external_id[
            _parameter_key(model_id=model_id, location_id=location_id, parameter_id=parameter_id)
        ]
    def pi_parameter_ids(self, parameter: str) -> FewsParameterId:
        """Map an internal parameter name to FEWS PI parameter identifiers."""
        return self._external_by_parameter[parameter]
    def _read_time_series(self, root: ET.Element) -> None:
        for time_series in _iter_by_local_name(root, "timeSeries"):
            internal_id = time_series.attrib.get("id")
            if not internal_id:
                continue
            pi_time_series = _first_child(time_series, "PITimeSeries")
            if pi_time_series is None:
                continue
            external = FewsSeriesId(
                location_id=_required_child_text(pi_time_series, "locationId"),
                parameter_id=_required_child_text(pi_time_series, "parameterId"),
                qualifier_id=tuple(_child_texts(pi_time_series, "qualifierId")),
            )
            external_id = external.key
            if internal_id in self._external_by_variable:
                raise ValueError(
                    f"More than one external time series maps to internal id {internal_id!r} "
                    f"in {self.path}."
                )
            if external_id in self._variable_by_external_id:
                raise ValueError(
                    f"More than one internal time series maps to external id {external_id!r} "
                    f"in {self.path}."
                )
            self._external_by_variable[internal_id] = external
            self._variable_by_external_id[external_id] = internal_id
    def _read_file_basenames(self, root: ET.Element) -> None:
        for section in _children(root):
            section_name = _local_name(section.tag)
            if section_name not in {"import", "export"}:
                continue
            time_series_file = _first_descendant(section, "timeSeriesFile")
            if time_series_file is not None and time_series_file.text:
                basename = Path(time_series_file.text.strip()).stem
                setattr(self, f"basename_{section_name}", basename)
    def _read_parameters(self, root: ET.Element) -> None:
        for parameter in _iter_by_local_name(root, "parameter"):
            internal_id = parameter.attrib.get("id")
            if not internal_id:
                continue
            pi_parameter = _first_child(parameter, "PIParameter")
            if pi_parameter is None:
                continue
            external = FewsParameterId(
                model_id=_child_text(pi_parameter, "modelId") or "",
                location_id=_child_text(pi_parameter, "locationId") or "",
                parameter_id=_required_child_text(pi_parameter, "parameterId"),
            )
            external_key = external.key
            if internal_id in self._external_by_parameter:
                raise ValueError(
                    f"More than one external parameter maps to internal id {internal_id!r} "
                    f"in {self.path}."
                )
            if external_key in self._parameter_by_external_id:
                raise ValueError(
                    "More than one internal parameter maps to external FEWS parameter "
                    f"{external_key!r} in {self.path}."
                )
            self._external_by_parameter[internal_id] = external
            self._parameter_by_external_id[external_key] = internal_id
    def _series_id_from_input(self, value: Any) -> str:
        if isinstance(value, PiSeriesKey):
            return value.id
        if isinstance(value, FewsSeriesId):
            return value.key
        if hasattr(value, "locationId") and hasattr(value, "parameterId"):
            return _series_key(
                value.locationId,
                value.parameterId,
                tuple(getattr(value, "qualifierId", None) or ()),
            )
        if hasattr(value, "tag"):
            return _series_key(
                _required_child_text(value, "locationId"),
                _required_child_text(value, "parameterId"),
                tuple(_child_texts(value, "qualifierId")),
            )
        raise TypeError(f"Cannot derive FEWS time series identifiers from {type(value)!r}.")
def _series_key(location_id: str, parameter_id: str, qualifier_ids: tuple[str, ...]) -> str:
    parts = [location_id, parameter_id, *sorted(qualifier_ids)]
    return ":".join(parts)
def _parameter_key(
    *, model_id: str | None, location_id: str | None, parameter_id: str | None
) -> tuple[str, str, str]:
    if parameter_id is None:
        raise KeyError("parameter_id cannot be None")
    return (model_id or "", location_id or "", parameter_id)
def _children(element: ET.Element) -> list[ET.Element]:
    return list(element)
def _iter_by_local_name(root: ET.Element, name: str):
    for element in root.iter():
        if _local_name(element.tag) == name:
            yield element
def _first_child(element: ET.Element, name: str) -> ET.Element | None:
    return next((child for child in _children(element) if _local_name(child.tag) == name), None)
def _first_descendant(element: ET.Element, name: str) -> ET.Element | None:
    return next(_iter_by_local_name(element, name), None)
def _child_text(element: ET.Element, name: str) -> str | None:
    child = _first_child(element, name)
    if child is None or child.text is None:
        return None
    return child.text.strip()
def _required_child_text(element: ET.Element, name: str) -> str:
    value = _child_text(element, name)
    if not value:
        raise ValueError(f"Missing required <{name}> in {element.tag}.")
    return value
def _child_texts(element: ET.Element, name: str) -> list[str]:
    return [child.text.strip() for child in _children(element) if _local_name(child.tag) == name and child.text]
def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if tag.startswith("{") else tag
