from __future__ import annotations
from collections.abc import Iterator
from pathlib import Path
from typing import Any
import defusedxml.ElementTree as DefusedElementTree
import numpy as np
import xml.etree.ElementTree as ElementTree


class ParameterConfig:
    """Reader/writer for PI model parameter configuration XML files."""

    def __init__(self, folder_or_file: str | Path, basename: str | None = None):
        path = _resolve_path(folder_or_file, basename)
        if not path.exists():
            raise FileNotFoundError(f"Parameter configuration file not found: {path}")
        self.path = path
        self._tree = DefusedElementTree.parse(path)
        self._root = self._tree.getroot()

    def get(
        self,
        group_id: str,
        parameter_id: str,
        location_id: str | None = None,
        model: str | None = None,
    ) -> Any:
        """Return the value of a parameter from the first matching group."""
        for group in self._matching_groups(group_id, location_id, model):
            parameter = _find_parameter(group, parameter_id)
            if parameter is not None:
                return _parse_parameter_value(parameter)
        raise KeyError(f"No such parameter ({group_id}, {parameter_id})")

    def set(
        self,
        group_id: str,
        parameter_id: str,
        new_value: Any,
        location_id: str | None = None,
        model: str | None = None,
    ) -> None:
        """Set a scalar parameter value in the first matching group."""
        for group in self._matching_groups(group_id, location_id, model):
            parameter = _find_parameter(group, parameter_id)
            if parameter is not None:
                _set_parameter_value(parameter, new_value)
                return
        raise KeyError(f"No such parameter ({group_id}, {parameter_id})")

    def write(
        self, folder: str | Path | None = None, basename: str | None = None
    ) -> Path:
        """Write the parameter configuration and return the output path."""
        output_path = _output_path(self.path, folder, basename)
        self._tree.write(output_path, encoding="utf-8", xml_declaration=True)
        return output_path

    def __iter__(self) -> Iterator[tuple[str | None, str | None, str, Any]]:
        """Iterate over `(location_id, model_id, parameter_id, value)` tuples."""
        for group in _children_named(self._root, "group"):
            location_id = _child_text(group, "locationId")
            model_id = _child_text(group, "model")
            for parameter in _children_named(group, "parameter"):
                parameter_id = parameter.attrib.get("id")
                if parameter_id:
                    yield (
                        location_id,
                        model_id,
                        parameter_id,
                        _parse_parameter_value(parameter),
                    )

    def _matching_groups(
        self, group_id: str, location_id: str | None, model: str | None
    ) -> Iterator[ElementTree.Element]:
        for group in _children_named(self._root, "group"):
            if group.attrib.get("id") != group_id:
                continue
            group_location = _child_text(group, "locationId")
            if (
                location_id is not None
                and group_location is not None
                and group_location != location_id
            ):
                continue
            group_model = _child_text(group, "model")
            if model is not None and group_model is not None and group_model != model:
                continue
            yield group


def _resolve_path(folder_or_file: str | Path, basename: str | None) -> Path:
    base = Path(folder_or_file)
    if basename is None:
        if base.is_dir():
            return base / "rtcParameterConfig.xml"
        return base if base.suffix == ".xml" else base.with_suffix(".xml")
    filename = basename if Path(basename).suffix == ".xml" else basename + ".xml"
    return base / filename


def _output_path(
    original: Path, folder: str | Path | None, basename: str | None
) -> Path:
    output_folder = Path(folder) if folder is not None else original.parent
    if not output_folder.exists():
        raise FileNotFoundError(f"Folder not found: {output_folder}")
    if basename is None:
        output_name = original.name
    else:
        output_name = basename if Path(basename).suffix == ".xml" else basename + ".xml"
    return output_folder / output_name


def _parse_parameter_value(parameter: ElementTree.Element) -> Any:
    for child in list(parameter):
        name = _local_name(child.tag)
        if name == "description":
            continue
        if name == "boolValue":
            return (child.text or "").strip().lower() == "true"
        if name == "intValue":
            return int((child.text or "0").strip())
        if name == "dblValue":
            return float((child.text or "nan").strip())
        if name == "stringValue":
            return child.text or ""
        if name == "table":
            return _parse_table(child)
        raise ValueError(f"Unsupported parameter value tag: {child.tag}")
    raise ValueError(
        f"Parameter {parameter.attrib.get('id')!r} has no supported value element."
    )


def _set_parameter_value(parameter: ElementTree.Element, new_value: Any) -> None:
    for child in list(parameter):
        name = _local_name(child.tag)
        if name == "description":
            continue
        if name == "boolValue":
            if not isinstance(new_value, bool):
                raise TypeError("boolValue parameters require a bool value.")
            child.text = "true" if new_value else "false"
            return
        if name == "intValue":
            child.text = str(int(new_value))
            return
        if name == "dblValue":
            child.text = str(float(new_value))
            return
        if name == "stringValue":
            child.text = str(new_value)
            return
        if name == "table":
            raise TypeError("Setting table parameters is not supported.")
        raise ValueError(f"Unsupported parameter value tag: {child.tag}")
    raise ValueError(
        f"Parameter {parameter.attrib.get('id')!r} has no supported value element."
    )


def _parse_table(table: ElementTree.Element) -> dict[str, np.ndarray]:
    rows = _children_named(table, "row")
    if not rows:
        return {}
    column_keys = list(rows[0].attrib)
    column_ids = {key: key for key in column_keys}
    column_types = {key: np.dtype("S128") for key in column_keys}
    column_ids_element = _first_child(table, "columnIds")
    if column_ids_element is not None:
        column_ids.update(column_ids_element.attrib)
    column_types_element = _first_child(table, "columnTypes")
    if column_types_element is not None:
        for key, value in column_types_element.attrib.items():
            column_types[key] = _dtype_from_pi_type(value)
    parsed = {
        column_ids[key]: np.empty(len(rows), dtype=column_types[key])
        for key in column_keys
    }
    for row_index, row in enumerate(rows):
        for key in column_keys:
            parsed[column_ids[key]][row_index] = row.attrib.get(key, "")
    return parsed


def _dtype_from_pi_type(value: str) -> np.dtype:
    if value == "double":
        return np.dtype(float)
    if value in {"int", "integer", "long"}:
        return np.dtype(int)
    return np.dtype("S128")


def _find_parameter(
    group: ElementTree.Element, parameter_id: str
) -> ElementTree.Element | None:
    for parameter in _children_named(group, "parameter"):
        if parameter.attrib.get("id") == parameter_id:
            return parameter
    return None


def _children_named(
    element: ElementTree.Element, name: str
) -> list[ElementTree.Element]:
    return [child for child in list(element) if _local_name(child.tag) == name]


def _first_child(element: ElementTree.Element, name: str) -> ElementTree.Element | None:
    return next(
        (child for child in list(element) if _local_name(child.tag) == name), None
    )


def _child_text(element: ElementTree.Element, name: str) -> str | None:
    child = _first_child(element, name)
    if child is None or child.text is None:
        return None
    return child.text.strip()


def _local_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if tag.startswith("{") else tag
