"""A plugin's settings form in the panel, shared by the Strategies and the Detectors tabs.

The form draws one input per field of the plugin's settings dataclass (`plugin_settings`),
prefilled from the config file's section of the plugin's name. `values` reads it back strictly:
a required field nobody gave (left blank and untouched, or emptied) is left out, so building the
settings refuses it.
"""

from __future__ import annotations

import threading
from typing import Any

from ..plugin_settings import PluginClass, SettingField

#: The dropdown's first option, nothing chosen. A tab starts on it, so no plugin package is imported
#: (a VLA's may take seconds) until the operator chooses one.
#: Typed as str so a dropdown built with it takes the plugins' names too.
NONE: str = "-"

#: The panel keeps as many decimals of a number as its step has, so floats get a fine step.
FLOAT_STEP = 0.001

#: What the form shows for a field without a default that the config section does not give.
BLANK = {str: "", int: 0, float: 0.0, bool: False}


class SettingsForm:
    """The inputs of one plugin's settings inside a folder of the panel."""

    def __init__(self, gui: Any, folder: Any):
        self.gui = gui
        self.folder = folder
        self._inputs: dict[str, Any] = {}
        #: Required fields the config section or the operator gave. A blank one nobody touched is missing.
        self._given: set[str] = set()
        # viser runs callbacks on several threads, and a form is filled inside its folder's context.
        self._lock = threading.Lock()

    @property
    def inputs(self) -> dict[str, Any]:
        """The input handles by field name."""
        return self._inputs

    def clear(self) -> None:
        """Removes every input and hides the folder."""
        with self._lock:
            for handle in self._inputs.values():
                handle.remove()
            self._inputs = {}
            self._given = set()
            self.folder.visible = False

    def fill(self, plugin: PluginClass, section: dict[str, Any]) -> str:
        """Draws the plugin's fields prefilled from `section`. Returns what is wrong with the
        section ("" = nothing); a broken section prefills nothing."""
        problem = ""
        try:
            prefill = plugin.check(section)
        except ValueError as exc:
            problem = str(exc)
            prefill = {}

        with self._lock:
            self._given = set(prefill)
            with self.folder:
                for field in plugin.fields:
                    handle = self._add_input(field, prefill.get(field.name, _start_value(field)))
                    handle.on_update(self._given_by_operator(field.name))
                    self._inputs[field.name] = handle

            self.folder.visible = bool(plugin.fields)

        return problem

    def values(self, plugin: PluginClass) -> dict[str, Any]:
        """The form's values for the plugin's fields, without required ones nobody gave."""
        with self._lock:
            values = {}

            for field in plugin.fields:
                value = self._inputs[field.name].value
                if field.required and not self._is_given(field.name, value):
                    continue
                values[field.name] = value

            return values

    def _add_input(self, field: SettingField, value: Any) -> Any:
        gui = self.gui
        kind = field.kind.__name__
        hint = f"{kind}, wymagane" if field.required else f"{kind}, domyslnie {field.default!r}"

        if field.kind is bool:
            return gui.add_checkbox(field.name, value, hint=hint)
        if field.kind is str:
            return gui.add_text(field.name, value, hint=hint)
        if field.kind is int:
            return gui.add_number(field.name, value, step=1, hint=hint)
        return gui.add_number(field.name, value, step=FLOAT_STEP, hint=hint)

    def _given_by_operator(self, name: str):
        def given(event) -> None:
            self._given.add(name)

        return given

    def _is_given(self, name: str, value: Any) -> bool:
        # The form always holds a value; a blank required field (untouched, or an emptied text) is not given.
        return name in self._given and value != ""


def _start_value(field: SettingField) -> Any:
    if field.required:
        return BLANK[field.kind]

    return field.default
