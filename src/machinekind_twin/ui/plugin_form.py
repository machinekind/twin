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
DEFAULT_CHOICE = "(domyslne)"


class SettingsForm:
    """The inputs of one plugin's settings inside a folder of the panel."""

    def __init__(self, gui: Any, folder: Any):
        self.gui = gui
        self.folder = folder
        self._inputs: dict[str, Any] = {}
        #: Required fields the config section or the operator gave. A blank one nobody touched is missing.
        self._given: set[str] = set()
        self._configured: dict[str, tuple[Any, str]] = {}
        self._choice_values: dict[str, dict[str, str]] = {}
        # viser runs callbacks on several threads, and a form is filled inside its folder's context.
        # Updating dropdown options can synchronously invoke another input's callback.
        self._lock = threading.RLock()

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
            self._configured = {}
            self._choice_values = {}
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
            values = {field.name: prefill.get(field.name, _start_value(field)) for field in plugin.fields}
            self._configured = {
                field.name: (values.get(field.choices_from) if field.choices_from else None, prefill[field.name])
                for field in plugin.fields if field.choices is not None and field.name in prefill
            }
            with self.folder:
                for field in plugin.fields:
                    options = self._options(field, values)
                    value = values[field.name]
                    if options and value not in options:
                        value = options[0]
                    values[field.name] = value
                    handle = self._add_input(field, value, options)
                    self._inputs[field.name] = handle

                for field in plugin.fields:
                    handle = self._inputs[field.name]
                    handle.on_update(self._given_by_operator(plugin, field, handle))

            self.folder.visible = bool(plugin.fields)

        return problem

    def values(self, plugin: PluginClass) -> dict[str, Any]:
        """The form's values for the plugin's fields, without required ones nobody gave."""
        with self._lock:
            # A click may arrive before an input's callback on another viser thread.
            self._refresh_choices(plugin)
            values = {}

            for field in plugin.fields:
                value = self._value(field)
                if field.required and not self._is_given(field.name, value):
                    continue
                values[field.name] = value

            return values

    def _add_input(self, field: SettingField, value: Any, options: tuple[str, ...]) -> Any:
        gui = self.gui
        kind = field.kind.__name__
        hint = f"{kind}, wymagane" if field.required else f"{kind}, domyslnie {field.default!r}"

        if options:
            labels = self._labels(field, options)
            label = next(label for label, raw in self._choice_values[field.name].items() if raw == value)
            return gui.add_dropdown(field.name, labels, initial_value=label, hint=hint)
        if field.kind is bool:
            return gui.add_checkbox(field.name, value, hint=hint)
        if field.kind is str:
            return gui.add_text(field.name, value, hint=hint)
        if field.kind is int:
            return gui.add_number(field.name, value, step=1, hint=hint)
        return gui.add_number(field.name, value, step=FLOAT_STEP, hint=hint)

    def _given_by_operator(self, plugin: PluginClass, field: SettingField, handle: Any):
        def given(event) -> None:
            with self._lock:
                if self._inputs.get(field.name) is not handle:
                    return  # A callback from a form that was replaced.
                self._given.add(field.name)
                self._refresh_choices(plugin)

        return given

    def _value(self, field: SettingField) -> Any:
        value = self._inputs[field.name].value
        return self._choice_values.get(field.name, {}).get(value, value)

    def _labels(self, field: SettingField, options: tuple[str, ...]) -> tuple[str, ...]:
        empty = "(wybierz)" if field.required else DEFAULT_CHOICE
        while empty in options:
            empty = f"({empty})"
        self._choice_values[field.name] = {value or empty: value for value in options}
        return tuple(self._choice_values[field.name])

    def _options(self, field: SettingField, values: dict[str, Any]) -> tuple[str, ...]:
        if field.choices is None:
            return ()
        options = field.options(values)
        configured = self._configured.get(field.name)
        selector = values.get(field.choices_from) if field.choices_from else None
        if configured is not None and configured[0] == selector:
            if configured[1] not in options:
                options = (*options, configured[1])
        # Even a config with an unknown selector stays a dropdown; plugin validation refuses it.
        options = options or (values[field.name],)
        if field.required and "" not in options:
            options = ("", *options)
        return options

    def _refresh_choices(self, plugin: PluginClass) -> None:
        values = {f.name: self._value(f) for f in plugin.fields}
        for field in plugin.fields:
            if field.choices_from is None:
                continue
            handle = self._inputs[field.name]
            options = self._options(field, values)
            selected = values[field.name] if values[field.name] in options else options[0]
            labels = self._labels(field, options)
            if handle.options != labels:
                handle.options = labels
            handle.value = labels[options.index(selected)]
            values[field.name] = self._value(field)

    def _is_given(self, name: str, value: Any) -> bool:
        # The form always holds a value; a blank required field (untouched, or an emptied text) is not given.
        return name in self._given and value != ""


def _start_value(field: SettingField) -> Any:
    if field.required:
        return BLANK[field.kind]

    return field.default
