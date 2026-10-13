"""The panel's Strategies tab: strategies from installed packages, their settings, Start and Stop.

The list comes from entry points (`twin.strategies`). Choosing a strategy imports its package and
draws a form from its settings dataclass, prefilled from the config file's section of that name.
Start builds the settings strictly from the form and refuses a broken section, so a wrong key or
type stops it before the arm moves. The strategy then runs through `StrategyRunner`, the same
runner as the twin's test strategy.

The arm rules are the Policies tab's: a fresh confirmation before every start on a real arm, no
start under STOP or while something else drives, and STOP, Home or Connect pressed while the
strategy is starting win over it.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import viser

from ..strategies import SettingField, StrategyPlugin, config_section, load_strategy, strategies
from ..strategy import OWNER, Strategy, StrategyRunner

if TYPE_CHECKING:
    from .app import TwinApp

logger = logging.getLogger(__name__)

#: The dropdown's first option, no strategy chosen. The panel starts on it, so no strategy package is
#: imported (a VLA's may take seconds) until the operator chooses one.
NONE = "-"

#: The panel keeps as many decimals of a number as its step has, so floats get a fine step.
FLOAT_STEP = 0.001

#: What the form shows for a field without a default that the config section does not give.
BLANK = {str: "", int: 0, float: 0.0, bool: False}

NOTHING_INSTALLED = (
    "Brak strategii. Pakiet rejestruje strategie w grupie entry points `lerobot_mp.strategies`; "
    "przyklad: `pip install -e examples/strategies/swing`, potem **Odswiez liste**.")

CHOOSE = "Wybierz strategie z listy - jej pakiet laduje sie dopiero wtedy."


class StrategyTab:
    def __init__(self, app: TwinApp):
        self.app = app
        self.runner: StrategyRunner | None = None
        self.plugin: StrategyPlugin | None = None
        #: Why the list of strategies could not be read ("" = it could).
        self.list_problem = ""
        #: What is wrong with the chosen strategy's config section ("" = nothing): Start refuses it.
        self.section_problem = ""

        self._inputs: dict[str, Any] = {}
        #: Required fields the config section or the operator gave. A blank one nobody touched is missing.
        self._given: set[str] = set()
        # viser runs callbacks on several threads, and a form is filled inside its folder's context.
        self._form_lock = threading.Lock()
        # Two Start clicks run on two viser threads: the second would start another runner under the same owner.
        self._starting = threading.Lock()

        self._build()
        self.refresh()

    # ------------------------------------------------------------------ widgets

    def _build(self) -> None:
        gui = self.app.server.gui
        self.app._tab_help("Strategie")

        self.pick = gui.add_dropdown(
            "Strategia", (NONE,), initial_value=NONE,
            hint="Strategie z zainstalowanych pakietow (entry points lerobot_mp.strategies).")
        refresh = gui.add_button(
            "Odswiez liste", icon=viser.Icon.REFRESH,
            hint="Szuka strategii w zainstalowanych pakietach od nowa i czyta na nowo sekcje "
                 "konfiguracji wybranej strategii.")
        self.about_md = gui.add_markdown("")
        self.form = gui.add_folder("Ustawienia")

        with gui.add_folder("Uruchom na blizniaku"):
            self.confirm = gui.add_checkbox(
                "Prawdziwe ramie: rozumiem, ze sie ruszy", False,
                hint="Prawdziwe ramie: potwierdzenie przed kazdym Uruchom (gasnie po starcie). W sim niepotrzebne.")
            start = gui.add_button(
                "Uruchom", icon=viser.Icon.PLAYER_PLAY, color="green",
                hint="Strategia jedzie na polaczonym ramieniu (przez nadzor) - RUSZA ramieniem. Odmowa przy "
                     "blednych ustawieniach, aktywnym STOP-ie albo gdy ramie ma polityka, fala lub "
                     "identyfikacja. Panel sie rozbraja.")
            halt = gui.add_button(
                "Zatrzymaj", icon=viser.Icon.PLAYER_STOP,
                hint="Zatrzymuje strategie - takze jeszcze startujaca; ramie trzyma ostatni cel.")
            self.run_md = gui.add_markdown("")

        @refresh.on_click
        def _(event):
            self.refresh()

        @self.pick.on_update
        def _(event):
            self.show()

        @start.on_click
        @self.app._safe
        def _(event):
            self.start()

        @halt.on_click
        def _(event):
            self.halt()

    # ------------------------------------------------------------- the list

    def refresh(self) -> None:
        """Lists the installed strategies again and redraws the chosen one's form."""
        try:
            names = (NONE, *sorted(strategies()))
            self.list_problem = ""
        except ValueError as exc:
            names = (NONE,)
            self.list_problem = str(exc)

        before = self.pick.value
        # A changed choice redraws the form through `on_update`; an unchanged one is redrawn here.
        self.pick.options = names
        if self.pick.value == before:
            self.show()

    def show(self) -> None:
        """Imports the chosen strategy and draws its settings form, prefilled from its config section."""
        with self._form_lock:
            self._clear_form()
            self.plugin = None
            self.section_problem = ""

            name = self.pick.value
            if name == NONE:
                self.about_md.content = self._nothing_chosen()
                return

            try:
                plugin = load_strategy(name)
                section = config_section(name)
            except Exception as exc:
                logger.exception("Strategia %s nie laduje sie", name)
                self.about_md.content = f"**{name} nie laduje sie:** {type(exc).__name__}: {exc}"
                return

            self.plugin = plugin
            self._fill_form(plugin, section)
            self.about_md.content = self._about(plugin)

    def _nothing_chosen(self) -> str:
        if self.list_problem:
            return f"**Blad listy strategii:** {self.list_problem}"
        if len(self.pick.options) == 1:
            return NOTHING_INSTALLED
        return CHOOSE

    def _clear_form(self) -> None:
        for handle in self._inputs.values():
            handle.remove()
        self._inputs = {}
        self._given = set()
        self.form.visible = False

    def _fill_form(self, plugin: StrategyPlugin, section: dict[str, Any]) -> None:
        try:
            prefill = plugin.check(section)
        except ValueError as exc:
            self.section_problem = str(exc)
            prefill = {}

        self._given = set(prefill)

        with self.form:
            for field in plugin.fields:
                handle = self._add_input(field, prefill.get(field.name, _start_value(field)))
                handle.on_update(self._given_by_operator(field.name))
                self._inputs[field.name] = handle

        self.form.visible = bool(plugin.fields)

    def _add_input(self, field: SettingField, value: Any) -> Any:
        gui = self.app.server.gui
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

    def _about(self, plugin: StrategyPlugin) -> str:
        lines = [f"**{plugin.name}**" + (f" - {plugin.summary}" if plugin.summary else "")]
        lines.append(f"pakiet `{plugin.source}`")

        if not plugin.fields:
            lines.append("bez ustawien")
        else:
            lines.append(f"ustawienia: formularz ponizej, wypelniony z sekcji `{plugin.name}:` pliku konfiguracji")

        if self.section_problem:
            lines.append(f"**Sekcja konfiguracji:** {self.section_problem} - Uruchom odmowi, popraw plik i "
                         "Odswiez liste")

        return "  \n".join(lines)

    # ------------------------------------------------------------ start, stop

    def start(self) -> None:
        """Builds the strategy from its settings and runs it on the arm, or raises with the reason."""
        if not self._starting.acquire(blocking=False):
            raise RuntimeError("start strategii juz trwa")

        try:
            self._start()
        finally:
            self._starting.release()

    def _start(self) -> None:
        app = self.app
        # Before every check: STOP, Home, Connect or Stop from here on and the strategy never takes the arm.
        mark = app._motion_mark()

        self._check_arm()
        runner = StrategyRunner(app.twin, self._create(), cameras=self._cameras())

        self.stop("nowa strategia")
        app._release_panel()

        self.runner = runner
        try:
            self.runner.start(**app._claim_kw(mark))
        except Exception as exc:
            if not self.runner.status.stopped_because:
                self.runner.status.stopped_because = f"nie ruszyla: {exc}"
            raise

        self.confirm.value = False

    def _check_arm(self) -> None:
        app = self.app
        if not app.twin.connected:
            raise RuntimeError("polacz ramie w zakladce Ramie (sim, zeby bezpiecznie sprawdzic)")
        if not app.twin.status.simulated and not self.confirm.value:
            raise RuntimeError("prawdziwe ramie: zaznacz potwierdzenie, ze sie ruszy")
        if app._estop_active():
            raise RuntimeError("aktywny STOP - Skasuj STOP (zolty, pod STOP-em)")

        busy = app._busy(OWNER)
        if busy:
            raise RuntimeError(f"ramie zajete: {busy} - najpierw je zatrzymaj")

    def _create(self) -> Strategy[Any]:
        """The strategy from the form, strictly. The config section only prefills the form, but a
        broken one is refused rather than silently left out."""
        with self._form_lock:
            plugin = self.plugin
            if plugin is None:
                raise RuntimeError("wybierz strategie, ktora sie laduje")

            plugin.check(config_section(plugin.name))
            return plugin.create(self._form_values(plugin))

    def _form_values(self, plugin: StrategyPlugin) -> dict[str, Any]:
        values = {}

        for field in plugin.fields:
            value = self._inputs[field.name].value
            if field.required and not self._is_given(field.name, value):
                continue
            values[field.name] = value

        return values

    def _is_given(self, name: str, value: Any) -> bool:
        # The form always holds a value; a blank required field (untouched, or an emptied text) is not given.
        return name in self._given and value != ""

    def _cameras(self) -> list[str]:
        """Enabled workspace cameras, then the cameras on the arm. A real camera named like one on the
        arm is that camera (the hub serves it in its place), so it is listed once."""
        fixed = [camera.name for camera in self.app.ws.cameras if camera.enabled]
        on_arm = [name for name in self.app.twin.spec.cameras if name not in fixed]
        return fixed + on_arm

    def halt(self) -> None:
        """Strategie > Zatrzymaj: also a strategy that is still starting never takes the arm."""
        self.app._motion_gen += 1
        self.stop("zatrzymana z panelu")

    def stop(self, reason: str) -> None:
        if self.runner is not None:
            self.runner.stop(reason)

    @property
    def running(self) -> bool:
        return self.runner is not None and self.runner.status.running

    # ---------------------------------------------------------------- status

    def tick(self) -> None:
        if self.runner is None:
            return

        status = self.runner.status
        line = f"{'**jedzie**' if status.running else 'stoi'} - krok {status.step}, {status.hz:.0f} Hz"
        if status.note:
            line += f", {status.note}"
        if status.error is not None:
            line += f", blad zadania {status.error:.3g}"
        if status.done:
            line += ", **sukces**" if status.success else ", bez sukcesu"
        if not status.running and status.stopped_because:
            line += f"  \nzatrzymana: {status.stopped_because}"

        self.run_md.content = line


def _start_value(field: SettingField) -> Any:
    if field.required:
        return BLANK[field.kind]

    return field.default
