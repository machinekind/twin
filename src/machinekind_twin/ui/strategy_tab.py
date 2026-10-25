"""The panel's Strategies tab: strategies from installed packages, their settings, Start and Stop.

The list comes from entry points (`strategies`). Choosing a strategy imports its package and
draws a form from its settings dataclass, prefilled from the config file's section of that name.
Start builds the settings strictly from the form and refuses a broken section, so a wrong key or
type stops it before the arm moves. The strategy then runs through `StrategyRunner`, the same
runner as the twin's test strategy.

The arm rules are the Policies tab's: a fresh confirmation before every start on a real arm, no
start under STOP or while something else drives, and STOP, Home or Connect pressed while the
strategy is starting win over it.

A handle in the 3D view gives the strategy its goal, `Observation.goal` in the base frame, every
tick, also while it runs. The twin only reports where the handle is: what the goal means and how
the arm gets there is the strategy's business.

Recording is opt-in. An enabled recording includes plugin settings and can retain camera frames
at one frame per camera per second.
"""

from __future__ import annotations

import logging
import threading
from typing import TYPE_CHECKING, Any

import numpy as np
import viser

from ..episode import EpisodeOptions
from ..kinematics import inverse
from ..strategies import StrategyPlugin, config_section, load_strategy, strategies
from ..strategy import OWNER, Strategy, StrategyRunner
from .plugin_form import BLANK, FLOAT_STEP, NONE, SettingsForm

if TYPE_CHECKING:
    from .app import TwinApp

logger = logging.getLogger(__name__)

# The form's constants live in `plugin_form`; they stay importable from here.
__all__ = ["BLANK", "FLOAT_STEP", "NONE", "StrategyTab"]

NOTHING_INSTALLED = (
    "Brak strategii. Pakiet rejestruje strategie w grupie entry points `machinekind_twin.strategies`; "
    "przykladowa `swing` instaluje `uv sync`, potem **Odswiez liste**.")

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

        # viser runs callbacks on several threads: the chosen plugin and its form change together.
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
            hint="Strategie z zainstalowanych pakietow (entry points machinekind_twin.strategies).")
        refresh = gui.add_button(
            "Odswiez liste", icon=viser.Icon.REFRESH,
            hint="Szuka strategii w zainstalowanych pakietach od nowa i czyta na nowo sekcje "
                 "konfiguracji wybranej strategii.")
        self.about_md = gui.add_markdown("")
        self.form = SettingsForm(gui, gui.add_folder("Ustawienia"))

        with gui.add_folder("Uruchom na blizniaku"):
            self.confirm = gui.add_checkbox(
                "Prawdziwe ramie: rozumiem, ze sie ruszy", False,
                hint="Prawdziwe ramie: potwierdzenie przed kazdym Uruchom (gasnie po starcie). W sim niepotrzebne.")
            self.record_episode = gui.add_checkbox(
                "Record episode", False, hint="Save this run for later inspection. Logs are kept until deleted.")
            self.keep_frames = gui.add_checkbox(
                "Save camera frames", False, disabled=True,
                hint="Save up to one image per camera per second, from frames read by the strategy.")
            self.frame_format = gui.add_dropdown(
                "Frame format", options=("JPEG", "WebP"), initial_value="JPEG", disabled=True)
            self.goal_on = gui.add_checkbox(
                "Cel z uchwytu 3D", False,
                hint="Uchwyt w widoku 3D, postawiony na koncowce, podaje strategii cel (obs.goal, uklad "
                     "podstawy) w kazdym takcie, takze w trakcie biegu. Ramieniem prowadzi strategia, nie "
                     "blizniak. Wylaczony: strategia nie dostaje celu.")
            start = gui.add_button(
                "Uruchom", icon=viser.Icon.PLAYER_PLAY, color="green",
                hint="Strategia jedzie na polaczonym ramieniu (przez nadzor) - RUSZA ramieniem. Odmowa przy "
                     "blednych ustawieniach, aktywnym STOP-ie albo gdy ramie ma polityka, fala lub "
                     "identyfikacja. Panel sie rozbraja.")
            halt = gui.add_button(
                "Zatrzymaj", icon=viser.Icon.PLAYER_STOP,
                hint="Zatrzymuje strategie - takze jeszcze startujaca; ramie trzyma ostatni cel.")
            self.run_md = gui.add_markdown("")

        self.goal_handle = self.app.server.scene.add_transform_controls(
            "/cel_strategii", scale=0.1, disable_rotations=True, visible=False)

        @self.record_episode.on_update
        def _(event):
            self.keep_frames.disabled = not self.record_episode.value
            self.frame_format.disabled = not (self.record_episode.value and self.keep_frames.value)
            if not self.record_episode.value:
                self.keep_frames.value = False

        @self.keep_frames.on_update
        def _(event):
            self.frame_format.disabled = not (self.record_episode.value and self.keep_frames.value)

        @self.goal_on.on_update
        def _(event):
            if self.goal_on.value:
                self.goal_handle.position = self._tcp_in_world()
            self.goal_handle.visible = self.goal_on.value

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
            self.form.clear()
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
            self.section_problem = self.form.fill(plugin, section)
            self.about_md.content = self._about(plugin)

    @property
    def _inputs(self) -> dict[str, Any]:
        """The form's input handles by field name."""
        return self.form.inputs

    def _nothing_chosen(self) -> str:
        if self.list_problem:
            return f"**Blad listy strategii:** {self.list_problem}"
        if len(self.pick.options) == 1:
            return NOTHING_INSTALLED
        return CHOOSE

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
        strategy, meta = self._create()
        episode = (EpisodeOptions(meta=meta, keep_frames_s=1.0 if self.keep_frames.value else None,
                                  frame_format="webp" if self.frame_format.value == "WebP" else "jpeg")
                   if self.record_episode.value else None)
        runner = StrategyRunner(app.twin, strategy, goal=self.goal, cameras=self._cameras(),
                                objects=app.perception.objects, episode=episode)

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

    def _create(self) -> tuple[Strategy[Any], dict[str, Any]]:
        """The strategy from the form, strictly, and what its run's log says about it. The config
        section only prefills the form, but a broken one is refused rather than silently left out."""
        with self._form_lock:
            plugin = self.plugin
            if plugin is None:
                raise RuntimeError("wybierz strategie, ktora sie laduje")

            plugin.check(config_section(plugin.name))
            values = self.form.values(plugin)
            meta = {"strategy": plugin.name, "source": plugin.source, "settings": dict(values)}
            return plugin.create(values), meta

    def _cameras(self) -> list[str]:
        """Enabled workspace cameras, then the cameras on the arm, each name once.

        With a real arm a camera on the arm has frames only through the enabled workspace camera of
        its name, which is listed already; one without it would die at the first tick as a missing
        frame, so it is left out.
        """
        fixed = [camera.name for camera in self.app.ws.cameras if camera.enabled]
        if not self.app.twin.arm_simulated():
            return fixed

        on_arm = [name for name in self.app.twin.spec.cameras if name not in fixed]
        return fixed + on_arm

    def goal(self) -> np.ndarray | None:
        """The handle's position in the base frame [m], or None while the handle is off."""
        if not self.goal_on.value:
            return None

        world = np.r_[self.goal_handle.position, 1.0]
        return (inverse(self.app.T_b2w) @ world)[:3]

    def _tcp_in_world(self) -> np.ndarray:
        twin = self.app.twin
        with twin.lock:
            return twin.scene.data.site_xpos[twin.scene.kin.site_id].copy()

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
        if status.episode is not None:
            line += f"  \nzapis: `{status.episode.name}`"
        if status.episode_warning:
            line += f"  \n**Recording warning:** {status.episode_warning}"
        if status.episode_error:
            line += f"  \n**zapis przebiegu:** {status.episode_error}"

        self.run_md.content = line

