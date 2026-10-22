"""The panel's Detectors tab: detectors from installed packages, their settings, Start, the running ones.

The list comes from entry points (`detectors`). Choosing a detector imports its package and draws
a form from its settings dataclass, prefilled from the config file's section of that name, the
same way as the Strategies tab. Start refuses a broken section, then starts the detector on a
short-lived thread (`detektor-start:<name>`), because a detector's `start` may take seconds (a
language model's login check). Several detectors may run at once; starting one whose name is
running restarts it with the new settings. `twin ui --detector NAME` starts one at panel start,
with its config section's settings (`start_named`).

A detector never moves the arm: Start needs no arm, no confirmation, and STOP does not stop it.
The objects the detectors see go to strategies through `Perception.objects` and are listed in the
tab and drawn in the 3D view.
"""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np
import viser

from ..detection import DetectorRunner
from ..detectors import DetectorPlugin, config_section, detectors, load_detector, shared_section
from ..kinematics import pose
from ..tracks import TrackView
from .bridge import mat_to_wxyz
from .plugin_form import NONE, SettingsForm

if TYPE_CHECKING:
    from .app import TwinApp

logger = logging.getLogger(__name__)

NOTHING_INSTALLED = (
    "Brak detektorow. Pakiet rejestruje detektory w grupie entry points `machinekind_twin.detectors`; "
    "przykladowy `cube` instaluje `uv sync`, potem **Odswiez liste**.")

CHOOSE = "Wybierz detektor z listy - jego pakiet laduje sie dopiero wtedy."

NOTHING_RUNNING = "zaden detektor nie dziala"

NO_OBJECTS = "brak obiektow"

#: The reason a stop from the tab's row gives.
STOPPED_HERE = "zatrzymany z panelu"

#: Colour of a fitted object's box in the 3D view (the Map tab's cube colour).
FITTED_COLOR = (255, 80, 200)
#: Colour and radius of a point object's marker in the 3D view.
POINT_COLOR, POINT_RADIUS = (255, 80, 200), 0.01

#: How the shared config section reads in the warning, by the other kind's word.
SHARED_WITH = {"strategia": "ze strategia", "backend": "z backendem"}


@dataclass(eq=False)
class _Row:
    """One detector in the "Dzialajace" folder."""

    md: Any
    stop: Any


@dataclass(eq=False)
class _Marker:
    """One object in the 3D view: a frame with its pose, a shape and a label under it."""

    frame: Any
    shape: Any
    label: Any
    #: What the shape is: ("pudelko", size) or ("kula",). Another kind replaces the shape.
    key: tuple


class DetectorTab:
    def __init__(self, app: TwinApp):
        self.app = app
        self.plugin: DetectorPlugin | None = None
        #: Why the list of detectors could not be read ("" = it could).
        self.list_problem = ""
        #: What is wrong with the chosen detector's config section ("" = nothing): Start refuses it.
        self.section_problem = ""
        #: The last start thread, so a caller (a test) can wait for it.
        self.start_thread: threading.Thread | None = None

        # viser runs callbacks on several threads: the chosen plugin and its form change together.
        self._form_lock = threading.Lock()
        # A second Start while one is starting is refused; the start thread releases it.
        self._starting = threading.Lock()
        # The rows' state: read by the panel loop, written by callbacks and start threads.
        self._lock = threading.Lock()
        #: The last runner seen per name, kept after it stops until the list is refreshed.
        self._last: dict[str, DetectorRunner] = {}
        #: Names whose start thread runs.
        self._launching: set[str] = set()
        #: Why a start failed, by name, until the next start or refresh.
        self._failed: dict[str, str] = {}
        # Rows and markers are built by the panel loop and by Odswiez liste.
        self._rows_lock = threading.Lock()
        self._rows: dict[str, _Row] = {}
        self._markers: list[_Marker] = []

        self._build()
        self.refresh()

    # ------------------------------------------------------------------ widgets

    def _build(self) -> None:
        gui = self.app.server.gui
        self.app._tab_help("Detektory")

        self.pick = gui.add_dropdown(
            "Detektor", (NONE,), initial_value=NONE,
            hint="Detektory z zainstalowanych pakietow (entry points machinekind_twin.detectors).")
        refresh = gui.add_button(
            "Odswiez liste", icon=viser.Icon.REFRESH,
            hint="Szuka detektorow w zainstalowanych pakietach od nowa i czyta na nowo sekcje "
                 "konfiguracji wybranego detektora.")
        self.about_md = gui.add_markdown("")
        self.form = SettingsForm(gui, gui.add_folder("Ustawienia"))

        with gui.add_folder("Uruchom"):
            start = gui.add_button(
                "Uruchom detektor", icon=viser.Icon.PLAYER_PLAY, color="green",
                hint="Detektor patrzy na kamery z zaufana poza we wlasnym tempie; strategie dostaja obiekty "
                     "w obs.objects. Ramieniem nie rusza. Ten sam detektor uruchomiony ponownie bierze nowe "
                     "ustawienia.")

        self.running_folder = gui.add_folder("Dzialajace")
        with self.running_folder:
            self.none_md = gui.add_markdown(NOTHING_RUNNING)

        with gui.add_folder("Obiekty"):
            self.show_3d = gui.add_checkbox("Pokaz obiekty w 3D", True,
                                            hint="Obiekty z detektorow w widoku 3D, w ukladzie podstawy.")
            self.objects_md = gui.add_markdown(NO_OBJECTS)

        @refresh.on_click
        def _(event):
            self.refresh()

        @self.pick.on_update
        def _(event):
            self.show()

        @start.on_click
        @self.app._safe
        def _(event):
            self.start(event)

    # ------------------------------------------------------------- the list

    def refresh(self) -> None:
        """Lists the installed detectors again, redraws the chosen one's form and forgets stopped rows."""
        try:
            names = (NONE, *sorted(detectors()))
            self.list_problem = ""
        except ValueError as exc:
            names = (NONE,)
            self.list_problem = str(exc)

        running = self.app.perception.running()
        with self._lock:
            self._last = {n: r for n, r in self._last.items() if n in running}
            self._failed.clear()

        before = self.pick.value
        # A changed choice redraws the form through `on_update`; an unchanged one is redrawn here.
        self.pick.options = names
        if self.pick.value == before:
            self.show()
        self._tick_rows()

    def show(self) -> None:
        """Imports the chosen detector and draws its settings form, prefilled from its config section."""
        with self._form_lock:
            self.form.clear()
            self.plugin = None
            self.section_problem = ""

            name = self.pick.value
            if name == NONE:
                self.about_md.content = self._nothing_chosen()
                return

            try:
                plugin = load_detector(name)
                section = config_section(name)
            except Exception as exc:
                logger.exception("Detektor %s nie laduje sie", name)
                self.about_md.content = f"**{name} nie laduje sie:** {type(exc).__name__}: {exc}"
                return

            self.plugin = plugin
            self.section_problem = self.form.fill(plugin, section)
            self.about_md.content = self._about(plugin)

    def _nothing_chosen(self) -> str:
        if self.list_problem:
            return f"**Blad listy detektorow:** {self.list_problem}"
        if len(self.pick.options) == 1:
            return NOTHING_INSTALLED
        return CHOOSE

    def _about(self, plugin: DetectorPlugin) -> str:
        lines = [f"**{plugin.name}**" + (f" - {plugin.summary}" if plugin.summary else "")]
        lines.append(f"pakiet `{plugin.source}`")

        if not plugin.fields:
            lines.append("bez ustawien")
        else:
            lines.append(f"ustawienia: formularz ponizej, wypelniony z sekcji `{plugin.name}:` pliku konfiguracji")

        if self.section_problem:
            lines.append(f"**Sekcja konfiguracji:** {self.section_problem} - Uruchom odmowi, popraw plik i "
                         "Odswiez liste")

        shared = shared_section(plugin.name)
        if shared:
            lines.append(f"uwaga: sekcja konfiguracji `{plugin.name}` jest wspolna "
                         f"{SHARED_WITH.get(shared, shared)} `{plugin.name}`")

        return "  \n".join(lines)

    # ------------------------------------------------------------ start, stop

    def start(self, event: Any = None) -> None:
        """Checks the chosen detector's settings and starts it on a start thread, or raises with the reason.

        A failure of the start itself (the settings refused by the detector, its `start` refusing)
        comes later, as a notification and in the detector's row.
        """
        if not self._starting.acquire(blocking=False):
            raise RuntimeError("start detektora juz trwa")

        try:
            with self._form_lock:
                plugin = self.plugin
                if plugin is None:
                    raise RuntimeError("wybierz detektor, ktory sie laduje")
                plugin.check(config_section(plugin.name))
                values = self.form.values(plugin)

            self._launch(plugin, values, event, release=True)
        except BaseException:
            self._starting.release()
            raise

    def start_named(self, name: str) -> threading.Thread | None:
        """`twin ui --detector NAME`: what Start does, for a detector named at panel start.

        The settings are its config section's, since there is no form to read. Nothing raises: a
        detector that does not load or whose section is refused is logged, notified and shown in
        its row, and the panel starts anyway. Returns the start thread, None when it did not start.
        """
        try:
            plugin = load_detector(name)
            values = plugin.check(config_section(plugin.name))
        except Exception as exc:
            logger.exception("Detektor %s nie ruszyl przy starcie panelu", name)
            with self._lock:
                self._failed[name.lower()] = f"{type(exc).__name__}: {exc}"
            self.app._notify(None, "Detektor nie ruszyl", f"{name}: {type(exc).__name__}: {exc}", error=True)
            return None

        # Not under `_starting`: several detectors may be named, and Start stays free meanwhile.
        return self._launch(plugin, values, None, release=False)

    def _launch(self, plugin: DetectorPlugin, values: dict[str, Any], event: Any, *,
                release: bool) -> threading.Thread:
        """Starts the start thread for `plugin`; with `release` it releases `_starting` when done."""
        with self._lock:
            self._failed.pop(plugin.name, None)
            self._launching.add(plugin.name)

        thread = threading.Thread(target=self._start, args=(plugin, values, event, release),
                                  name=f"detektor-start:{plugin.name}", daemon=True)
        self.start_thread = thread
        thread.start()
        return thread

    def _start(self, plugin: DetectorPlugin, values: dict[str, Any], event: Any, release: bool) -> None:
        """The start thread: the detector's `create` and `start`, which may take seconds."""
        name = plugin.name
        try:
            runner = self.app.perception.start_detector(plugin, values)
            with self._lock:
                self._last[name] = runner
        except Exception as exc:
            logger.exception("Detektor %s nie ruszyl", name)
            with self._lock:
                self._failed[name] = f"{type(exc).__name__}: {exc}"
            self.app._notify(event, "Detektor nie ruszyl", f"{name}: {type(exc).__name__}: {exc}", error=True)
        finally:
            with self._lock:
                self._launching.discard(name)
            if release:
                self._starting.release()

    def stop(self, name: str, reason: str = STOPPED_HERE) -> None:
        """Stops the detector `name` and forgets its objects; its row says why until the next refresh."""
        self.app.perception.stop_detector(name, reason)

    def _stop_clicked(self, name: str):
        def stop(event) -> None:
            self.stop(name)
            self._tick_rows()

        return stop

    # ---------------------------------------------------------------- status

    def tick(self) -> None:
        """The panel loop: the rows of the running detectors and the objects they see."""
        self._tick_rows()
        self._tick_objects(self.app.perception.snapshot())

    def _tick_rows(self) -> None:
        perception = self.app.perception
        running = perception.running()
        now = perception.clock()

        with self._lock:
            self._last.update(running)
            names = sorted(set(self._last) | self._launching | set(self._failed))
            lines = {name: self._run_line(name, running.get(name), now) for name in names}

        with self._rows_lock:
            if list(self._rows) != names:
                self._rebuild_rows(names)
            for name in names:
                self._rows[name].md.content = lines[name]
            self.none_md.visible = not names

    def _rebuild_rows(self, names: list[str]) -> None:
        for row in self._rows.values():
            row.md.remove()
            row.stop.remove()
        self._rows = {}

        gui = self.app.server.gui
        with self.running_folder:
            for name in names:
                md = gui.add_markdown("")
                stop = gui.add_button(f"Zatrzymaj {name}", icon=viser.Icon.PLAYER_STOP,
                                      hint="Zatrzymuje detektor; jego obiekty znikaja.")
                stop.on_click(self._stop_clicked(name))
                self._rows[name] = _Row(md, stop)

    def _run_line(self, name: str, runner: DetectorRunner | None, now: float) -> str:
        """The row's text; called under `_lock`."""
        if runner is not None:
            st = runner.status
            if st.starting or (name in self._launching and not st.running):
                return "**startuje...**"
            if st.running:
                return _running_line(runner, now)
        if name in self._launching:
            return "**startuje...**"
        if name in self._failed:
            return f"**nie ruszyl:** {self._failed[name]}"
        last = self._last.get(name)
        reason = last.status.stopped_because if last is not None else ""
        return f"zatrzymany: {reason or '-'}"

    def _tick_objects(self, views: list[TrackView]) -> None:
        self.objects_md.content = "\n".join(object_line(v) for v in views) if views else NO_OBJECTS

        shown = views if self.show_3d.value else []
        with self._rows_lock:
            for i, view in enumerate(shown):
                marker = self._marker(i, view)
                T = self.app.T_b2w @ pose(view.rot, view.pos)
                marker.frame.position, marker.frame.wxyz = T[:3, 3], mat_to_wxyz(T[:3, :3])
                marker.label.text = view.label
                marker.frame.visible = True
            for marker in self._markers[len(shown):]:
                marker.frame.visible = False

    def _marker(self, i: int, view: TrackView) -> _Marker:
        """The 3D marker number `i`, its shape made fit for `view`; reused by index."""
        scene = self.app.server.scene
        size = view.size_m if view.method == "sylwetka" and view.size_m is not None else None
        dims = (float(size[0]), float(size[1]), float(size[2])) if size is not None else None
        key = ("pudelko", dims) if dims is not None else ("kula",)
        base = f"/obiekty/{i}"

        if i == len(self._markers):
            frame = scene.add_frame(base, show_axes=False)
            label = scene.add_label(f"{base}/opis", "", position=(0.0, 0.0, 0.03))
            self._markers.append(_Marker(frame, None, label, ()))
        marker = self._markers[i]

        if marker.key != key:
            if marker.shape is not None:
                marker.shape.remove()
            if dims is not None:
                marker.shape = scene.add_box(f"{base}/ksztalt", color=FITTED_COLOR, dimensions=dims,
                                             opacity=0.5, wireframe=True)
            else:
                marker.shape = scene.add_icosphere(f"{base}/ksztalt", radius=POINT_RADIUS, color=POINT_COLOR)
            marker.key = key
        return marker


def _running_line(runner: DetectorRunner, now: float) -> str:
    st = runner.status
    line = f"**dziala** - wywolan {st.calls}"
    if st.calls:
        line += f", ostatnie {max(0.0, now - st.last_done):.1f} s temu (trwalo {st.last_latency_s:.1f} s)"
    line += f", widzi: {', '.join(st.last_labels) or 'nic'}"
    line += f", kamery: {', '.join(st.cameras) or 'brak'}"
    if st.dropped:
        line += f", odrzuconych wykryc {st.dropped}"
    if st.failures:
        line += f"  \n**blad:** {st.last_error} - kolejna proba za {max(0.0, st.next_try - now):.0f} s"
    return line


def object_line(view: TrackView) -> str:
    """One object as the tab lists it: position, yaw for a fitted one, source, cameras, confidence, age."""
    x, y, z = (100.0 * float(c) for c in view.pos[:3])
    line = f"- **{view.label}**: x {x:.1f} cm, y {y:.1f} cm, z {z:.1f} cm"
    if view.method == "sylwetka":
        yaw = float(np.degrees(np.arctan2(view.rot[1, 0], view.rot[0, 0])))
        line += f", obrot {yaw:.0f} st."
    line += (f", {view.source}, {view.n_cameras} kam. ({view.method}), pewnosc {view.confidence:.2f}, "
             f"wiek {view.age_s:.1f} s [{view.detector}]")
    return line
