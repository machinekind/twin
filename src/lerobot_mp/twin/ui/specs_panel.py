"""Panel folder "Spectacles": drives `SpecsReach` (glasses -> relay -> arm).

    self.specs = add_specs_folder(self, g)             # inside the tab that should hold the folder
    self.specs.tick()                                  # from the panel's slow (1 Hz) tick

Kept as its own module so app.py only changes by a couple of lines. The status line shows the last target, the
last gripper command, the last follow stream (teleop) and grab, and "chwytak: brak sterowania przez XCAN" when the
backend can't drive the gripper (A1X over macOS XCAN). The button connects the panel's twin to the
relay (`server.js`, path /twin). Moves go through the same arm ownership as the rest of the panel (owner "specs"),
so while the panel is armed the glasses get a "busy" refusal.
"""

from __future__ import annotations

import os

import logging
from typing import Any

import viser

from ..specsreach import MODES, SpecsReach

logger = logging.getLogger(__name__)

# The relay usually runs on the machine with Lens Studio and the glasses, not on the rig's:
# SPECS_RELAY=ws://<that machine>:8080/twin pre-fills the field.
DEFAULT_RELAY = os.environ.get("SPECS_RELAY", "ws://127.0.0.1:8080/twin")


class SpecsFolder:
    def __init__(self, app: Any, gui: Any):
        self.app = app
        self.driver: SpecsReach | None = None
        with gui.add_folder("Spectacles", expand_by_default=False):
            self.url = gui.add_text("Przekaznik", DEFAULT_RELAY,
                                    hint="Adres przekaznika okularow (server.js), sciezka /twin.")
            self.mode = gui.add_dropdown("Tryb okularow", MODES, initial_value="dry",
                                         hint="dry - tylko sprawdzenia i nakladka, ramie stoi; sim - rusza ramieniem "
                                              "w symulacji; real - rusza prawdziwym ramieniem (wymaga zaznaczenia nizej).")
            self.real_ok = gui.add_checkbox("Okulary moga ruszac prawdziwym ramieniem", False,
                                            hint="Tylko tryb real. Obowiazuje, dopoki zaznaczone (inaczej niz "
                                                 "potwierdzenie polityki, ktore gasnie po starcie).")
            self.hold_off = gui.add_slider("Odczekanie przed ruchem [s]", min=0.0, max=3.0, step=0.1, initial_value=1.0,
                                           hint="Tryb real: ile sekund po przyjeciu celu z reki ramie czeka, zanim ruszy. "
                                                "0 = od razu. Dziala od nastepnego polaczenia z okularami.")
            self.follow_speed = gui.add_slider("Sledzenie dloni [m/s]", min=0.0, max=0.3, step=0.01, initial_value=0.0,
                                               hint="Tryb teleop w okularach: najwieksza predkosc punktu, za ktorym idzie "
                                                    "chwytak (przyspieszenie 4x tyle na sekunde). 0 = wedlug trybu: "
                                                    "0,20 w symulacji, 0,10 na prawdziwym ramieniu. Dziala od razu.")
            connect = gui.add_button("Polacz z okularami", icon=viser.Icon.PLUG_CONNECTED, color="green",
                                     hint="Laczy blizniaka z przekaznikiem: pudelko zasiegu w okularach, pinch = cel.")
            disconnect = gui.add_button("Rozlacz okulary", icon=viser.Icon.PLUG_CONNECTED_X, color="gray",
                                        hint="Konczy polaczenie; ruch w toku staje w zmierzonej pozie.")
            self.md = gui.add_markdown("Okulary: rozlaczone")

        @connect.on_click
        def _(event):
            try:
                self.start()
            except Exception as exc:                    # show it in the folder, do not break the panel loop
                logger.exception("Spectacles: nie polaczono")
                self.md.content = f"Okulary: blad - {exc}"

        @disconnect.on_click
        def _(event):
            self.stop()

        @self.mode.on_update
        def _(event):
            self.apply_mode()

        @self.real_ok.on_update
        def _(event):
            self.apply_mode()

        @self.follow_speed.on_update
        def _(event):
            if self.driver is not None:
                self.driver.set_follow_speed(self._follow_speed())

    def _follow_speed(self) -> float | None:
        """The slider's speed; None (0 on the slider) = the driver's default for the mode."""
        v = float(self.follow_speed.value)
        return v if v > 0 else None

    def apply_mode(self) -> None:
        """Push mode and consent to the driver; a move in progress stops (hold "mode")."""
        if self.driver is not None:
            self.driver.set_mode(self.mode.value, bool(self.real_ok.value))

    def start(self) -> None:
        self.stop(keep_consent=True)
        # Same trust rule as `lerobot-twin specs` without --trusted-only: any registration is enough.
        self.driver = SpecsReach(self.app.twin, relay_url=str(self.url.value).strip() or DEFAULT_RELAY,
                                 mode=self.mode.value, real_ok=bool(self.real_ok.value), allow_untrusted=True,
                                 hold_off_s=float(self.hold_off.value), follow_speed=self._follow_speed())
        self.driver.start()
        self.tick()

    def stop(self, *, keep_consent: bool = False) -> None:
        drv, self.driver = self.driver, None
        if drv is not None:
            drv.stop()
        # Real-arm consent must not carry over to the next connection (except a reconnect via "Polacz").
        if not keep_consent and self.real_ok.value:
            self.real_ok.value = False
        self.tick()

    def tick(self) -> None:
        drv = self.driver
        if drv is None:
            text = "Okulary: rozlaczone"
        else:
            st = drv.status()
            link = "polaczone" if st["connected"] else f"lacze z {drv.relay_url}..."
            session = st["session"] or "brak rejestracji"
            text = f"Okulary: {link}, tryb **{st['mode']}**, sesja: {session}"
            last = st["last"]
            if last:
                pos = ", ".join(f"{v * 100:.0f}" for v in last.get("pos", []))
                why = f" ({last['reason']})" if last.get("reason") else ""
                text += f"  \nostatni cel [{pos}] cm: **{last['state']}**{why}"
            grip = st.get("grip")
            if grip:
                why = f" ({grip['reason']})" if grip.get("reason") else ""
                text += f"  \nchwytak {grip['action']}: **{grip['state']}**{why}"
            follow = st.get("follow")
            if follow:
                text += f"  \nsledzenie {follow.get('stream')}: {_state_text(follow)}, {drv.follow_v_max:.2f} m/s"
            grab = st.get("grab")
            if grab:
                text += f"  \nchwyt: {_state_text(grab)}"
            if not st.get("gripper_drivable", True):
                text += "  \nchwytak: brak sterowania przez XCAN"
        if self.md.content != text:
            self.md.content = text


def _state_text(msg: dict) -> str:
    """`targetState` as "**state** (reason) - detail"."""
    text = f"**{msg['state']}**"
    if msg.get("reason"):
        text += f" ({msg['reason']})"
    if msg.get("detail"):
        text += f" - {msg['detail']}"
    return text


def add_specs_folder(app: Any, gui: Any) -> SpecsFolder:
    """Add the folder in the current GUI context (`gui` = `app.server.gui`)."""
    return SpecsFolder(app, gui)
