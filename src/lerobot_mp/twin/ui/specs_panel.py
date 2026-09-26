"""Folder "Spectacles" panelu: kierowca `SpecsReach` (okulary -> przekaznik -> ramie).

    self.specs = add_specs_folder(self, g)             # w zakladce, gdzie ma byc folder
    self.specs.tick()                                  # z wolnego taktu panelu (1 Hz)

Samodzielny modul, zeby app.py zmienial sie o dwie linijki. Przycisk laczy blizniaka
panelu z przekaznikiem (`server.js`, sciezka /twin); ruchy ida przez te sama wlasnosc
ramienia co reszta panelu (wlasciciel "specs"), wiec uzbrojony panel = odmowa "busy".
"""

from __future__ import annotations

import logging
from typing import Any

import viser

from ..specsreach import MODES, SpecsReach

logger = logging.getLogger(__name__)

DEFAULT_RELAY = "ws://127.0.0.1:8080/twin"


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
            connect = gui.add_button("Polacz z okularami", icon=viser.Icon.PLUG_CONNECTED, color="green",
                                     hint="Laczy blizniaka z przekaznikiem: pudelko zasiegu w okularach, pinch = cel.")
            disconnect = gui.add_button("Rozlacz okulary", icon=viser.Icon.PLUG_CONNECTED_X, color="gray",
                                        hint="Konczy polaczenie; ruch w toku staje w zmierzonej pozie.")
            self.md = gui.add_markdown("Okulary: rozlaczone")

        @connect.on_click
        def _(event):
            try:
                self.start()
            except Exception as exc:                    # blad do napisu, nie do petli panelu
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

    def apply_mode(self) -> None:
        """Tryb i zgoda z panelu do kierowcy; ruch w toku staje (hold "mode")."""
        if self.driver is not None:
            self.driver.set_mode(self.mode.value, bool(self.real_ok.value))

    def start(self) -> None:
        self.stop(keep_consent=True)
        self.driver = SpecsReach(self.app.twin, relay_url=str(self.url.value).strip() or DEFAULT_RELAY,
                                 mode=self.mode.value, real_ok=bool(self.real_ok.value))
        self.driver.start()
        self.tick()

    def stop(self, *, keep_consent: bool = False) -> None:
        drv, self.driver = self.driver, None
        if drv is not None:
            drv.stop()
        # Zgoda na prawdziwe ramie nie przechodzi na nastepne polaczenie (poza ponownym "Polacz").
        if not keep_consent and self.real_ok.value:
            self.real_ok.value = False
        self.tick()

    def tick(self) -> None:
        """Napis stanu - wolany z wolnego taktu panelu."""
        drv = self.driver
        if drv is None:
            text = "Okulary: rozlaczone"
        else:
            st = drv.status()
            link = "polaczone" if st["connected"] else f"lacze z {drv.relay_url}..."
            session = st["session"] or "brak zaufanej rejestracji"
            text = f"Okulary: {link}, tryb **{st['mode']}**, sesja: {session}"
            last = st["last"]
            if last:
                pos = ", ".join(f"{v * 100:.0f}" for v in last.get("pos", []))
                why = f" ({last['reason']})" if last.get("reason") else ""
                text += f"  \nostatni cel [{pos}] cm: **{last['state']}**{why}"
        if self.md.content != text:
            self.md.content = text


def add_specs_folder(app: Any, gui: Any) -> SpecsFolder:
    """Folder "Spectacles" w biezacym kontekscie GUI (`gui` = `app.server.gui`)."""
    return SpecsFolder(app, gui)
