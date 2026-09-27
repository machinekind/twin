"""Zakladka "Mrowka": kalibracja mrowki RealAnt z kamerami i jazda WSAD polityka chodu.

    self.antp = AntTab(self, gui)                      # w zakladce Mrowka
    self.antp.tick(frames)                             # z taktu panelu
    self.antp.stop("STOP z panelu")                    # STOP panelu zatrzymuje tez mrowke

Osobny modul, zeby app.py zmienil sie o kilka linii (jak `specs_panel`). Kolejnosc pracy, od gory zakladki:

1. **Tagi**: numery (2 na grzbiecie, 1 na maszcie) i bok taga; **Ucz ukladu** - kadry, na ktorych kamery
   widza oba tagi naraz (poza kamer niepotrzebna, wystarczy K).
2. **Dokalibrowanie wzgledem stolu**: **Poziom z blatu** (mrowka lezy albo stoi na blacie: wysokosc
   i pochylenie taga wzgledem korpusu), **Kalibruj kierunek** (przejazd W i S: gdzie jest przod mrowki
   wzgledem taga). Przesuniecie taga na grzbiecie (x, y) - linijka, kilka mm, dla chodu bez znaczenia.
3. **Jazda**: mrowka symulowana (w scenie) albo prawdziwa (ZeroMQ do antproxy / ant_server z realant-hw),
   pulpit WSAD w zakladce i pod wlasnym adresem. Mrowka ma wlasna petle - Spectacles na ramieniu i WSAD na
   mrowce jada naraz; STOP u gory panelu zatrzymuje oba.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import viser

from ..ant import LEG_POSES, Ant, level_mount, locate, snap_yaw, square_layout
from ..tagbody import TagBody, learn_layout, observe

logger = logging.getLogger(__name__)

LEGS = {"lezy na brzuchu (serwa wylaczone)": "lezy", "stoi na palcach (start polityki)": "stoi"}
FRAMES = {"wzgledem widoku 3D (W = w glab ekranu)": "widok", "wzgledem mrowki (W = jej przod)": "mrowka"}
CAMERAS = {"tagi z kamer stanowiska": "tagi", "prawda z fizyki (tylko symulacja)": "prawda",
           "nominalna (bez kamery, jak robot_teleop)": "nominal"}
SOURCES = ("symulowana (w scenie)", "prawdziwa (ZeroMQ)")
PAD_PORT = 8091


class AntTab:
    def __init__(self, app: Any, g: Any):
        self.app = app
        self.ant: Ant = app.ws.ant_obj() or Ant()
        self._obs: list = []
        self.drive = None
        self.pad = None
        self._last_calib = None
        self._t_md = 0.0
        self._t_rebuild = -1e9
        g.add_markdown("Mrowka RealAnt z AprilTagami 36h11: **2** plasko na grzbiecie (uklad), **1** na maszcie. "
                       "Kamery stanowiska daja jej poze - sama nie wie, jak jest obrocona.")
        with g.add_folder("1. Tagi", expand_by_default=True):
            self.ids = g.add_text("id tagow", " ".join(map(str, self.ant.tags.ids)),
                                  hint="Numery AprilTag 36h11 po spacji. PIERWSZY lezy plasko na grzbiecie - to uklad tagow. "
                                       "Karta kalibracyjna ramienia ma 0 i 1: mrowki nie moze byc w kadrze podczas fali.")
            self.mm = g.add_number("Bok taga [mm]", round(self.ant.tags.tag_size * 1000, 1), min=5.0, max=150.0,
                                   step=0.5, hint="Bok CZARNEGO kwadratu, zmierzony linijka na wydruku.")
            learn = g.add_button("Ucz ukladu: dodaj kadr", icon=viser.Icon.CAMERA_PLUS,
                                 hint="Biezace kadry kamer z K: pary tagow widziane naraz -> gdzie jest tag na maszcie "
                                      "wzgledem taga na grzbiecie. Kilka kadrow z roznych stron.")
            forget = g.add_button("Nauka od nowa", icon=viser.Icon.TRASH, color="gray",
                                  hint="Wyrzuca zebrane kadry i uklad (np. po przeklejeniu taga).")
            self.square = g.add_checkbox("Maszt prostopadle do grzbietu", True,
                                         hint="Tag na maszcie stoi prostopadle do taga na grzbiecie, krawedzie rownolegle: "
                                              "wyuczony obrot przyciagany do najblizszego prostopadlego (do 15 st.); "
                                              "odchylka w opisie ponizej to sprawdzian naklejenia.")
        with g.add_folder("2. Wzgledem stolu", expand_by_default=True):
            self.legs = g.add_dropdown("Mrowka teraz", tuple(LEGS), initial_value=self._legs_label(),
                                       hint="Jak stoi przy Poziom z blatu - od tego wysokosc korpusu nad blatem "
                                            "(na brzuchu 24 mm, na palcach 78 mm).")
            level = g.add_button("Poziom z blatu", icon=viser.Icon.RULER_MEASURE,
                                 hint="Mrowka na blacie: wysokosc taga nad srodkiem korpusu i jego pochylenie z kamer.")
            self.mx = g.add_number("Tag na grzbiecie: x [mm]", round(self.ant.mount["x"] * 1000, 1), step=0.5,
                                   hint="Przesuniecie srodka taga od srodka korpusu do przodu mrowki (linijka).")
            self.my = g.add_number("Tag na grzbiecie: y [mm]", round(self.ant.mount["y"] * 1000, 1), step=0.5,
                                   hint="Przesuniecie w lewo mrowki (linijka).")
            self.myaw = g.add_number("Obrot taga wzgledem przodu [st.]", round(self.ant.mount["yaw"], 1), step=0.5,
                                     hint="Osie taga wzdluz nog = 45 st. (+ k * 90) od przodu mrowki (przod miedzy "
                                          "nogami 2 i 3). Kalibruj kierunek wybiera, ktora noga; mozna poprawic recznie.")
            self.snap = g.add_checkbox("Tag wzdluz nog (45 + k * 90 st.)", True,
                                       hint="Kalibracja kierunku przyciaga wynik do 45 + k * 90 st. (krawedzie taga "
                                            "rownolegle do plyty korpusu); przejazd W/S ma kilka stopni dryfu.")
            calib = g.add_button("Kalibruj kierunek (jazda W i S)", icon=viser.Icon.ARROWS_MOVE_VERTICAL,
                                 color="orange",
                                 hint="RUSZA mrowka: ~3 s do przodu, ~3 s do tylu (polityka z kamera nominalna), "
                                      "kamery mierza kierunek przejazdu -> gdzie jest przod wzgledem taga. Wymaga "
                                      "uruchomionej mrowki (ponizej) i miejsca 30 cm przed i za nia.")
            find = g.add_button("Znajdz mrowke w kamerach", icon=viser.Icon.TARGET,
                                hint="Poza mrowki z kamer z poza -> mrowka w 3D. Mrowka nie rusza.")
            self.follow = g.add_checkbox("Sledz na zywo", False, hint="Poza z kamer w kazdym takcie panelu.")
            save = g.add_button("Zapisz do stanowiska", icon=viser.Icon.DEVICE_FLOPPY, color="green",
                                hint="Tagi, uklad, montaz i ostatnia poza do twin.json (sekcja ant).")
            self.calib_md = g.add_markdown("")
        with g.add_folder("3. Jazda (WSAD)", expand_by_default=True):
            self.source = g.add_dropdown("Mrowka", SOURCES, initial_value=SOURCES[0 if self.ant.simulated else 1],
                                         hint="Symulowana chodzi w fizyce sceny (demo z mrowka); prawdziwa - rozkazy do "
                                              "antproxy.py / ant_server.py (realant-hw) po ZeroMQ.")
            self.host = g.add_text("Adres antproxy", "127.0.0.1",
                                   hint="Maszyna z antproxy.py i ant_server.py (porty 3001 i 3002).")
            self.camera = g.add_dropdown("Poza dla polityki", tuple(CAMERAS), initial_value=tuple(CAMERAS)[0],
                                         hint="Tagi - jak na prawdziwej mrowce; prawda - porownanie w symulacji; "
                                              "nominalna - idealny przejazd bez kamery (robot_teleop.py).")
            self.frame = g.add_dropdown("Kierunki WSAD", tuple(FRAMES), initial_value=tuple(FRAMES)[0],
                                        hint="Widok: W jedzie w glab widoku 3D panelu, A / D w lewo / prawo ekranu - "
                                             "mrowka wybiera ten z czterech kierunkow (przemianowanie nog), ktory "
                                             "lezy najblizej. Mrowka: W = przod mrowki z realant-rl.")
            run = g.add_button("Uruchom mrowke", icon=viser.Icon.PLAYER_PLAY, color="green",
                               hint="Petla mrowki 20 Hz i pulpit WSAD. Mrowka wstaje do pozy startowej polityki.")
            halt = g.add_button("Zatrzymaj mrowke", icon=viser.Icon.PLAYER_STOP, color="gray",
                                hint="Konczy petle i pulpit; prawdziwa mrowka zostaje w ostatniej pozie (serwa trzymaja).")
            unstop = g.add_button("Skasuj STOP mrowki", icon=viser.Icon.HAND_STOP, color="yellow",
                                  hint="Po STOP-ie (panelu, przewroceniu, bledzie) mrowka stoi i ignoruje klawisze, "
                                       "dopoki go tu nie skasujesz.")
            self.pad_html = g.add_html("")
            self.drive_md = g.add_markdown("Mrowka: niewlaczona")

        for handle in (self.mx, self.my, self.myaw):
            @handle.on_update
            def _(event):
                self.ant.mount = {**self.ant.mount, "x": float(self.mx.value) / 1000.0,
                                  "y": float(self.my.value) / 1000.0, "yaw": float(self.myaw.value)}

        @self.legs.on_update
        def _(event):
            self.ant.joints = dict(LEG_POSES[LEGS[self.legs.value]])

        @self.frame.on_update
        def _(event):
            if self.drive is not None:
                self.drive.frame = FRAMES[self.frame.value]

        @self.camera.on_update
        def _(event):
            if self.drive is not None and not self.drive.calibrating:
                self.drive.camera = CAMERAS[self.camera.value]

        self._button(learn, self.learn)
        self._button(forget, self.forget)
        self._button(level, self.level)
        self._button(calib, self.calibrate)
        self._button(find, lambda: self.find(self.app._grab()))
        self._button(save, self.save)
        self._button(run, self.start)
        self._button(halt, self.halt)
        self._button(unstop, self.clear)
        self._note("")

    # ---------------------------------------------------------------- pomocnicze
    def _button(self, handle, fn) -> None:
        @handle.on_click
        def _(event):
            try:
                fn()
            except Exception as exc:                     # do uzytkownika, nie do watku visera
                logger.exception("mrowka")
                self.app._notify(event, "Mrowka", str(exc), error=True)

    def _legs_label(self) -> str:
        for label, key in LEGS.items():
            if dict(LEG_POSES[key]) == dict(self.ant.joints):
                return label
        return tuple(LEGS)[0]

    def _sync_tags(self) -> TagBody:
        """Numery i bok z pol: nowe numery zeruja uklad, nowy bok skaluje uklad (jak kamera PMD)."""
        try:
            ids = tuple(int(t) for t in self.ids.value.replace(",", " ").split())
        except ValueError:
            raise ValueError(f"id tagow: liczby po spacji, jest {self.ids.value!r}") from None
        if not ids or len(set(ids)) != len(ids):
            raise ValueError("id tagow: co najmniej jeden numer, bez powtorzen")
        size = float(self.mm.value) / 1000.0
        body = self.ant.tags
        if ids != body.ids:
            body = TagBody("mrowka", ids, size)
            self._obs = []
        elif abs(size - body.tag_size) > 1e-9:
            layout = {i: T.copy() for i, T in body.layout.items()}
            for T in layout.values():
                T[:3, 3] *= size / body.tag_size
            body = TagBody("mrowka", ids, size, layout)
            self._obs = []
        self.ant.tags = body
        return body

    def _note(self, text: str) -> None:
        shared = sorted(set(self.ant.tags.ids) & set(self.app.ws.card_obj().ids))
        if shared:
            text += (f"\n\n**Uwaga:** tag {', '.join(map(str, shared))} jest tez na karcie kalibracyjnej ramienia - "
                     f"mrowka nie moze byc w kadrze podczas fali (albo inny numer na maszcie).")
        known = self.ant.tags.known()
        m = self.ant.mount
        head = (f"Uklad: {len(known)}/{len(self.ant.tags.ids)} tagow; tag na grzbiecie {m['z'] * 1000:.1f} mm nad "
                f"srodkiem korpusu, pochylony {m['pitch']:+.1f} / {m['roll']:+.1f} st., obrot {m['yaw']:+.1f} st.")
        self.calib_md.content = head + ("\n\n" + text if text else "")

    def _publish(self, rebuild: bool = False) -> None:
        """Mrowka z zakladki do stanowiska w pamieci (scena przebudowuje sie z niego); bez zapisu na dysk."""
        before = self.app.ws.ant
        self.app.ws.ant = self.ant.to_dict()
        if rebuild or before is None or self.app.twin.scene.cfg.ant is None:
            self.app._dirty_rebuild = time.monotonic()

    # ---------------------------------------------------------------- kalibracja
    def learn(self) -> None:
        body = self._sync_tags()
        obs = observe(self.app._grab(), self.app.ws.cameras, body)
        if not obs:
            self._note(f"zadna kamera z K nie widzi tagow {', '.join(map(str, body.ids))}")
            return
        self._obs.extend(obs)
        fit = learn_layout(self._obs, body)
        body.layout = fit.layout
        squared = ""
        if self.square.value:
            others = {i: T for i, T in fit.layout.items() if i != body.ref}
            snapped, offs = square_layout(others)
            body.layout = {body.ref: np.eye(4), **snapped}
            squared = "".join(f" Tag {i}: przyciagniety do prostopadlego o {off:.1f} st." if off <= 15.0 else
                              f" **Tag {i}: {off:.0f} st. od prostopadlego - zostawiony, jak wyszedl.**"
                              for i, off in offs.items())
        pairs = ", ".join(f"{a}-{b} x{n}" for (a, b), n in sorted(fit.pairs.items()))
        missing = fit.missing(body.ids)
        seen = "; ".join(f"{o.camera}: {', '.join(map(str, o.ids))}" for o in obs)
        text = f"kadr {len(self._obs)}: {seen}. Pary naraz: {pairs or 'brak'}. "
        text += (f"**Bez pozy: tag {', '.join(map(str, missing))}** - pokaz go kamerze razem z tagiem na grzbiecie."
                 if missing else "Uklad kompletny - teraz Poziom z blatu.") + squared
        self._note(text)
        self._publish(rebuild=True)

    def forget(self) -> None:
        body = self._sync_tags()
        body.layout = {body.ref: np.eye(4)}
        self._obs = []
        self._note("nauka ukladu od nowa - kadry z oboma tagami naraz")

    def _fix(self, frames=None):
        self._sync_tags()
        fix = locate(frames if frames is not None else self.app._grab(), self.app.ws.cameras, self.ant)
        if fix is None:
            posed = [c.name for c in self.app.ws.cameras if c.enabled and c.T_cam2base is not None]
            raise RuntimeError(f"tagi {', '.join(map(str, self.ant.tags.known()))} nie widziane przez kamery z poza "
                               f"({', '.join(posed) or 'brak - najpierw Kalibracja'})")
        return fix

    def level(self) -> None:
        fix = self._fix()
        self.ant.joints = dict(LEG_POSES[LEGS[self.legs.value]])
        self.ant.mount = level_mount(fix.tags.T_body2base, self.ant)
        self._note(f"poziom z blatu: {len(fix.tags.per_camera)} kamer, rozrzut {fix.tags.spread_mm:.1f} mm")
        self._publish(rebuild=True)

    def find(self, frames) -> None:
        fix = self._fix(frames)
        self._show_pose(fix.T_ant2base)
        p = fix.position
        cams = "; ".join(f"{n}: tagi {', '.join(map(str, fix.tags.tags_seen[n]))}, {r:.2f} px"
                         for n, r in fix.tags.reproj_px.items())
        above = p[2] - self.ant.rest_height()
        self._note(f"mrowka: x {p[0] * 1000:.0f} mm, y {p[1] * 1000:.0f} mm, kurs {fix.heading_deg:+.1f} st., "
                   f"pochylenie {fix.tilt_deg:.1f} st., {above * 1000:+.0f} mm wzgledem pozy "
                   f"'{self.legs.value}'; {cams}; rozrzut kamer {fix.tags.spread_mm:.1f} mm")

    def _show_pose(self, T: np.ndarray) -> None:
        """Poza z kamer -> mrowka w scenie. Symulowana chodzi w fizyce - jej nie przestawiamy."""
        self.ant.T_ant2base = np.asarray(T, float)
        if self.ant.simulated:
            return
        if not self.app.twin.place_ant(T) and time.monotonic() - self._t_rebuild > 3.0:
            # Scena jeszcze bez mrowki - jedna przebudowa (panel czeka na 1 s spokoju, wiec nie co takt).
            self._t_rebuild = time.monotonic()
            self._publish(rebuild=True)

    def calibrate(self) -> None:
        if self.drive is None:
            raise RuntimeError("najpierw Uruchom mrowke (3. Jazda) - kalibracja kierunku jedzie mrowka")
        if len(self.ant.tags.known()) < 1:
            raise RuntimeError("najpierw uklad tagow (1. Tagi)")
        self.drive.calibrate_heading(3.0)
        self._last_calib = None
        self._note("kalibracja kierunku: mrowka jedzie ~3 s do przodu i ~3 s do tylu...")

    def save(self) -> None:
        self._sync_tags()
        self._publish(rebuild=True)
        self.app._save()
        self._note(f"zapisano do {self.app.ws_path.name if self.app.ws_path else 'twin.json'} (sekcja ant)")

    # ---------------------------------------------------------------- jazda
    def start(self) -> None:
        from ..antdrive import AntDrive, RealAnt, SimAnt
        from ..antpad import AntPad
        from ..antwalk import ZmqAnt, ZmqAntConfig

        if self.drive is not None:
            raise RuntimeError("mrowka juz uruchomiona")
        self._sync_tags()
        if self.source.value == SOURCES[0]:
            if not self.ant.simulated:
                raise RuntimeError("stanowisko nie ma mrowki symulowanej (lerobot-twin demo --ant) - wybierz prawdziwa")
            io = SimAnt(self.app.twin)
        else:
            try:
                io = RealAnt(ZmqAnt(ZmqAntConfig(host=str(self.host.value).strip() or "127.0.0.1")))
            except ImportError:
                raise RuntimeError("brak pyzmq: pip install -e .[ant]") from None
        drive = AntDrive(self.app.twin, lambda: self.ant, io=io, camera=CAMERAS[self.camera.value],
                         on_pose=self._on_drive_pose, frame=FRAMES[self.frame.value], view_dir=self._view_dir)
        if self.app._estop_active():
            drive.stop("STOP panelu aktywny - Skasuj STOP")
        drive.start()
        try:
            pad = AntPad(drive, port=PAD_PORT, host=self.app.host).start()
        except OSError:
            pad = AntPad(drive, port=0, host=self.app.host).start()     # zajety port - dowolny wolny
        self.drive, self.pad = drive, pad
        self.pad_html.content = pad.html_embed()
        self.drive_md.content = f"Mrowka: startuje; pulpit WSAD takze pod http://{self.app.host}:{pad.port}/"

    def _view_dir(self) -> np.ndarray | None:
        """Kierunek patrzenia kamery 3D panelu na blacie, w ukladzie podstawy (najnowszy klient)."""
        clients = list(self.app.server.get_clients().values())
        if not clients:
            return None
        cam = clients[-1].camera
        d = np.asarray(cam.look_at, float) - np.asarray(cam.position, float)
        d = self.app.T_b2w[:3, :3].T @ d
        n = float(np.hypot(d[0], d[1]))
        return None if n < 1e-6 else d[:2] / n

    def _on_drive_pose(self, T: np.ndarray) -> None:
        self._show_pose(T)

    def halt(self) -> None:
        drive, pad = self.drive, self.pad
        self.drive, self.pad = None, None
        self.pad_html.content = ""
        if pad is not None:
            pad.close()
        if drive is not None:
            drive.close()
        self.drive_md.content = "Mrowka: niewlaczona"

    def stop(self, reason: str) -> None:
        if self.drive is not None:
            self.drive.stop(reason)

    def clear(self) -> None:
        if self.drive is not None:
            self.drive.clear()

    def close(self) -> None:
        self.halt()

    # ---------------------------------------------------------------- takt
    def tick(self, frames: dict[str, np.ndarray]) -> None:
        if self.follow.value and self.drive is None:
            try:
                self.find(frames)
            except RuntimeError as exc:
                self._note(str(exc))
        d = self.drive
        if d is None:
            return
        res = d.last_calibration
        if res is not None and res is not self._last_calib:
            self._last_calib = res
            if res.get("error"):
                self._note(f"kalibracja kierunku: **{res['error']}**")
            else:
                raw = float(res["yaw_deg"])
                yaw, off = snap_yaw(raw) if self.snap.value else (raw, 0.0)
                self.ant.mount = {**self.ant.mount, "yaw": yaw}
                self.myaw.value = round(yaw, 1)
                how = (f"przejazd {raw:+.1f} st., przyciagniety do {yaw:+.0f} (odchylka {off:+.1f})" if yaw != raw
                       else f"{raw:+.1f} st." + (" - **daleko od 45 + k * 90, bez przyciagania**" if self.snap.value
                                                  else ""))
                self._note(f"kalibracja kierunku: obrot taga {how}; przod / tyl zgodne co do {res['spread_deg']:.1f} st., "
                           f"przejazdy {', '.join(f'{x:.2f}' for x in res['dists'])} m. Zapisz do stanowiska, zeby zostalo.")
                self._publish()
        now = time.monotonic()
        if now - self._t_md < 0.3:
            return
        self._t_md = now
        st = d.status
        lines = [f"**Mrowka: {st.mode}**" + (f" - {st.direction}" if st.direction else "")
                 + (f", klawisze {st.keys}" if st.keys else "")]
        if st.note:
            lines.append(st.note)
        if st.T_ant2base is not None:
            p = st.T_ant2base[:3, 3]
            age = "" if st.pose_age is None else f", poza sprzed {st.pose_age:.2f} s"
            lines.append(f"x {p[0] * 1000:.0f} mm, y {p[1] * 1000:.0f} mm, kurs {st.heading_deg:+.0f} st.{age}")
        if st.height:
            lines.append(f"wysokosc: {st.height}")
        lines.append(f"poza dla polityki: {d.camera}; polityka {d.policy.name}")
        self.drive_md.content = "  \n".join(lines)
