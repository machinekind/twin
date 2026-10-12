"""Zakladka "Mrowka": kopia mrowki RealAnt w bliznaku - tagi, samokalibracja wzgledem stolu, sledzenie; demo WSAD.

    self.antp = AntTab(self, gui)                      # w zakladce Mrowka
    self.antp.tick(frames)                             # z taktu panelu
    self.antp.stop("STOP z panelu")                    # STOP panelu zatrzymuje mrowke udajaca prawdziwa (demo)

Osobny modul, zeby app.py zmienil sie o kilka linii (jak `specs_panel`). Prawdziwa mrowka jezdzi na wlasnej
polityce (realant-rl `robot_teleop.py` po jej stronie) - blizniak niczego jej nie wysyla. Kolejnosc pracy:

1. **Tagi**: numery (2 na grzbiecie, 1 na maszcie) i bok taga; **Ucz ukladu** - kadry z oboma tagami naraz.
2. **Wzgledem stolu**: kalibruje sie samo w trakcie sledzenia (kierunek przodu z porownania ruchu kopii na
   odczytach serw z ruchem z tagow; wysokosc i pochylenie taga z korpusu kopii stojacego na nogach). Recznie:
   **Poziom z blatu** i pola montazu.
3. **Sledzenie**: kopia w fizyce sceny - nogi z odczytow serw (ZeroMQ, tylko nasluch ant_server), poza dociagana
   tagami. W demo odczyty serw daje mrowka udajaca prawdziwa.
4. **Demo**: mrowka udajaca prawdziwa jezdzi na polityce chodu z WSAD (jej `robot_teleop.py`), pulpit w zakladce
   i pod wlasnym adresem. Wlasna petla - Spectacles na ramieniu i WSAD naraz.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import numpy as np
import viser

from ..ant import LEG_POSES, Ant, level_mount, locate, square_layout
from ..tagbody import TagBody, learn_layout, observe

logger = logging.getLogger(__name__)

LEGS = {"lezy na brzuchu (serwa wylaczone)": "lezy", "stoi na palcach (start polityki)": "stoi"}
FRAMES = {"wzgledem widoku 3D (W = w glab ekranu)": "widok", "wzgledem mrowki (W = jej przod)": "mrowka"}
CAMERAS = {"nominalna (bez kamery, jak robot_teleop)": "nominal", "tagi z kamer stanowiska": "tagi",
           "prawda z fizyki": "prawda"}
SOURCES = ("prawdziwa: odczyty serw z ant_server (ZeroMQ)", "demo: mrowka udajaca prawdziwa")
PAD_PORT = 8091


class AntTab:
    def __init__(self, app: Any, g: Any):
        self.app = app
        self.ant: Ant = app.ws.ant_obj() or Ant()
        self._obs: list = []
        self.mirror = None
        self.drive = None
        self.pad = None
        self._t_md = 0.0
        self._t_rebuild = -1e9
        self._mount_shown: dict[str, float] = {}
        g.add_markdown("Mrowka RealAnt z AprilTagami 36h11: **2** plasko na grzbiecie, **1** na maszcie. Mrowka "
                       "jezdzi na wlasnej polityce; blizniak ma jej kopie - nogi z odczytow serw, poza z tagow.")
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
            self.auto_yaw = g.add_checkbox("Sam: kierunek przodu", True,
                                           hint="W trakcie sledzenia: dokad kopia idzie na odczytach serw vs dokad mrowka "
                                                "wedlug tagow -> obrot taga wzgledem przodu mrowki.")
            self.snap = g.add_checkbox("Tag wzdluz nog (45 + k * 90 st.)", True,
                                       hint="Krawedzie taga rownolegle do plyty korpusu: obrot przyciagany do "
                                            "45 + k * 90 st.; wtedy odcinki ruchu wybieraja tylko, ktora noga.")
            self.auto_level = g.add_checkbox("Sam: wysokosc i pochylenie taga", True,
                                             hint="W trakcie sledzenia: tag wzgledem korpusu kopii, ktora stoi na nogach "
                                                  "z odczytow serw (wysokosc i pochylenie z fizyki).")
            with g.add_folder("Recznie", expand_by_default=False):
                self.legs = g.add_dropdown("Mrowka teraz", tuple(LEGS), initial_value=self._legs_label(),
                                           hint="Jak stoi przy Poziom z blatu - od tego wysokosc korpusu nad blatem "
                                                "(na brzuchu 24 mm, na palcach 78 mm).")
                level = g.add_button("Poziom z blatu", icon=viser.Icon.RULER_MEASURE,
                                     hint="Mrowka stoi na blacie: wysokosc taga nad srodkiem korpusu i jego pochylenie.")
                self.mx = g.add_number("Tag na grzbiecie: x [mm]", round(self.ant.mount["x"] * 1000, 1), step=0.5,
                                       hint="Przesuniecie srodka taga od srodka korpusu do przodu mrowki (linijka).")
                self.my = g.add_number("Tag na grzbiecie: y [mm]", round(self.ant.mount["y"] * 1000, 1), step=0.5,
                                       hint="Przesuniecie w lewo mrowki (linijka).")
                self.myaw = g.add_number("Obrot taga wzgledem przodu [st.]", round(self.ant.mount["yaw"], 1), step=0.5,
                                         hint="Osie taga wzdluz nog = 45 st. (+ k * 90) od przodu (przod miedzy nogami "
                                              "2 i 3). Sledzenie przestawia go samo.")
            save = g.add_button("Zapisz do stanowiska", icon=viser.Icon.DEVICE_FLOPPY, color="green",
                                hint="Tagi, uklad, montaz i ostatnia poza do twin.json (sekcja ant).")
            self.calib_md = g.add_markdown("")
        with g.add_folder("3. Sledzenie (kopia w bliznaku)", expand_by_default=True):
            self.source = g.add_dropdown("Odczyty serw", SOURCES, initial_value=SOURCES[1 if self.ant.simulated else 0],
                                         hint="Prawdziwa: nasluch pomiarow ant_server.py przez antproxy.py (port 3001) - "
                                              "blizniak nic mrowce nie wysyla. Demo: mrowka udajaca prawdziwa w scenie.")
            self.host = g.add_text("Adres antproxy", "127.0.0.1", hint="Maszyna z antproxy.py (port 3001).")
            track = g.add_button("Sledz mrowke", icon=viser.Icon.PLAYER_PLAY, color="green",
                                 hint="Kopia w fizyce: nogi z odczytow serw, poza dociagana tagami, montaz kalibruje sie sam.")
            untrack = g.add_button("Zatrzymaj sledzenie", icon=viser.Icon.PLAYER_STOP, color="gray")
            self.mirror_md = g.add_markdown("Kopia: nie sledzi")
        self.demo_box = g.add_folder("4. Demo: mrowka udajaca prawdziwa (WSAD)", expand_by_default=True,
                                     visible=self.ant.simulated)
        with self.demo_box:
            g.add_markdown("Tu jezdzi mrowka udajaca prawdziwa - jej wlasna polityka chodu (jak `robot_teleop.py` na "
                           "prawdziwej). Kopia w bliznaku (niebieska) sledzi ja tylko z tagow i odczytow serw.")
            self.frame = g.add_dropdown("Kierunki WSAD", tuple(FRAMES), initial_value=tuple(FRAMES)[0],
                                        hint="Widok: W jedzie w glab widoku 3D panelu, A / D w lewo / prawo ekranu - "
                                             "wybierany najblizszy z czterech kierunkow mrowki. Mrowka: W = jej przod.")
            self.camera = g.add_dropdown("Poza dla jej polityki", tuple(CAMERAS), initial_value=tuple(CAMERAS)[0],
                                         hint="Nominalna - jak prawdziwa mrowka z robot_teleop.py (bez kamery); tagi - "
                                              "tryb live z robot_policy.py; prawda - z fizyki.")
            run = g.add_button("Uruchom mrowke udajaca prawdziwa", icon=viser.Icon.PLAYER_PLAY, color="green",
                               hint="Jej petla 20 Hz i pulpit WSAD. Wstaje do pozy startowej polityki.")
            halt = g.add_button("Zatrzymaj ja", icon=viser.Icon.PLAYER_STOP, color="gray")
            unstop = g.add_button("Skasuj STOP mrowki", icon=viser.Icon.HAND_STOP, color="yellow",
                                  hint="Po STOP-ie (panelu, przewroceniu) mrowka stoi i ignoruje klawisze, dopoki go "
                                       "tu nie skasujesz.")
            self.pad_html = g.add_html("")
            self.drive_md = g.add_markdown("Mrowka udajaca prawdziwa: niewlaczona")

        for handle in (self.mx, self.my, self.myaw):
            @handle.on_update
            def _(event):
                if event.client is None:
                    return                                # z kodu (samokalibracja) - montaz juz ustawiony
                self.ant.mount = {**self.ant.mount, "x": float(self.mx.value) / 1000.0,
                                  "y": float(self.my.value) / 1000.0, "yaw": float(self.myaw.value)}

        @self.legs.on_update
        def _(event):
            self.ant.joints = dict(LEG_POSES[LEGS[self.legs.value]])

        for handle in (self.auto_yaw, self.snap, self.auto_level):
            @handle.on_update
            def _(event):
                if self.mirror is not None:
                    self.mirror.auto_yaw, self.mirror.snap = bool(self.auto_yaw.value), bool(self.snap.value)
                    self.mirror.auto_level = bool(self.auto_level.value)

        @self.frame.on_update
        def _(event):
            if self.drive is not None:
                self.drive.frame = FRAMES[self.frame.value]

        @self.camera.on_update
        def _(event):
            if self.drive is not None:
                self.drive.camera = CAMERAS[self.camera.value]

        self._button(learn, self.learn)
        self._button(forget, self.forget)
        self._button(level, self.level)
        self._button(save, self.save)
        self._button(track, self.start_tracking)
        self._button(untrack, self.stop_tracking)
        self._button(run, self.start_standin)
        self._button(halt, self.halt_standin)
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
        if rebuild or before is None or "ant/" not in self.app.twin.scene.ants():
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
                 if missing else "Uklad kompletny - teraz Sledz mrowke.") + squared
        self._note(text)
        self._publish(rebuild=True)

    def forget(self) -> None:
        body = self._sync_tags()
        body.layout = {body.ref: np.eye(4)}
        self._obs = []
        self._note("nauka ukladu od nowa - kadry z oboma tagami naraz")

    def level(self) -> None:
        self._sync_tags()
        fix = locate(self.app._grab(), self.app.ws.cameras, self.ant)
        if fix is None:
            raise RuntimeError("tagi nie widziane przez kamery z poza")
        self.ant.joints = dict(LEG_POSES[LEGS[self.legs.value]])
        self.ant.mount = level_mount(fix.tags.T_body2base, self.ant)
        self._note(f"poziom z blatu: {len(fix.tags.per_camera)} kamer, rozrzut {fix.tags.spread_mm:.1f} mm")
        self._publish()

    def save(self) -> None:
        self._sync_tags()
        self._publish(rebuild=True)
        self.app._save()
        self._note(f"zapisano do {self.app.ws_path.name if self.app.ws_path else 'twin.json'} (sekcja ant)")

    # ---------------------------------------------------------------- sledzenie
    def start_tracking(self) -> None:
        from ..antmirror import AntMirror, StandInServos
        from ..antwalk import ZmqAnt, ZmqAntConfig

        if self.mirror is not None:
            raise RuntimeError("sledzenie juz dziala")
        self._sync_tags()
        if self.source.value == SOURCES[1]:
            if not self.ant.simulated:
                raise RuntimeError("stanowisko bez mrowki udajacej prawdziwa (lerobot-twin demo --ant)")
            servos = StandInServos(self.app.twin)
        else:
            try:
                servos = ZmqAnt(ZmqAntConfig(host=str(self.host.value).strip() or "127.0.0.1"))
            except ImportError:
                raise RuntimeError("brak pyzmq: pip install -e .[ant]") from None
        self.mirror = AntMirror(self.app.twin, lambda: self.ant, servos, auto_yaw=bool(self.auto_yaw.value),
                                snap=bool(self.snap.value), auto_level=bool(self.auto_level.value),
                                on_missing=self._on_missing)
        self.mirror.start()
        self.mirror_md.content = "Kopia: startuje"

    def _on_missing(self, T: np.ndarray) -> None:
        """Pierwsza poza z tagow, a w scenie nie ma jeszcze kopii - stanowisko dostaje poze, scena sie przebudowuje."""
        if time.monotonic() - self._t_rebuild > 3.0:
            self._t_rebuild = time.monotonic()
            self.ant.T_ant2base = np.asarray(T, float)
            self._publish(rebuild=True)

    def stop_tracking(self) -> None:
        m, self.mirror = self.mirror, None
        if m is not None:
            m.close()
        self.mirror_md.content = "Kopia: nie sledzi"

    # ---------------------------------------------------------------- demo: mrowka udajaca prawdziwa
    def start_standin(self) -> None:
        from ..antdrive import AntDrive, SimAnt
        from ..antpad import AntPad

        if self.drive is not None:
            raise RuntimeError("mrowka udajaca prawdziwa juz jezdzi")
        if not self.ant.simulated:
            raise RuntimeError("to tylko demo (lerobot-twin demo --ant) - prawdziwa mrowka jezdzi na robot_teleop.py")
        self._sync_tags()
        drive = AntDrive(self.app.twin, lambda: self.ant, io=SimAnt(self.app.twin), camera=CAMERAS[self.camera.value],
                         frame=FRAMES[self.frame.value], view_dir=self._view_dir)
        if self.app._estop_active():
            drive.stop("STOP panelu aktywny - Skasuj STOP mrowki")
        drive.start()
        try:
            pad = AntPad(drive, port=PAD_PORT, host=self.app.host).start()
        except OSError:
            pad = AntPad(drive, port=0, host=self.app.host).start()     # zajety port - dowolny wolny
        self.drive, self.pad = drive, pad
        self.pad_html.content = pad.html_embed()
        self.drive_md.content = f"Mrowka udajaca prawdziwa: startuje; pulpit takze pod http://{self.app.host}:{pad.port}/"

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

    def halt_standin(self) -> None:
        drive, pad = self.drive, self.pad
        self.drive, self.pad = None, None
        self.pad_html.content = ""
        if pad is not None:
            pad.close()
        if drive is not None:
            drive.close()
        self.drive_md.content = "Mrowka udajaca prawdziwa: niewlaczona"

    def stop(self, reason: str) -> None:
        if self.drive is not None:
            self.drive.stop(reason)

    def clear(self) -> None:
        if self.drive is not None:
            self.drive.clear()

    def close(self) -> None:
        self.halt_standin()
        self.stop_tracking()

    # ---------------------------------------------------------------- takt
    def tick(self, frames: dict[str, np.ndarray]) -> None:
        now = time.monotonic()
        if now - self._t_md < 0.3:
            return
        self._t_md = now
        m = self.mirror
        if m is not None:
            st = m.status
            lines = [f"**Kopia: {st.mode}**" + (f" - {st.note}" if st.note else "")]
            if st.T_ant2base is not None:
                p = st.T_ant2base[:3, 3]
                heading = float(np.degrees(np.arctan2(st.T_ant2base[1, 0], st.T_ant2base[0, 0])))
                lines.append(f"x {p[0] * 1000:.0f} mm, y {p[1] * 1000:.0f} mm, kurs {heading:+.0f} st.")
            ages = []
            if st.servo_age is not None:
                ages.append(f"serwa sprzed {st.servo_age:.2f} s")
            if st.tag_age is not None:
                ages.append(f"tagi sprzed {st.tag_age:.2f} s")
            if st.err_mm is not None:
                ages.append(f"ostatnia poprawka z tagow {st.err_mm:.0f} mm / {st.err_deg:+.1f} st.")
            if ages:
                lines.append(", ".join(ages))
            if st.calib:
                lines.append(st.calib)
            self.mirror_md.content = "  \n".join(lines)
            mount = {k: round(v, 4) for k, v in self.ant.mount.items()}
            if mount != self._mount_shown:                # samokalibracja przestawila montaz - pola i opis
                self._mount_shown = mount
                self.myaw.value = round(self.ant.mount["yaw"], 1)
                self._note(st.calib)
        d = self.drive
        if d is not None:
            st = d.status
            lines = [f"**Mrowka udajaca prawdziwa: {st.mode}**" + (f" - {st.direction}" if st.direction else "")
                     + (f", klawisze {st.keys}" if st.keys else "")]
            if st.note:
                lines.append(st.note)
            if st.height:
                lines.append(f"wysokosc: {st.height}")
            lines.append(f"poza dla jej polityki: {d.camera}; polityka {d.policy.name}")
            self.drive_md.content = "  \n".join(lines)
