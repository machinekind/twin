"""Wiersz polecen blizniaka: `twin <polecenie>`.

    twin check                       # czy srodowisko jest gotowe (bez renderu)
    twin demo                        # przykladowe stanowisko w symulacji (2 kamery), pierwsze zainstalowane ramie
    twin demo --robot NAZWA          # ... z innym zainstalowanym ramieniem (jego demo.json, jesli je ma)
    twin card --tag-mm 50            # arkusz karty kalibracyjnej do druku
    twin board --square-mm 28        # tablica ChArUco do intrynsyk kamery
    twin calib-sim --n 5 --cameras 2 # kalibracja w symulacji, oceniana wzgledem prawdy
    twin calib-sim --robot NAZWA     # ... na innym zainstalowanym ramieniu
    twin workspace                   # co wiadomo o stanowisku (kamery, kalibracja)
    twin export <folder>             # stanowisko jako dane (MJCF + station.json) dla treningu
    twin episode                     # raport ostatniego przebiegu strategii (opoznienia stawow, dryf chwytaka)
    twin episode <plik.jsonl>        # ... z tego zapisu
    twin episode <folder>            # lista przebiegow: koniec, sukces, blad zadania
    twin eval-strategy NAZWA --episodes 50 --seed 50000   # ocena strategii w symulacji: obiekt losowany z ziarna, sukces wg sceny (touch/lift)
    twin --config configs/local.yaml eval-strategy touch --robot so101 --rule lift   # ... ustawienia z sekcji pliku i --set KLUCZ=WARTOSC
    twin ui                          # panel w przegladarce: http://localhost:8080
    twin ui --host 0.0.0.0           # ... dostepny z sieci (bez hasla - tylko w zaufanej)
    twin ui --stream-port 8766       # ... plus podglad kamer (MJPEG, tylko odczyt): http://<maszyna>:8766
    twin --config configs/local.yaml ui   # tiki chwytaka, baudrate itd. z pliku YAML

`--config` ustawia zmienna MACHINEKIND_TWIN_CONFIG (patrz `_use_config`).

Ramiona i ich backendy daja zainstalowane pakiety (SO-101: pakiet soarm z repozytorium so-arm101,
Galaxea A1X: pakiet galaxeo z repozytorium galaxeo-manipulators); `twin check` wypisuje,
ktore sa.

Trening i polityki sa w pakiecie soarm-lab (repozytorium so-arm101), ktorego blizniak nie
importuje: dostaje on stanowisko z `export`, a wytrenowana polityka jedzie w panelu jako
strategia `policy` (zakladka Strategie).
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path
from typing import Any


def _check(_: argparse.Namespace) -> int:
    """Sprawdza srodowisko bez renderowania czegokolwiek."""
    ok = True

    def line(label: str, value: str, good: bool = True) -> None:
        nonlocal ok
        ok &= good
        print(f"  {'OK ' if good else 'BLAD'}  {label:<28} {value}")

    print("twin: sprawdzenie srodowiska\n")
    line("Python", f"{sys.version.split()[0]} ({platform.machine()}, {platform.system()})")
    for mod in ("mujoco", "viser", "cv2", "numpy"):
        try:
            m = __import__(mod)
            line(mod, getattr(m, "__version__", "?"))
        except ImportError as exc:
            line(mod, f"brak ({exc}) - uv sync", False)
    line("MUJOCO_GL", os.environ.get("MUJOCO_GL", "(domyslny: glfw; bez ekranu ustaw egl)"))
    from .paths import CONFIG_ENV, config_from_env

    cfg_file = config_from_env()
    line("konfiguracja", f"{cfg_file} ({CONFIG_ENV})" if cfg_file else "domyslna (bez --config)")

    try:
        from .kinematics import RobotKinematics
        from .robots import INSTALL_HINT, get_spec, robot_folders

        if not robot_folders():
            line("ramiona", f"brak - {INSTALL_HINT}", False)
        for name, folder in robot_folders().items():
            try:
                spec = get_spec(name)
                m = RobotKinematics(spec).model
                faces = int(m.mesh_facenum.sum())
                line(f"model {spec.title}", f"{m.nbody} cial, {faces} trojkatow ({folder})")
            except Exception as exc:  # noqa: BLE001 - zly model to wiersz raportu, nie wyjatek
                line(f"model {name}", f"{type(exc).__name__}: {exc}", False)
    except Exception as exc:  # noqa: BLE001 - a broken robot registry is a report line, not a crash
        line("ramiona", f"{type(exc).__name__}: {exc}", False)

    try:
        import serial.tools.list_ports as lp

        ports = [p for p in lp.comports() if p.vid is not None]
        text = ", ".join(f"{p.device} ({p.description})" for p in ports) or "brak przejsciowek USB-serial"
        line("porty USB-serial", text)
    except ImportError:
        line("pyserial", "brak - potrzebny do prawdziwego ramienia", False)

    print("\nWszystko gotowe." if ok else "\nSa bledy - patrz wyzej.")
    return 0 if ok else 1


def _card(a: argparse.Namespace) -> int:
    import cv2

    from .calib.card import Card

    card = Card(tag_size=a.tag_mm / 1000.0)
    sheet = card.sheet(dpi=a.dpi)
    out = Path(a.out)
    cv2.imwrite(str(out), sheet[..., ::-1])
    print(f"Zapisano {out} ({sheet.shape[1]} x {sheet.shape[0]} px, {a.dpi} dpi).")
    print(f"Drukuj w skali 100%. Bok czarnego kwadratu ma miec {a.tag_mm:.1f} mm - ZMIERZ go po wydruku.")
    return 0


def _board(a: argparse.Namespace) -> int:
    import cv2

    from .calib.intrinsics import Board

    board = Board(square=a.square_mm / 1000.0, marker=0.75 * a.square_mm / 1000.0)
    sheet = board.image(dpi=a.dpi)
    out = Path(a.out)
    cv2.imwrite(str(out), sheet)
    print(f"Zapisano {out} ({sheet.shape[1]} x {sheet.shape[0]} px, {a.dpi} dpi).")
    print(f"Drukuj w skali 100%. Bok kwadratu ma miec {a.square_mm:.1f} mm - ZMIERZ go po wydruku "
          f"i wpisz w panelu (zakladka Kalibracja).")
    return 0


def _calib_sim(a: argparse.Namespace) -> int:
    """`calib.simulate` dla wybranego ramienia - te same wiersze wyniku, co `python -m ...simulate`."""
    import numpy as np

    from .calib import simulate
    from .robots import get_spec

    robot = _installed_robot(a.robot)
    if robot is None:
        return 1
    spec = get_spec(robot)
    print(f"Ramie: {spec.title}")

    def show(r):
        if a.verbose:
            prog = "  ".join(f"{c}: {o} obs {s:.0f}st" for c, (o, _, s) in r.progress.items())
            print(f"    poza {r.index:2d}  {prog}  {r.note}")

    rows = []
    for seed in range(a.seed, a.seed + a.n):
        res = simulate.run(seed, a.cameras, spec=spec, extreme=a.extreme, on_step=show)
        rows.append(res)
        for cam in sorted(res.trusted):
            flag = "" if res.trusted[cam] else f"  NIEZAUFANA ({res.reasons[cam]})"
            print(f"ziarno {seed:3d} {cam}: pozy={res.poses:2d} rms={res.rms_px[cam]:.2f}px  "
                  f"blad={res.err_mm[cam]:.2f}mm {res.err_deg[cam]:.3f}st  "
                  f"karta={res.card_mm:.2f}mm {res.card_deg:.2f}st{flag}")

    ok = [(r, c) for r in rows for c in r.trusted if r.trusted[c]]
    total = sum(len(r.trusted) for r in rows)
    print(f"\nzaufane kamery: {len(ok)}/{total}")
    if ok:
        mm = np.array([r.err_mm[c] for r, c in ok])
        deg = np.array([r.err_deg[c] for r, c in ok])
        px = np.array([r.rms_px[c] for r, c in ok])
        for name, v, unit in (("przesuniecie", mm, "mm"), ("obrot", deg, "st"), ("residuum", px, "px")):
            print(f"{name:>13}: mediana {np.median(v):7.3f} {unit}  najgorzej {v.max():7.3f} {unit}")
    return 0


def _workspace(a: argparse.Namespace) -> int:
    from .workspace import Workspace

    ws = Workspace.load(a.path)
    exists = ws.path is not None and Path(ws.path).is_file()
    where = Path(ws.path).resolve() if ws.path is not None else "-"
    print(f"Stanowisko: {where} {'' if exists else '(jeszcze nie zapisane - wartosci domyslne)'}")
    print(f"  ramie: {ws.robot}, backend: {ws.backend}, port: {ws.port or '-'}")
    print(f"  karta: tag {ws.card_obj().tag_size * 1000:.1f} mm")
    if not ws.cameras:
        print("  kamery: brak  (przyklad w symulacji z dwiema kamerami: twin demo)")
    for cam in ws.cameras:
        cal = cam.calibration
        state = ("zaufana" if cam.trusted else f"NIEZAUFANA ({cal.get('reason', '')})") if cam.calibrated \
            else "nieskalibrowana"
        rms = f", residuum {cal['rms_px']:.2f} px" if "rms_px" in cal else ""
        print(f"  kamera {cam.name}: zrodlo {cam.source}, {cam.width}x{cam.height}, "
              f"intrynsyki {cam.intrinsics_from}, {state}{rms}")
    return 0


def _export(a: argparse.Namespace) -> int:
    from .workspace import Workspace

    missing = _missing_robot(_workspace_path(a.path))
    if missing:
        print(missing, file=sys.stderr)
        return 1
    ws = Workspace.load(a.path)
    folder = ws.export(a.folder)
    print(f"Stanowisko -> {folder.resolve()}: scene.xml, assets/, station.json "
          f"(ramie {ws.robot}, kamer {sum(c.enabled for c in ws.cameras)})")
    return 0


def _episode(a: argparse.Namespace) -> int:
    """The report of one run's episode log, or one line per run for a folder of them.

    Without a path it reports the newest log next to the workspace file.
    """
    from .episode import EpisodeLog, episodes_dir
    from .episode_report import report

    target = Path(a.target) if a.target else episodes_dir(a.path)
    if not target.exists():
        print(f"twin episode: nie ma {target}", file=sys.stderr)
        return 1

    if target.is_dir():
        logs = sorted(target.glob("*.jsonl"), key=lambda path: _natural(path.stem), reverse=True)
        if not logs:
            print(f"brak zapisow przebiegow w {target}", file=sys.stderr)
            return 1
        if a.target:
            _list_episodes(target, logs)
            return 0
        target = logs[0]

    try:
        log = EpisodeLog.load(target)
    except ValueError as exc:
        print(f"twin episode: {exc}", file=sys.stderr)
        return 1

    offset_m = tuple(value / 1000.0 for value in a.offset_mm)
    print("\n".join(report(log, gripper=a.gripper, offset_m=offset_m).lines()))
    return 0


def _list_episodes(folder: Path, logs: list[Path]) -> None:
    from .episode import EpisodeLog

    print(f"Przebiegi w {folder} (najnowsze pierwsze):")
    for path in logs:
        try:
            summary = EpisodeLog.summary(path)
        except (OSError, ValueError) as exc:
            print(f"  {path.stem:<24} nieczytelny: {exc}")
            continue

        end = summary.end
        success = "-" if end.success is None else ("tak" if end.success else "nie")
        error = "-" if end.error is None else f"{end.error:.3g}"
        length = "-" if end.t is None else f"{end.t:.2f}"
        print(f"  {path.stem:<24} {end.code:<13} sukces {success:<4} blad zadania {error:<9} "
              f"czas {length:>8} s  krokow {end.steps:<6} {end.reason}")


def _eval_strategy(a: argparse.Namespace) -> int:
    """Runs one strategy many times in simulation, each episode with an object drawn from the seed,
    and counts what the scene judges a success. Refuses before writing anything when a check fails.
    """
    import math

    from . import evaluation
    from .detectors import config_section as detector_section
    from .detectors import load_detector
    from .scene_truth import Area, ObjectSpec
    from .strategies import config_section, load_strategy
    from .workspace import Workspace

    def refuse(message: str) -> int:
        print(f"twin eval-strategy: {message}", file=sys.stderr)
        return 1

    try:
        plugin = load_strategy(a.name)
    except KeyError as exc:
        return refuse(exc.args[0])
    except (TypeError, ValueError) as exc:
        return refuse(str(exc))

    if a.robot:
        robot = _installed_robot(a.robot)
        if robot is None:
            return 1
        workspace = Workspace(robot=robot)
    else:
        missing = _missing_robot(_workspace_path(a.path))
        if missing:
            print(missing, file=sys.stderr)
            return 1
        try:
            workspace = Workspace.load(a.path)
        except (OSError, TypeError, ValueError) as exc:
            return refuse(f"stanowisko: {exc}")

    try:
        values = evaluation.plugin_values(plugin, config_section(plugin.name), a.assignments)
        detector, detector_values = None, {}
        if a.detector:
            detector = load_detector(a.detector)
            detector_values = evaluation.plugin_values(detector, detector_section(detector.name))

        label = evaluation.object_label(a.label, plugin, values, detector, detector_values)
        obj = ObjectSpec(label=label, shape=a.shape, size_m=evaluation.object_size_m(a.size_mm),
                         mass_kg=a.mass_g / 1000.0)
        desk = workspace.spec().desk
        radius = (a.radius_m[0], a.radius_m[1]) if a.radius_m else desk.cube_radius
        bearing = ((math.radians(a.bearing_deg[0]), math.radians(a.bearing_deg[1])) if a.bearing_deg
                   else desk.cube_bearing)
        area = Area(radius, bearing, (math.radians(a.yaw_deg[0]), math.radians(a.yaw_deg[1])))
        campaign = evaluation.Campaign(
            plugin=plugin, settings=values, rule=a.rule, label=label, obj=obj, area=area, seed=a.seed,
            episodes=a.episodes, time_limit_s=a.time_limit_s, speed_deg_s=a.speed_deg_s,
            wall_limit_s=a.wall_limit_s, detector=detector, detector_settings=detector_values,
        )
        facts = evaluation.preflight(workspace, campaign)
        folder = evaluation.campaign_folder(Path(a.out) if a.out else None, workspace, plugin.name)
    except KeyError as exc:
        return refuse(exc.args[0])
    except (OSError, TypeError, ValueError) as exc:
        return refuse(str(exc))

    print("\n".join(evaluation.header_lines(campaign, workspace, folder)), flush=True)
    result = evaluation.run_campaign(workspace, campaign, folder, facts=facts,
                                     on_episode=lambda row: print(evaluation.row_line(row), flush=True))
    print("\n".join(evaluation.footer_lines(campaign, result)))

    if result.aborted:
        return refuse(f"przerwano: {result.aborted}")
    return 130 if result.interrupted else 0


def _natural(name: str) -> list[Any]:
    """`name` split into text and numbers, so `-10` sorts after `-2`."""
    import re

    return [int(part) if part.isdigit() else part for part in re.split(r"(\d+)", name)]


#: The example station in simulation: two cameras whose calibrated pose is their true pose in
#: the scene (trusted), so the cube detector (Detektory tab, or `twin ui --detector cube`) sees
#: the cube at once. `demo` writes the chosen arm into it; an arm with its own example (a table
#: and cameras for its reach) takes that one: `robots.DEMO_FILE` from the arm's folder, which its
#: package provides.
DEMO_WORKSPACE = Path("examples") / "twin.sim.json"


def _installed_robot(name: str | None) -> str | None:
    """Ramie z --robot albo pierwsze zainstalowane; None (i komunikat), gdy nie ma go w srodowisku."""
    from .robots import INSTALL_HINT, default_robot, robot_folders

    installed = robot_folders()
    robot = (name or default_robot()).lower()
    if robot not in installed:
        known = ", ".join(installed) or "zadne"
        print(f"Ramie {robot!r} nie jest zainstalowane (zainstalowane: {known}) - {INSTALL_HINT}.",
              file=sys.stderr)
        return None
    return robot


def _copy_demo(robot: str, dst: Path) -> Path | None:
    """Writes the example workstation with `robot` to `dst` and returns the file it copied: the robot
    folder's own `DEMO_FILE`, else `DEMO_WORKSPACE` from the clone. None when neither exists."""
    import json

    from .robots import DEMO_FILE, resolve_asset, robot_folders

    src = robot_folders()[robot] / DEMO_FILE
    if not src.is_file():
        src = resolve_asset(DEMO_WORKSPACE)
        if not src.is_file():
            return None
    dst.parent.mkdir(parents=True, exist_ok=True)
    data = json.loads(src.read_text(encoding="utf-8"))
    data["robot"] = robot
    dst.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return src


def _demo(a: argparse.Namespace) -> int:
    from .paths import data_path
    from .robots import DEMO_FILE, robot_folders

    robot = _installed_robot(a.robot)
    if robot is None:
        return 1
    # To samo co workspace.default_path(), ale bez importu `workspace` (ciagnie mujoco).
    dst = Path(a.path) if a.path else data_path(Path("workspace") / "twin.json")
    if dst.exists() and not a.force:
        print(f"{dst} juz istnieje - nie nadpisuje (to moze byc Twoja kalibracja).\n"
              f"Nadpisz: twin demo --force   albo inny plik: twin demo --path <plik>",
              file=sys.stderr)
        return 1
    src = _copy_demo(robot, dst)
    if src is None:
        print(f"Brak {DEMO_WORKSPACE} - to polecenie dziala z klonu repozytorium.", file=sys.stderr)
        return 1
    own = src == robot_folders()[robot] / DEMO_FILE
    print(f"Skopiowano {src} -> {dst}")
    if own:
        print(f"Stanowisko w symulacji z pakietu ramienia {robot}.")
    else:
        print("Stanowisko w symulacji: ramie sim, dwie kamery symulowane (sym-lewa, sym-prawa) z zaufana kalibracja.")
    others = [n for n in robot_folders() if n != robot]
    if others and not a.robot:
        print(f"Zainstalowane tez: {', '.join(others)} - inne ramie: twin demo --robot <nazwa>")
    ws_flag = f" --workspace {dst}" if a.path else ""
    print(f"Ramie: {robot}. Dalej: twin ui{ws_flag}  ->  Ramie: sim, Polacz.")
    return 0


def _workspace_path(workspace: str | None) -> Path:
    """Ktory plik stanowiska wezmie panel - widac bez przegladarki.

    Ta sama regula co `Workspace.load`, ale bez jego importu (mujoco), zeby napis
    byl przed wolnym startem panelu.
    """
    from .paths import data_path

    return (Path(workspace) if workspace else data_path(Path("workspace") / "twin.json")).resolve()


def _first_start(ws: Path) -> str | None:
    """With no workstation file yet, writes the example workstation with the first installed robot,
    as `twin demo` does. Returns that robot, or None when it wrote nothing."""
    from .robots import default_robot, robot_folders

    if ws.exists() or not robot_folders():
        return None
    robot = default_robot()
    return robot if _copy_demo(robot, ws) is not None else None


def _missing_robot(ws: Path) -> str | None:
    """What to do when the workstation file names a robot no installed package registers; None when
    it opens."""
    import json

    from .robots import INSTALL_HINT, robot_folders

    try:
        robot = json.loads(ws.read_text(encoding="utf-8")).get("robot")
    except (OSError, ValueError, AttributeError):
        return None
    installed = robot_folders()
    if not robot or str(robot).lower() in installed:
        return None
    known = ", ".join(installed) or "zadne"
    return (f"Stanowisko {ws} ma ramie {robot!r}, ktorego nie ma w srodowisku (zainstalowane: {known}).\n"
            f"Dalej: {INSTALL_HINT}.\n"
            f"Albo stanowisko przykladowe w tym pliku: twin demo --force --path {ws}")


def _ui_banner(ws: Path, example: str | None = None) -> str:
    """The workstation file the panel opens and its state; `example` is the robot `_first_start` wrote."""
    if example:
        state = f"nowe: przykladowe stanowisko w symulacji, ramie {example}"
    elif ws.is_file():
        state = "istnieje"
    else:
        state = "nowe, puste - przyklad: twin demo"
    return f"Stanowisko: {ws} ({state})"


def _ui_startup_line(a: argparse.Namespace) -> str:
    """What the panel does by itself at start (--arm, --detector): one line before the slow import."""
    arm = "polacz (backend i port stanowiska, bez domu)" if a.arm else "nie"
    line = f"Start panelu: ramie {arm}"
    if a.detector:
        line += f", detektory {', '.join(a.detector)}"
    return line


def _ui(a: argparse.Namespace) -> int:
    ws = _workspace_path(a.workspace)
    print(_ui_banner(ws, _first_start(ws)), flush=True)
    missing = _missing_robot(ws)
    if missing:
        print(missing, file=sys.stderr)
        return 1
    print(_ui_startup_line(a), flush=True)
    from .ui.app import main as ui_main

    argv = ["--host", a.host, "--port", str(a.port)]
    if a.workspace:
        argv += ["--workspace", a.workspace]
    if a.stream_port is not None:
        argv += ["--stream-port", str(a.stream_port), "--stream-host", a.stream_host]
    if a.arm:
        argv.append("--arm")
    for name in a.detector or []:
        argv += ["--detector", name]
    return ui_main(argv)


def _use_config(path: str) -> None:
    """Ustawia MACHINEKIND_TWIN_CONFIG (sciezka bezwzgledna) i od razu sprawdza plik.

    Zmienna, a nie argument, bo blizniak wola `load_config()` w wielu miejscach.
    """
    import yaml

    from .config import config_from_data, read_config_data
    from .paths import CONFIG_ENV

    cfg = Path(path).expanduser().resolve()
    if not cfg.is_file():
        raise SystemExit(f"twin: --config {path}: nie ma takiego pliku")
    os.environ[CONFIG_ENV] = str(cfg)
    try:
        # An unknown key in `safety` or `robot` fails here, not mid-session. Joint tables need the
        # robot and backend sections need their plugin: both are checked when the arm connects.
        config_from_data({k: v for k, v in read_config_data().items() if k != "joints"})
    except (ValueError, TypeError, yaml.YAMLError) as exc:
        raise SystemExit(f"twin: --config {path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    if getattr(a, "config", None):
        _use_config(a.config)
    return a.fn(a)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="twin", description="Cyfrowy blizniak stanowiska z ramieniem robota.")
    config_help = ("plik YAML konfiguracji (np. configs/local.yaml: robot.port, sekcje backendow "
                   "nazwane jak backend, np. testbot:); ustawia MACHINEKIND_TWIN_CONFIG")
    ap.add_argument("--config", default=None, help=config_help)
    # To samo po nazwie polecenia (`twin ui --config ...`); SUPPRESS nie nadpisuje wartosci sprzed niej.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS, help=config_help)
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_parser = sub.add_parser

    def add_parser(*args, **kw):
        return _add_parser(*args, parents=[common], **kw)
    sub.add_parser = add_parser  # type: ignore[method-assign]

    sub.add_parser("check", help="sprawdz srodowisko (bez renderu)").set_defaults(fn=_check)

    p = sub.add_parser("demo", help="przykladowe stanowisko w symulacji -> workspace/twin.json")
    p.add_argument("--path", default=None, help="plik docelowy (domyslnie workspace/twin.json)")
    p.add_argument("--force", action="store_true", help="nadpisz istniejacy plik stanowiska")
    p.add_argument("--robot", default=None, help="ktore ramie (domyslnie pierwsze zainstalowane)")
    p.set_defaults(fn=_demo)

    p = sub.add_parser("card", help="arkusz karty kalibracyjnej do druku")
    p.add_argument("--out", default="karta_kalibracyjna.png")
    p.add_argument("--tag-mm", type=float, default=50.0, help="bok czarnego kwadratu taga [mm]")
    p.add_argument("--dpi", type=int, default=300)
    p.set_defaults(fn=_card)

    p = sub.add_parser("board", help="tablica ChArUco do intrynsyk kamery (PNG, A4)")
    p.add_argument("--out", default="tablica_charuco.png")
    p.add_argument("--square-mm", type=float, default=28.0, help="bok kwadratu szachownicy [mm]")
    p.add_argument("--dpi", type=int, default=300)
    p.set_defaults(fn=_board)

    p = sub.add_parser("calib-sim", help="kalibracja w symulacji wzgledem prawdy (renderuje!)")
    p.add_argument("--n", type=int, default=5, help="ile losowych stanowisk")
    p.add_argument("--cameras", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--extreme", action="store_true", help="karta na granicy przekrzywienia")
    p.add_argument("--robot", default=None, help="ktore ramie (domyslnie pierwsze zainstalowane)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=_calib_sim)

    p = sub.add_parser("workspace", help="co wiadomo o stanowisku")
    p.add_argument("--path", default=None)
    p.set_defaults(fn=_workspace)

    p = sub.add_parser("export", help="stanowisko jako dane (MJCF + station.json) dla pakietow bez importu blizniaka")
    p.add_argument("folder")
    p.add_argument("--path", default=None, help="plik stanowiska (domyslnie workspace/twin.json)")
    p.set_defaults(fn=_export)

    p = sub.add_parser("episode", help="zapis przebiegu strategii: raport albo lista przebiegow")
    p.add_argument("target", nargs="?", default=None, metavar="PATH",
                   help="plik .jsonl (raport) albo folder (lista); domyslnie najnowszy zapis obok stanowiska")
    p.add_argument("--path", default=None, help="plik stanowiska (domyslnie workspace/twin.json)")
    p.add_argument("--gripper", default=None, metavar="ETYKIETA",
                   help="etykieta obiektu widzianego przez kamery, ktory oznacza chwytak (dryf odczyty - kamery)")
    p.add_argument("--offset-mm", nargs=3, type=float, default=(0.0, 0.0, 0.0), metavar=("X", "Y", "Z"),
                   help="gdzie ten znacznik siedzi w ukladzie TCP [mm]")
    p.set_defaults(fn=_episode)

    p = sub.add_parser("eval-strategy",
                       help="ocena strategii w symulacji: wiele przebiegow z obiektem losowanym z ziarna, sukces wg sceny")
    p.add_argument("name", metavar="NAZWA", help="strategia: nazwa punktu wejscia, jak w zakladce Strategie")
    p.add_argument("--episodes", type=int, default=10, help="ile przebiegow (domyslnie 10)")
    p.add_argument("--seed", type=int, default=0,
                   help="ziarno: przebieg i losuje obiekt z ziarna S+i, wiec --seed S+i --episodes 1 go powtarza")
    p.add_argument("--rule", choices=("touch", "lift"), default="touch",
                   help="kiedy scena uznaje sukces: touch = szczeka dotyka obiektu, lift = obie szczeki "
                        "i obiekt uniesiony o 4 cm (domyslnie touch)")
    p.add_argument("--set", dest="assignments", action="append", default=[], metavar="KLUCZ=WARTOSC",
                   help="ustawienie strategii ponad sekcje pliku konfiguracji; mozna kilka razy")
    where = p.add_mutually_exclusive_group()
    where.add_argument("--path", default=None, help="plik stanowiska (domyslnie workspace/twin.json)")
    where.add_argument("--robot", default=None, metavar="NAZWA",
                       help="ramie bez pliku stanowiska (bez kamer), np. w srodowisku innego pakietu")
    p.add_argument("--out", default=None, metavar="FOLDER",
                   help="nowy albo pusty folder na zapisy i summary.json "
                        "(domyslnie episodes/eval-<strategia>-<czas> obok stanowiska)")
    p.add_argument("--label", default=None, metavar="ETYKIETA",
                   help="klucz obiektu w Observation.objects (domyslnie ustawienie label strategii albo cube)")
    p.add_argument("--shape", choices=("box",), default="box", help="ksztalt obiektu (na razie tylko box)")
    p.add_argument("--size-mm", nargs="+", type=float, default=[30.0], metavar="MM",
                   help="krawedz szescianu albo trzy krawedzie pudelka [mm] (domyslnie 30)")
    p.add_argument("--mass-g", type=float, default=30.0, help="masa obiektu [g] (domyslnie 30)")
    p.add_argument("--radius-m", nargs=2, type=float, default=None, metavar=("OD", "DO"),
                   help="odleglosc obiektu od podstawy [m] (domyslnie pierscien kostki z opisu ramienia)")
    p.add_argument("--bearing-deg", nargs=2, type=float, default=None, metavar=("OD", "DO"),
                   help="kierunek od osi x podstawy [st] (domyslnie z opisu ramienia)")
    p.add_argument("--yaw-deg", nargs=2, type=float, default=(-180.0, 180.0), metavar=("OD", "DO"),
                   help="obrot obiektu wokol pionu [st] (domyslnie -180 180)")
    p.add_argument("--time-limit-s", type=float, default=120.0,
                   help="najdluzszy przebieg w czasie symulacji [s]; krotszy limit strategii zostaje (domyslnie 120)")
    p.add_argument("--speed-deg-s", type=float, default=None,
                   help="granica predkosci stawow jak suwak w panelu [st./s] (domyslnie bez granicy)")
    p.add_argument("--wall-limit-s", type=float, default=None,
                   help="przerwij przebieg po tylu sekundach rzeczywistych (wynik zalezy wtedy od maszyny)")
    p.add_argument("--detector", default=None, metavar="NAZWA",
                   help="pozy obiektu z tego detektora przez zaufane kamery symulowane stanowiska "
                        "(domyslnie prawda symulacji); ustawienia z jego sekcji konfiguracji")
    p.set_defaults(fn=_eval_strategy)

    p = sub.add_parser("ui", help="panel blizniaka w przegladarce (viser)")
    # Panel rusza prawdziwym ramieniem i nie ma hasla, a viser dzieli stan GUI miedzy
    # wszystkich podlaczonych - kazdy, kto dosiegnie portu, moze nacisnac "Polacz" albo
    # uruchomic strategie. Dlatego domyslnie tylko ten komputer; siec na wyrazne zyczenie.
    p.add_argument("--host", default="127.0.0.1",
                   help="adres nasluchu (domyslnie tylko ten komputer; 0.0.0.0 = cala siec, BEZ hasla)")
    p.add_argument("--port", type=int, default=8080,
                   help="port panelu (domyslnie 8080; zajety przez inna usluge? np. --port 8765)")
    p.add_argument("--workspace", default=None)
    # Podglad kamer nie steruje niczym, wiec moze sluchac w sieci nawet gdy panel jest tylko lokalny.
    p.add_argument("--stream-port", type=int, default=None,
                   help="podglad kamer po HTTP (MJPEG, tylko odczyt) na tym porcie, np. 8766; bez flagi - brak")
    p.add_argument("--stream-host", default="0.0.0.0",
                   help="adres nasluchu podgladu kamer (domyslnie cala siec - obraz, bez sterowania)")
    p.add_argument("--arm", action="store_true",
                   help="polacz ramie przy starcie: backend i port ze stanowiska, bez jazdy do domu")
    p.add_argument("--detector", action="append", metavar="NAME",
                   help="uruchom detektor przy starcie (np. cube), z ustawieniami z jego sekcji "
                        "konfiguracji; mozna kilka razy")
    p.set_defaults(fn=lambda a: _ui(a))
    return ap


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
