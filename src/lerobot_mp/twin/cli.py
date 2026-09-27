"""Wiersz polecen blizniaka: `lerobot-twin <polecenie>`.

    lerobot-twin check                       # czy srodowisko jest gotowe (bez renderu), takze GPU
    lerobot-twin demo                        # przykladowe stanowisko w symulacji (2 kamery)
    lerobot-twin demo --robot a1x            # ... z ramieniem Galaxea A1X
    lerobot-twin demo --ant                  # ... i z mrowka RealAnt z tagami (zakladka Mrowka, WSAD)
    lerobot-twin card --tag-mm 50            # arkusz karty kalibracyjnej do druku
    lerobot-twin board --square-mm 28        # tablica ChArUco do intrynsyk kamery
    lerobot-twin calib-sim --n 5 --cameras 2 # kalibracja w symulacji, oceniana wzgledem prawdy
    lerobot-twin calib-sim --robot a1x       # ... na ramieniu A1X
    lerobot-twin workspace                   # co wiadomo o stanowisku (kamery, kalibracja)
    lerobot-twin train --task reach          # polityka PPO na GPU (MuJoCo Warp) + ewaluacja na CPU
    lerobot-twin train --task pickplace      # ... kostka przeniesiona na cel na blacie
    lerobot-twin eval <policy.pt> --rand     # ewaluacja polityki w zwyklym MuJoCo
    lerobot-twin policies                    # zapisane polityki i ich wyniki
    lerobot-twin ui                          # panel w przegladarce: http://localhost:8080
    lerobot-twin ui --host 0.0.0.0           # ... dostepny z sieci (bez hasla - tylko w zaufanej)
    lerobot-twin ui --stream-port 8766       # ... plus podglad kamer (MJPEG, tylko odczyt): http://<maszyna>:8766
    lerobot-twin --config configs/local.yaml ui   # tiki chwytaka, baudrate itd. z pliku YAML
    lerobot-twin specs                       # Spectacles glasses via the relay: checks only (dry)
    lerobot-twin specs --mode sim            # ... simulated arm moves to the pointed target
    lerobot-twin specs --mode real --i-know  # ... REAL arm (backend from the workspace)
    lerobot-twin specs --mode real --i-know --start-center  # ... first (after Enter) above the box centre
    lerobot-twin specs --mode sim --follow-speed 0.1        # ... teleop: the gripper follows the hand at 0.1 m/s

`--config` ustawia zmienna LEROBOT_MP_CONFIG, wiec plik widzi tez proces treningu
uruchamiany z panelu.
"""

from __future__ import annotations

import argparse
import os
import platform
import sys
from pathlib import Path


def _check(_: argparse.Namespace) -> int:
    """Sprawdza srodowisko bez renderowania czegokolwiek."""
    import numpy as np

    ok = True

    def line(label: str, value: str, good: bool = True) -> None:
        nonlocal ok
        ok &= good
        print(f"  {'OK ' if good else 'BLAD'}  {label:<28} {value}")

    print("lerobot-twin: sprawdzenie srodowiska\n")
    line("Python", f"{sys.version.split()[0]} ({platform.machine()}, {platform.system()})")
    for mod in ("mujoco", "gymnasium", "viser", "cv2", "numpy", "torch"):
        try:
            m = __import__(mod)
            line(mod, getattr(m, "__version__", "?"))
        except ImportError as exc:
            line(mod, f"brak ({exc}) - pip install -e \".[twin]\"", False)
    line("MUJOCO_GL", os.environ.get("MUJOCO_GL", "(domyslny: glfw; bez ekranu ustaw egl)"))
    from ..paths import CONFIG_ENV, config_from_env

    cfg_file = config_from_env()
    line("konfiguracja", f"{cfg_file} ({CONFIG_ENV})" if cfg_file else "domyslna (bez --config)")

    try:
        from .kinematics import RobotKinematics
        from .robots import REGISTRY, SO101

        for spec in REGISTRY.values():
            try:
                m = RobotKinematics(spec).model
                line(f"model {spec.title}", f"{m.nbody} cial, {int(m.mesh_facenum.sum())} trojkatow")
            except Exception as exc:  # noqa: BLE001 - zly model to wiersz raportu, nie wyjatek
                line(f"model {spec.title}", f"{type(exc).__name__}: {exc}", False)
        # Zgodnosc z niezaleznym modelem podgladu 3D - tylko SO-101 go ma.
        kin = RobotKinematics(SO101)
        from ..preview.model import load_model

        art = load_model()
        if art is not None:
            base = art.link_names.index("base")
            rng = np.random.default_rng(0)
            worst = 0.0
            for _ in range(20):
                j = {n: float(rng.uniform(np.degrees(lo), np.degrees(hi)))
                     for n, lo, hi in zip(SO101.joints, kin.lo, kin.hi)}
                Ta = art.link_transforms(art.from_lerobot(j))
                for name in ("upper_arm", "lower_arm", "wrist", "gripper"):
                    ref = (np.linalg.inv(Ta[base]) @ Ta[art.link_names.index(name)])[:3, 3]
                    worst = max(worst, float(np.linalg.norm(ref - kin.body(name, j)[:3, 3])))
            line("zgodnosc z Articulusem", f"{worst * 1e6:.1f} um (musi byc < 10 um)", worst < 1e-5)
    except Exception as exc:
        line("model SO-101", f"{type(exc).__name__}: {exc}", False)

    try:
        import serial.tools.list_ports as lp

        ports = [p for p in lp.comports() if p.vid is not None]
        text = ", ".join(f"{p.device} ({p.description})" for p in ports) or "brak przejsciowek USB-serial"
        line("porty USB-serial", text)
    except ImportError:
        line("pyserial", "brak - potrzebny do prawdziwego ramienia", False)

    gpu_ok, gpu_lines = _gpu_report()
    print("\nTrening na GPU (lerobot-twin train) - opcjonalny:")
    for label, value, good in gpu_lines:
        # OK gdy gotowe; "--" (nie BLAD) gdy niedostepne - trening jest opcjonalny
        print(f"  {'OK ' if good else '-- '}  {label:<28} {value}")
    print("  " + ("Trening na GPU: gotowy." if gpu_ok else
                  "Trening na GPU: niedostepny. Wymaga karty NVIDIA z CUDA, torcha z CUDA i mujoco-warp\n"
                  "  (pip install -e \".[train]\"). Panel, kalibracja, percepcja i ewaluacja polityk\n"
                  "  na CPU dzialaja bez tego."))

    print("\nWszystko gotowe." if ok else "\nSa bledy - patrz wyzej.")
    return 0 if ok else 1


def _gpu_report() -> tuple[bool, list[tuple[str, str, bool]]]:
    """Co wiadomo o GPU: torch (wersja, CUDA, karta), warp i mujoco_warp. Nic tu nie jest bledem."""
    from importlib.metadata import PackageNotFoundError, version

    out: list[tuple[str, str, bool]] = []
    ready = True
    try:
        import torch

        cuda_build = torch.version.cuda
        avail = bool(torch.cuda.is_available())
        name = torch.cuda.get_device_name(0) if avail else "-"
        out.append(("torch CUDA", f"kompilacja CUDA {cuda_build or 'brak (kolo CPU)'}, "
                                  f"cuda.is_available={avail}, karta: {name}", avail))
        ready &= avail
    except ImportError:
        out.append(("torch", "brak", False))
        ready = False
    try:
        import warp as wp

        try:  # bez bannera przy inicjalizacji
            wp.config.log_level = wp.LOG_WARNING
        except AttributeError:  # pragma: no cover - starszy warp
            wp.config.quiet = True
        wp.init()
        n = wp.get_cuda_device_count()
        names = ", ".join(d.name for d in wp.get_cuda_devices()) if n else "brak karty CUDA"
        out.append(("warp", f"{wp.config.version}, widzi GPU: {names}", n > 0))
        ready &= n > 0
    except ImportError:
        out.append(("warp", "brak - pip install -e \".[train]\"", False))
        ready = False
    except Exception as exc:  # noqa: BLE001 - sterownik, CUDA: tylko raport
        out.append(("warp", f"{type(exc).__name__}: {exc}", False))
        ready = False
    try:
        import mujoco_warp  # noqa: F401

        try:
            ver = version("mujoco-warp")
        except PackageNotFoundError:  # pragma: no cover
            ver = "?"
        out.append(("mujoco_warp", ver, True))
    except ImportError:
        out.append(("mujoco_warp", "brak - pip install -e \".[train]\"", False))
        ready = False
    return ready, out


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

    spec = get_spec(a.robot)
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
        print("  kamery: brak  (przyklad w symulacji z dwiema kamerami: lerobot-twin demo)")
    for cam in ws.cameras:
        cal = cam.calibration
        state = ("zaufana" if cam.trusted else f"NIEZAUFANA ({cal.get('reason', '')})") if cam.calibrated \
            else "nieskalibrowana"
        rms = f", residuum {cal['rms_px']:.2f} px" if "rms_px" in cal else ""
        print(f"  kamera {cam.name}: zrodlo {cam.source}, {cam.width}x{cam.height}, "
              f"intrynsyki {cam.intrinsics_from}, {state}{rms}")
    return 0


#: Przykladowe stanowiska w symulacji, po jednym na ramie - dwie kamery, ktorych
#: skalibrowana poza jest ich prawdziwa poza w scenie (zaufane), wiec "kostka z kamer"
#: dziala od razu.
DEMO_WORKSPACES = {
    "so101": Path("examples") / "twin.sim.json",
    "a1x": Path("examples") / "twin.a1x.sim.json",
}
DEMO_WORKSPACE = DEMO_WORKSPACES["so101"]


def _demo(a: argparse.Namespace) -> int:
    import shutil

    from ..paths import data_path
    from .robots import resolve_asset

    example = DEMO_WORKSPACES[a.robot]
    src = resolve_asset(example)
    if not src.is_file():
        print(f"Brak {example} - to polecenie dziala z klonu repozytorium.", file=sys.stderr)
        return 1
    # To samo co workspace.default_path(), ale bez importu `workspace` (ciagnie mujoco).
    dst = Path(a.path) if a.path else data_path(Path("workspace") / "twin.json")
    if dst.exists() and not a.force:
        print(f"{dst} juz istnieje - nie nadpisuje (to moze byc Twoja kalibracja).\n"
              f"Nadpisz: lerobot-twin demo --force   albo inny plik: lerobot-twin demo --path <plik>",
              file=sys.stderr)
        return 1
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)
    print(f"Skopiowano {src} -> {dst}")
    print("Stanowisko w symulacji: ramie sim, dwie kamery symulowane (sym-lewa, sym-prawa) z zaufana kalibracja.")
    ws_flag = f" --workspace {dst}" if a.path else ""
    if a.ant:
        import json

        from .ant import add_demo_ant
        data = add_demo_ant(json.loads(dst.read_text(encoding="utf-8")))
        dst.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
        print("+ mrowka RealAnt w symulacji (tagi 2 na grzbiecie, 1 na maszcie), blat 1,6 x 1,2 m, kamery "
              "sym-mrowka-przod i sym-mrowka-bok.")
        print(f"Dalej: lerobot-twin ui{ws_flag}  ->  zakladka Mrowka: Ucz ukladu (kilka kadrow), Poziom z blatu, "
              f"Sledz mrowke (demo), Uruchom mrowke udajaca prawdziwa, WSAD - kopia kalibruje sie sama.")
        return 0
    if a.robot == "so101":
        print(f"Dalej: lerobot-twin ui{ws_flag}  ->  Ramie: sim, Polacz  ->  Polityka: lift-v3,"
              f" 'lift: skad polozenie kostki' = kamery.")
    else:
        # Polityki w assets/policies sa uczone na SO-101 - dla innego ramienia tylko panel.
        print(f"Ramie: {a.robot}. Dalej: lerobot-twin ui{ws_flag}  ->  Ramie: sim, Polacz.")
    return 0


def _train_plan(dyn, no_rand: bool, spread: float):
    """(randomizacja treningu, [(etykieta, klucz w meta.evals, randomizacja ewaluacji)]).

    "Bez randomizacji" to zmierzona dynamika bez rozrzutu - nie model Menagerie
    (`Randomization.none()` gubil identyfikacje i model percepcji kostki, a log
    i tak pisal "dynamika: identyfikacja"). Ewaluacja "z randomizacja" jest tylko
    wtedy, gdy trening ja mial: wczesniej przy --no-rand druga ewaluacja nominalna
    trafiala do `cpu_rand` i panel pokazywal ja jako odpornosc na randomizacje.
    """
    from .rl.randomize import Randomization

    rand = Randomization.around(dyn, 0.0 if no_rand else spread)
    evals = [("bez randomizacji", "cpu_nominal", Randomization.nominal(dyn))]
    if rand.randomized:
        evals.append(("z randomizacja", "cpu_rand", rand))
    return rand, evals


def _describe(rand) -> str:
    c = rand.centre
    delay = f"{rand.min_delay}" if rand.min_delay == rand.max_delay else f"{rand.min_delay}-{rand.max_delay}"
    return (f"dynamika: {c.source} (kp x{c.kp:.2f}, tlumienie x{c.damping:.2f}, opoznienie {delay} takt.); "
            f"randomizacja: {'tak' if rand.randomized else 'brak'}; "
            f"percepcja kostki: {'jak z kamer' if rand.fold_yaw or rand.cube_delay else 'idealna'}")


def _train(a: argparse.Namespace) -> int:
    import json
    import time

    from .rl.evaluate import evaluate
    from .rl.policy import DEFAULT_DIR
    from .rl.ppo import PPOConfig, train
    from .rl.randomize import Dynamics
    from .workspace import Workspace

    ws = Workspace.load(a.workspace)
    name = a.name or f"{a.task}-{time.strftime('%Y%m%d-%H%M%S')}"
    out = Path(a.out or DEFAULT_DIR) / name
    dyn = Dynamics.from_dict(ws.dynamics)
    rand, evals = _train_plan(dyn, a.no_rand, a.spread)
    cfg = PPOConfig(num_envs=a.envs, iterations=a.iters, seed=a.seed)
    init = None
    if a.init:
        from .rl.policy import Policy

        init = Policy.load(a.init)
        a.task = init.task.name
        cfg.init_std = 0.25                            # douczanie: mniejsza eksploracja na starcie
        cfg.critic_warmup = 30                         # dla polityk bez zapisanego krytyka (patrz ppo)
        print(f"Start z polityki {a.init} ({init.task.name})")
    print(f"Trening {a.task}: {a.envs} swiatow x {a.iters} iteracji -> {out}")
    print(f"  {_describe(rand)}{'' if a.no_rand else f', rozrzut x{a.spread}'}")

    def show(p):
        if p.iteration == 1 or p.iteration % 10 == 0 or p.status != "uczenie":
            print(f"  it {p.iteration:4d}/{p.iterations}  {p.steps / 1e6:7.2f} M krokow  {p.fps / 1e3:5.0f} k/s  "
                  f"sukces {p.success:5.1%}  nagroda {p.reward:7.2f}  {p.elapsed:5.0f} s", flush=True)

    stop = None
    if a.stop_file:
        # Panel zatrzymuje trening, tworzac plik - polityka z tego miejsca zostaje zapisana.
        import threading

        stop = threading.Event()
        flag = Path(a.stop_file)

        def watch():
            while not stop.is_set():
                if flag.exists():
                    stop.set()
                time.sleep(0.5)
        threading.Thread(target=watch, daemon=True).start()
    pol = train(a.task, cfg, workspace=ws, randomization=rand, out_dir=out, on_progress=show, stop=stop,
                init=init)
    pol = pol.to("cpu")
    print("Ewaluacja na CPU (zwykle MuJoCo, inny silnik niz w treningu):")
    for label, key, r in evals:
        res = evaluate(pol, a.eval_episodes, randomization=r, workspace=ws)
        pol.meta.evals[key] = res
        print(f"  {label:17s} sukces {res['success']:5.1%}  ({json.dumps({k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()})})")
    pol.save(out / "policy.pt")
    print(f"Zapisano {out / 'policy.pt'}")
    return 0


def _eval(a: argparse.Namespace) -> int:
    from .rl.evaluate import evaluate
    from .rl.policy import Policy
    from .rl.randomize import Dynamics, Randomization
    from .workspace import Workspace

    ws = Workspace.load(a.workspace)
    pol = Policy.load(a.policy)
    # Randomizacja wokol zmierzonej dynamiki - jak w treningu i w panelu, nie wokol Menagerie.
    rand = Randomization.around(Dynamics.from_dict(ws.dynamics)) if a.rand else None
    res = evaluate(pol, a.episodes, randomization=rand, workspace=ws)
    print(f"{a.policy}: {res}")
    return 0


def _policies(a: argparse.Namespace) -> int:
    from .rl.policy import DEFAULT_DIR, list_policies

    items = list_policies(a.dir or DEFAULT_DIR)
    if not items:
        print("Brak zapisanych polityk. Naucz pierwsza:  lerobot-twin train --task reach")

    def pct(key):
        v = p["evals"].get(key, {}).get("success")
        return "  -  " if v is None else f"{v:5.1%}"

    for p in items:
        print(f"  {p['name']:32s} {p['task']:6s} GPU {p['success'] or 0:5.1%}  "
              f"CPU bez/z rand. {pct('cpu_nominal')} / {pct('cpu_rand')}  "
              f"{p['steps'] / 1e6:6.1f} M  {p['created']}")
    return 0


def _ui_banner(workspace: str | None) -> str:
    """Ktory plik stanowiska i katalog polityk wezmie panel - widac bez przegladarki.

    Te same reguly co `Workspace.load` i `rl.policy.DEFAULT_DIR`, ale bez ich
    importu (mujoco, torch), zeby napis byl przed wolnym startem panelu.
    """
    from ..paths import data_path

    ws = (Path(workspace) if workspace else data_path(Path("workspace") / "twin.json")).resolve()
    state = "istnieje" if ws.is_file() else "nowe, puste - przyklad: lerobot-twin demo"
    policies = data_path(Path("workspace") / "policies").resolve()
    return f"Stanowisko: {ws} ({state})\nPolityki:   {policies}"


def _ui(a: argparse.Namespace) -> int:
    from .ui.app import main as ui_main

    print(_ui_banner(a.workspace), flush=True)
    argv = ["--host", a.host, "--port", str(a.port)]
    if a.workspace:
        argv += ["--workspace", a.workspace]
    if a.stream_port is not None:
        argv += ["--stream-port", str(a.stream_port), "--stream-host", a.stream_host]
    return ui_main(argv)


# Summary of docs/TWIN.md ("Czym A1X rozni sie od SO-101").
SPECS_REAL_WARNING = """\
UWAGA: okulary rusza PRAWDZIWYM ramieniem.
  - Brak hamulcow: odciecie zasilania zrzuca ramie. STOP trzyma zmierzona poze strumieniem.
  - Jedyna ochrona to limity, ograniczenie predkosci i wykrywanie uderzenia (opoznienie stawu,
    wysilek) z wycofaniem - progi wysilku na A1X sa zgadywane. Najpierw --mode dry, potem sim.
  - Rece i przedmioty z dala od ramienia poza wskazanym punktem. Enter albo Ctrl-C = STOP."""


def _specs(a: argparse.Namespace) -> int:
    import json
    import logging
    import threading
    import time

    from .runtime import Twin
    from .specsreach import SpecsReach
    from .workspace import Workspace

    if a.mode == "real" and not a.i_know:
        print("lerobot-twin specs --mode real rusza prawdziwym ramieniem - dodaj --i-know.", file=sys.stderr)
        return 2
    # Refusals, impacts, a missing relay and joint efforts (dry) are reported via logging.
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ws = Workspace.load(a.workspace)
    # dry and sim default to the simulated arm so nothing touches hardware without --backend.
    backend = a.backend or ("sim" if a.mode in ("dry", "sim") else None)
    if a.mode == "real":
        print(SPECS_REAL_WARNING, flush=True)
    twin = Twin(ws)
    try:
        twin.connect(backend, port=a.port, go_home=False, threaded=True)
    except Exception as exc:
        print(f"Nie polaczono z ramieniem: {exc}", file=sys.stderr)
        twin.close()
        return 1
    st = twin.status
    print(f"Ramie: {st.backend} ({'symulacja' if st.simulated else 'prawdziwe'}), tryb {a.mode}, "
          f"przekaznik {a.relay}", flush=True)

    def show(msg: dict) -> None:
        print(json.dumps(msg), flush=True)

    drv = SpecsReach(twin, relay_url=a.relay, mode=a.mode, grasp=a.grasp, allow_untrusted=not a.trusted_only,
                     real_ok=a.mode == "real" and a.i_know, on_state=show, max_jump_deg=a.max_jump_deg,
                     hold_off_s=a.hold_off, follow_speed=a.follow_speed,
                     follow_floor_m=a.follow_floor)
    # From the home/zero pose every box point is a > 45 deg jump ("jump" refusal), so go above the centre first.
    if a.start_center:
        if not st.simulated:
            if a.mode != "real":
                print("--start-center na prawdziwym ramieniu tylko w --mode real.", file=sys.stderr)
                twin.close()
                return 2
            try:
                input("Enter = ramie jedzie etapami (10 st./s) nad srodek pudelka zasiegu; Ctrl-C = koniec. ")
            except (EOFError, KeyboardInterrupt):
                print("\nBez startu nad srodek - koniec.", flush=True)
                twin.hold_measured()
                twin.close()
                return 1
        why = drv.start_center(10.0)
        if why is not None:
            print(f"Start nad srodek pudelka nieudany: {why} - ramie trzyma poze.", file=sys.stderr)
            if not st.simulated:
                twin.hold_measured()
                twin.close()
                return 1
    elif st.simulated:
        goal = drv.center_pose()
        try:
            if goal is not None:
                twin.move(goal, 2.0, owner="specs-start")
        except RuntimeError as exc:
            print(f"Symulacja: nie ustawiono pozy startowej: {exc}", file=sys.stderr)
        finally:
            twin.release("specs-start")
    stop = threading.Event()
    if sys.stdin is not None and sys.stdin.isatty():
        def keyboard() -> None:
            try:
                sys.stdin.readline()
            except Exception:
                return
            stop.set()
        threading.Thread(target=keyboard, name="specs-klawiatura", daemon=True).start()
    drv.start()
    code = 0
    try:
        t_end = time.monotonic() + a.seconds if a.seconds else None
        while not stop.is_set() and (t_end is None or time.monotonic() < t_end):
            if not twin.connected:
                print(f"Ramie rozlaczone: {twin.status.error}", file=sys.stderr)
                code = 1
                break
            stop.wait(0.2)
        if stop.is_set():                                  # Enter pressed
            twin.estop("STOP z klawiatury (specs)")
    except KeyboardInterrupt:
        twin.estop("Ctrl-C (specs)")
    finally:
        drv.stop()
        twin.hold_measured()
        twin.close()
        print("Koniec: okulary odlaczone, ramie rozlaczone.", flush=True)
    return code


def _use_config(path: str) -> None:
    """Ustawia LEROBOT_MP_CONFIG (sciezka bezwzgledna) i od razu sprawdza plik.

    Zmienna, a nie argument, bo blizniak wola `load_config()` w wielu miejscach,
    a trening z panelu idzie w osobnym procesie - dziedziczy ja ze srodowiska.
    """
    import yaml

    from ..config import load_config
    from ..paths import CONFIG_ENV

    cfg = Path(path).expanduser().resolve()
    if not cfg.is_file():
        raise SystemExit(f"lerobot-twin: --config {path}: nie ma takiego pliku")
    os.environ[CONFIG_ENV] = str(cfg)
    try:
        load_config()                                  # nieznany klucz = blad teraz, nie w polowie sesji
    except (ValueError, TypeError, yaml.YAMLError) as exc:
        raise SystemExit(f"lerobot-twin: --config {path}: {exc}") from exc


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    if getattr(a, "config", None):
        _use_config(a.config)
    return a.fn(a)


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="lerobot-twin", description="Cyfrowy blizniak stanowiska SO-101.")
    config_help = ("plik YAML konfiguracji (np. configs/local.yaml: robot.center_ticks, "
                   "gripper_*_ticks, baudrate); ustawia LEROBOT_MP_CONFIG")
    ap.add_argument("--config", default=None, help=config_help)
    # To samo po nazwie polecenia (`lerobot-twin ui --config ...`); SUPPRESS nie nadpisuje wartosci sprzed niej.
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--config", default=argparse.SUPPRESS, help=config_help)
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add_parser = sub.add_parser

    def add_parser(*args, **kw):
        return _add_parser(*args, parents=[common], **kw)
    sub.add_parser = add_parser  # type: ignore[method-assign]

    sub.add_parser("check", help="sprawdz srodowisko (bez renderu), takze GPU do treningu").set_defaults(fn=_check)

    p = sub.add_parser("demo", help="przykladowe stanowisko w symulacji -> workspace/twin.json")
    p.add_argument("--path", default=None, help="plik docelowy (domyslnie workspace/twin.json)")
    p.add_argument("--force", action="store_true", help="nadpisz istniejacy plik stanowiska")
    p.add_argument("--robot", choices=tuple(DEMO_WORKSPACES), default="so101",
                   help="ktore ramie (domyslnie so101)")
    p.add_argument("--ant", action="store_true",
                   help="do tego mrowka RealAnt z tagami w symulacji (zakladka Mrowka: kalibracja i WSAD)")
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
    p.add_argument("--robot", choices=tuple(DEMO_WORKSPACES), default="so101",
                   help="ktore ramie (domyslnie so101)")
    p.add_argument("-v", "--verbose", action="store_true")
    p.set_defaults(fn=_calib_sim)

    p = sub.add_parser("workspace", help="co wiadomo o stanowisku")
    p.add_argument("--path", default=None)
    p.set_defaults(fn=_workspace)

    p = sub.add_parser("train", help="naucz polityke PPO na GPU (MuJoCo Warp)")
    p.add_argument("--task", choices=("reach", "lift", "pickplace"), default="reach")
    p.add_argument("--envs", type=int, default=4096, help="ile swiatow naraz")
    p.add_argument("--iters", type=int, default=300)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--name", default=None, help="nazwa przebiegu (domyslnie zadanie + data)")
    p.add_argument("--out", default=None, help="katalog polityk (domyslnie workspace/policies)")
    p.add_argument("--workspace", default=None, help="plik stanowiska (domyslnie workspace/twin.json)")
    p.add_argument("--no-rand", action="store_true", help="bez randomizacji dziedziny")
    p.add_argument("--spread", type=float, default=1.0, help="szerokosc randomizacji wokol zmierzonej dynamiki")
    p.add_argument("--eval-episodes", type=int, default=50)
    p.add_argument("--stop-file", default=None, help="przerwij trening, gdy ten plik sie pojawi")
    p.add_argument("--init", default=None, help="douczanie: start z tej polityki (policy.pt)")
    p.set_defaults(fn=_train)

    p = sub.add_parser("eval", help="ewaluacja polityki na CPU (zwykle MuJoCo)")
    p.add_argument("policy", help="sciezka do policy.pt")
    p.add_argument("--episodes", type=int, default=50)
    p.add_argument("--rand", action="store_true", help="z randomizacja dziedziny")
    p.add_argument("--workspace", default=None)
    p.set_defaults(fn=_eval)

    p = sub.add_parser("policies", help="zapisane polityki i ich wyniki")
    p.add_argument("--dir", default=None)
    p.set_defaults(fn=_policies)

    p = sub.add_parser("ui", help="panel blizniaka w przegladarce (viser)")
    # Panel rusza prawdziwym ramieniem i nie ma hasla, a viser dzieli stan GUI miedzy
    # wszystkich podlaczonych - kazdy, kto dosiegnie portu, moze nacisnac "Polacz" albo
    # uruchomic polityke. Dlatego domyslnie tylko ten komputer; siec na wyrazne zyczenie.
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
    p.set_defaults(fn=lambda a: _ui(a))

    p = sub.add_parser("specs", help="okulary Spectacles: pudelko zasiegu, pinch -> ruch ramienia (przez przekaznik)")
    p.add_argument("--relay", default="ws://127.0.0.1:8080/twin", help="adres przekaznika (server.js), sciezka /twin")
    p.add_argument("--backend", default=None,
                   help="ramie: sim, a1x, feetech, ... (domyslnie sim w dry/sim, stanowisko w real)")
    p.add_argument("--port", default=None, help="port ramienia (domyslnie ze stanowiska)")
    p.add_argument("--mode", choices=("dry", "sim", "real"), default="dry",
                   help="dry - tylko sprawdzenia i nakladka; sim - rusza ramieniem w symulacji; real - prawdziwym")
    p.add_argument("--grasp", action="store_true", help="cel obiektu: zejscie, chwyt i powrot (inaczej tylko nad obiekt)")
    p.add_argument("--trusted-only", action="store_true",
                   help="przyjmuj cele tylko przy zaufanej rejestracji okularow (domyslnie kazda rejestracja wystarczy)")
    p.add_argument("--i-know", action="store_true", help="wymagane w --mode real: rozumiem, ze ramie sie ruszy")
    p.add_argument("--max-jump-deg", type=float, default=150.0,
                   help="odrzuc cel, gdy jakis staw mialby sie obrocic o wiecej w jednym ruchu (domyslnie 150)")
    p.add_argument("--hold-off", type=float, default=1.0,
                   help="w --mode real: ile sekund odczekac po przyjeciu celu z reki, zanim ramie ruszy (domyslnie 1; 0 = od razu)")
    p.add_argument("--start-center", action="store_true",
                   help="przed startem etapami (10 st./s, straznik uderzen) nad srodek pudelka; w real po Enter")
    p.add_argument("--follow-speed", type=float, default=None,
                   help="sledzenie dloni (teleop): najwieksza predkosc punktu chwytaka w m/s, przyspieszenie 4x tyle "
                        "na sekunde (domyslnie 0.20 w sim, 0.10 w real)")
    p.add_argument("--follow-floor", type=float, default=0.01,
                   help="sledzenie dloni: punkt chwytaka co najmniej tyle metrow nad dnem pudelka (domyslnie 0.01)")
    p.add_argument("--seconds", type=float, default=None, help="zakoncz po tylu sekundach (testy)")
    p.add_argument("--workspace", default=None, help="plik stanowiska (domyslnie workspace/twin.json)")
    p.set_defaults(fn=_specs)
    return ap


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
