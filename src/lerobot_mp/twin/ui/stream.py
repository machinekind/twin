"""Podglad kamer panelu po HTTP na osobnym porcie: MJPEG i pojedyncze kadry, tylko do odczytu.

    server = FrameServer(app.stream_frames, host="0.0.0.0", port=8766)
    server.start()                                   # http://<maszyna>:8766/

Kamery (V4L2) otwiera tylko jeden proces, wiec drugi proces nie moze ich
podgladac - stad serwer w procesie panelu, na kadrach, ktore panel i tak juz pobiera
(`TwinApp.frames`, co ok. 0.2 s). Nic tu nie steruje ramieniem ani panelem, dlatego
moze sluchac w calej sieci, inaczej niz sam panel (`--host`).

Adresy:
    /                  strona ze wszystkimi kamerami (obrazki MJPEG)
    /cameras.json      lista kamer: nazwa, rozmiar, wiek kadru [s]
    /<kamera>.jpg      ostatni kadr jako JPEG (404 bez kadru)
    /<kamera>.mjpg     strumien multipart/x-mixed-replace, nowy kadr gdy panel ma nowszy
"""

from __future__ import annotations

import html
import json
import logging
import threading
import time
from collections.abc import Callable
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import unquote

import cv2
import numpy as np

logger = logging.getLogger(__name__)

#: Kadry: nazwa -> (obraz RGB, chwila wykonania wg time.monotonic).
Frames = Callable[[], dict[str, tuple[np.ndarray, float]]]

_BOUNDARY = "kadr"


class FrameServer:
    def __init__(self, frames: Frames, host: str = "0.0.0.0", port: int = 8766,
                 quality: int = 80, poll_s: float = 0.03):
        self.frames = frames
        self.host, self.port = host, port
        self.quality = int(quality)
        self.poll_s = float(poll_s)
        self._jpeg: dict[str, tuple[float, bytes]] = {}       # nazwa -> (chwila kadru, JPEG) - jeden encode na kadr
        self._lock = threading.Lock()
        self._httpd: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    # ------------------------------------------------------------------ cykl zycia
    def start(self) -> None:
        server = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, fmt, *args):            # zadania HTTP nie sa wydarzeniami panelu
                logger.debug("%s " + fmt, self.address_string(), *args)

            def do_GET(self):
                server._handle(self)

            def do_HEAD(self):
                server._handle(self, head=True)

        class Server(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True

        self._httpd = Server((self.host, self.port), Handler)
        self.port = self._httpd.server_address[1]
        self._thread = threading.Thread(target=self._httpd.serve_forever, name="kamery-http", daemon=True)
        self._thread.start()
        logger.info("Podglad kamer: %s", self.url)

    def stop(self) -> None:
        if self._httpd is not None:
            self._httpd.shutdown()
            self._httpd.server_close()
            self._httpd = None

    @property
    def url(self) -> str:
        host = "localhost" if self.host in ("0.0.0.0", "127.0.0.1", "") else self.host
        return f"http://{host}:{self.port}/"

    # ------------------------------------------------------------------ kadry
    def _snapshot(self) -> dict[str, tuple[np.ndarray, float]]:
        try:
            return dict(self.frames())
        except Exception:                                  # noqa: BLE001 - brak kadrow to pusta strona, nie 500
            logger.exception("Podglad kamer: kadry")
            return {}

    def jpeg(self, name: str) -> tuple[float, bytes] | None:
        """(chwila kadru, JPEG) kamery albo None bez kadru; ten sam kadr koduje sie raz dla wszystkich klientow."""
        got = self._snapshot().get(name)
        if got is None:
            return None
        img, t = got
        with self._lock:
            cached = self._jpeg.get(name)
            if cached is not None and cached[0] == t:
                return cached
        ok, buf = cv2.imencode(".jpg", cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2BGR),
                               [int(cv2.IMWRITE_JPEG_QUALITY), self.quality])
        if not ok:
            return None
        out = (t, buf.tobytes())
        with self._lock:
            self._jpeg[name] = out
        return out

    def cameras(self) -> list[dict]:
        now = time.monotonic()
        return [dict(name=n, width=int(img.shape[1]), height=int(img.shape[0]), age_s=round(max(0.0, now - t), 3))
                for n, (img, t) in sorted(self._snapshot().items())]

    # ------------------------------------------------------------------ HTTP
    def _handle(self, h: BaseHTTPRequestHandler, head: bool = False) -> None:
        path = unquote(h.path.split("?", 1)[0])
        try:
            if path in ("/", "/index.html"):
                self._send(h, 200, "text/html; charset=utf-8", self._index().encode(), head)
            elif path == "/cameras.json":
                body = json.dumps({"cameras": self.cameras()}, ensure_ascii=False).encode()
                self._send(h, 200, "application/json; charset=utf-8", body, head)
            elif path.endswith(".jpg"):
                got = self.jpeg(path[1:-4])
                if got is None:
                    self._send(h, 404, "text/plain; charset=utf-8", b"brak kadru", head)
                else:
                    self._send(h, 200, "image/jpeg", got[1], head)
            elif path.endswith(".mjpg"):
                if head:
                    self._send(h, 200, f"multipart/x-mixed-replace; boundary={_BOUNDARY}", b"", True)
                else:
                    self._stream(h, path[1:-5])
            else:
                self._send(h, 404, "text/plain; charset=utf-8", b"nie ma takiej strony", head)
        except (BrokenPipeError, ConnectionResetError, TimeoutError):
            pass                                            # klient zamknal karte - zwykla sprawa

    @staticmethod
    def _send(h: BaseHTTPRequestHandler, code: int, ctype: str, body: bytes, head: bool) -> None:
        h.send_response(code)
        h.send_header("Content-Type", ctype)
        h.send_header("Content-Length", str(len(body)))
        h.send_header("Cache-Control", "no-store")
        h.send_header("Access-Control-Allow-Origin", "*")
        h.end_headers()
        if not head:
            h.wfile.write(body)

    def _stream(self, h: BaseHTTPRequestHandler, name: str) -> None:
        """Kolejne kadry, poki klient slucha; bez kadru czeka (kamera moze dopiero wstawac)."""
        h.send_response(200)
        h.send_header("Content-Type", f"multipart/x-mixed-replace; boundary={_BOUNDARY}")
        h.send_header("Cache-Control", "no-store")
        h.send_header("Access-Control-Allow-Origin", "*")
        h.end_headers()
        last = None
        while self._httpd is not None:
            got = self.jpeg(name)
            if got is None or got[0] == last:
                time.sleep(self.poll_s)
                continue
            last, data = got
            h.wfile.write(f"--{_BOUNDARY}\r\nContent-Type: image/jpeg\r\nContent-Length: {len(data)}\r\n\r\n".encode())
            h.wfile.write(data)
            h.wfile.write(b"\r\n")
            h.wfile.flush()

    def _index(self) -> str:
        cams = self.cameras()
        if cams:
            items = "".join(
                f'<figure><img src="/{html.escape(c["name"])}.mjpg" alt="{html.escape(c["name"])}">'
                f'<figcaption>{html.escape(c["name"])} &middot; {c["width"]}&times;{c["height"]}</figcaption></figure>'
                for c in cams)
        else:
            items = "<p>Brak kadrow - panel nie ma jeszcze zadnej wlaczonej kamery z obrazem. Odswiez za chwile.</p>"
        return ("<!doctype html><html lang=\"pl\"><head><meta charset=\"utf-8\"><title>Kamery blizniaka</title>"
                "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">"
                "<style>body{margin:0;padding:12px;background:#111;color:#ddd;font:14px system-ui,sans-serif}"
                "main{display:flex;flex-wrap:wrap;gap:12px}figure{margin:0}img{max-width:100%;height:auto;"
                "display:block;background:#000}figcaption{padding:4px 0;color:#aaa}</style></head>"
                f"<body><main>{items}</main></body></html>")
