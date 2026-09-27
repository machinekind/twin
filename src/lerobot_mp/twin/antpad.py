"""Pulpit WSAD mrowki: maly serwer HTTP ze strona - klawisze i przyciski -> `AntDrive.keys`.

    pad = AntPad(drive, port=8091).start()              # http://<host>:8091/ - osobna karta albo ramka w panelu
    pad.html_embed()                                    # ramka do `gui.add_html` w panelu viser

Viser nie podaje zdarzen klawiatury, wiec klawisze zbiera strona pulpitu: W S A D (przod, tyl, lewo, prawo -
wzgledem widoku 3D albo mrowki, `AntDrive.frame`), Q E (obrot w miejscu), spacja - stop. Kamera 3D visera tez
jezdzi na WASD QE: pulpit przechwytuje te klawisze wczesniej (faza przechwytywania na window) i ich nie oddaje -
kamere przesuwaja wtedy strzalki i mysz. Strona wysyla stan wcisnietych klawiszy co 0,1 s, dopoki ktorys jest
wcisniety, i od razu po puszczeniu; petla mrowki traktuje go jak czuwak (`antdrive.DEADMAN`) - zamkniete okno,
zerwana siec albo uspiona karta = mrowka staje. Utrata fokusu (Alt+Tab) puszcza wszystkie klawisze.

Ramka w panelu (`html_embed`) jest z tego samego zrodla co panel (srcdoc), wiec slucha klawiszy calego panelu
(poza polami tekstowymi) i pyta ten serwer o adres z nazwy hosta panelu - dziala takze przy panelu zdalnym.
Serwer przyjmuje zapytania tylko o ten jeden stan; zadnych plikow, zadnych innych polecen.
"""

from __future__ import annotations

import html
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs

#: Klawisze, ktore pulpit przepuszcza (reszta ignorowana).
PAD_KEYS = "wsadqe"

_SCRIPT = r"""
const CODES = {KeyW: "w", KeyS: "s", KeyA: "a", KeyD: "d", KeyQ: "q", KeyE: "e"};   // event.code: kazdy uklad klawiatury
const held = [];                                     // kolejnosc wcisniecia - pierwszy wygrywa
let url = null, timer = null;
function base() {
  if (url !== null) return url;
  let host = location.hostname;
  try { if (!host && window.parent) host = window.parent.location.hostname; } catch (e) {}
  url = location.protocol.startsWith("http") && location.port == "__PORT__" ? "" : "http://" + (host || "127.0.0.1") + ":__PORT__";
  return url;
}
function send() {
  fetch(base() + "/keys", {method: "POST", mode: "no-cors", body: "k=" + held.join("")}).catch(() => {});
  paint();
}
function press(k) { if (!held.includes(k)) { held.push(k); send(); } if (!timer) timer = setInterval(send, 100); }
function release(k) { const i = held.indexOf(k); if (i >= 0) held.splice(i, 1); send(); if (!held.length && timer) { clearInterval(timer); timer = null; } }
function releaseAll() { held.length = 0; send(); if (timer) { clearInterval(timer); timer = null; } }
function typing(e) { const t = e.target; return t && (t.tagName === "INPUT" || t.tagName === "TEXTAREA" || t.tagName === "SELECT" || t.isContentEditable); }
// Faza przechwytywania na window: przed kamera 3D visera (slucha WASD QE na document) - klawisze mrowki jej nie ruszaja,
// strzalki zostaja dla kamery.
function onDown(e) { if (typing(e) || e.ctrlKey || e.metaKey || e.altKey) return;
  if (e.code === "Space") { releaseAll(); e.preventDefault(); e.stopImmediatePropagation(); return; }
  const k = CODES[e.code]; if (!k) return; e.preventDefault(); e.stopImmediatePropagation(); if (!e.repeat || !held.includes(k)) press(k); }
function onUp(e) { const k = CODES[e.code]; if (!k) return; e.stopImmediatePropagation(); release(k); }
function attach(win) {
  // Jeden zestaw nasluchow na okno - viser potrafi przemontowac ramke, a stare nasluchy zostalyby w panelu.
  if (win.__antpadOff) win.__antpadOff();
  win.addEventListener("keydown", onDown, true); win.addEventListener("keyup", onUp, true); win.addEventListener("blur", releaseAll);
  win.__antpadOff = () => { win.removeEventListener("keydown", onDown, true); win.removeEventListener("keyup", onUp, true);
                            win.removeEventListener("blur", releaseAll); releaseAll(); };
}
attach(window);
try { if (window.parent && window.parent !== window) attach(window.parent); } catch (e) {}
window.addEventListener("pagehide", releaseAll);
document.querySelectorAll("button[data-k]").forEach(b => {
  const k = b.dataset.k;
  b.addEventListener("pointerdown", e => { e.preventDefault(); b.setPointerCapture(e.pointerId); press(k); });
  b.addEventListener("pointerup", () => release(k)); b.addEventListener("pointercancel", () => release(k));
});
document.getElementById("stop").addEventListener("click", releaseAll);
function paint() { document.querySelectorAll("button[data-k]").forEach(b => b.classList.toggle("on", held.includes(b.dataset.k))); }
async function poll() {
  try { const r = await fetch(base() + "/state"); const s = await r.json();
    document.getElementById("st").textContent = s.text; document.getElementById("st").className = s.level; }
  catch (e) { document.getElementById("st").textContent = "brak polaczenia z pulpitem (port __PORT__)"; document.getElementById("st").className = "bad"; }
}
setInterval(poll, 300); poll();
"""

_PAGE = """<!doctype html><html lang="pl"><head><meta charset="utf-8"><title>Mrowka WSAD</title>
<meta name="viewport" content="width=device-width,initial-scale=1">
<style>
:root{color-scheme:light dark;--bg:#fff;--fg:#1d2330;--btn:#e9edf3;--on:#2f7de1;--ok:#1f8a4c;--bad:#c0392b;--mid:#8a6d1f}
@media (prefers-color-scheme:dark){:root{--bg:#1b1f27;--fg:#e8ecf2;--btn:#2b313c;--on:#4a90e2}}
body{margin:0;padding:8px;background:var(--bg);color:var(--fg);font:14px system-ui,sans-serif;user-select:none}
.pad{display:grid;grid-template-columns:repeat(3,56px);gap:6px;justify-content:center;margin:4px 0 8px}
button{height:48px;border:0;border-radius:8px;background:var(--btn);color:var(--fg);font-size:18px;font-weight:600;touch-action:none}
button.on{background:var(--on);color:#fff}#stop{grid-column:1/4;height:36px;font-size:14px;background:var(--bad);color:#fff}
#st{text-align:center;min-height:2.6em;font-size:13px}.ok{color:var(--ok)}.bad{color:var(--bad)}.mid{color:var(--mid)}
small{display:block;text-align:center;opacity:.7}
</style></head><body>
<div class="pad"><button data-k="q">Q</button><button data-k="w">W</button><button data-k="e">E</button>
<button data-k="a">A</button><button data-k="s">S</button><button data-k="d">D</button>
<button id="stop">stop (spacja)</button></div>
<div id="st">...</div><small>W S A D: od siebie, do siebie, w lewo, w prawo (ta sama polityka); Q E: obrot. Trzymaj klawisz. Kamera: strzalki i mysz.</small>
<script>__SCRIPT__</script></body></html>"""


def page(port: int) -> str:
    return _PAGE.replace("__SCRIPT__", _SCRIPT).replace("__PORT__", str(int(port)))


class AntPad:
    """Serwer pulpitu: POST /keys (stan klawiszy), GET /state (stan mrowki), GET / (strona)."""

    def __init__(self, drive, port: int = 8091, host: str = "0.0.0.0"):
        self.drive, self.port, self.host = drive, int(port), host
        self._server: ThreadingHTTPServer | None = None
        self._thread: threading.Thread | None = None

    def state(self) -> dict:
        st = self.drive.status
        level = "bad" if st.mode in ("STOP", "wylaczona") else ("ok" if st.mode in ("jazda", "obrot") else "mid")
        text = f"{st.mode}" + (f" - {st.direction}" if st.direction else "")
        if st.note:
            text += f": {st.note}"
        return {"text": text, "level": level}

    def start(self) -> AntPad:
        pad = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):                   # ciche - 10 zapytan na sekunde w logu panelu
                pass

            def _send(self, code: int, body: bytes, ctype: str) -> None:
                self.send_response(code)
                self.send_header("Content-Type", ctype)
                self.send_header("Access-Control-Allow-Origin", "*")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_GET(self):
                if self.path.split("?")[0] == "/state":
                    self._send(200, json.dumps(pad.state()).encode(), "application/json")
                elif self.path.split("?")[0] == "/":
                    self._send(200, page(pad.port).encode(), "text/html; charset=utf-8")
                else:
                    self._send(404, b"", "text/plain")

            def do_POST(self):
                if self.path.split("?")[0] != "/keys":
                    self._send(404, b"", "text/plain")
                    return
                n = min(int(self.headers.get("Content-Length") or 0), 256)
                body = self.rfile.read(n).decode("utf-8", "ignore")
                keys = parse_qs(body).get("k", [""])[0]
                pad.drive.keys("".join(k for k in keys.lower() if k in PAD_KEYS))
                self._send(204, b"", "text/plain")

        self._server = ThreadingHTTPServer((self.host, self.port), Handler)
        self.port = self._server.server_address[1]
        self._thread = threading.Thread(target=self._server.serve_forever, name="pulpit-mrowki", daemon=True)
        self._thread.start()
        return self

    def html_embed(self, height: int = 260) -> str:
        """Ramka do panelu viser: strona pulpitu w srcdoc (to samo zrodlo co panel - slucha jego klawiszy)."""
        doc = html.escape(page(self.port), quote=True)
        return (f'<iframe srcdoc="{doc}" style="width:100%;height:{int(height)}px;border:0;border-radius:8px" '
                f'title="Pulpit WSAD mrowki"></iframe>')

    def close(self) -> None:
        if self._server is not None:
            self._server.shutdown()
            self._server.server_close()
            self._server = None
