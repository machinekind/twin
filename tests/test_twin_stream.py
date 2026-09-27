"""Podglad kamer po HTTP (`ui.stream.FrameServer`): lista, kadr, strumien MJPEG - bez panelu i bez kamer."""

from __future__ import annotations

import json
import socket
import time
import urllib.request

import numpy as np
import pytest

pytest.importorskip("cv2")

from lerobot_mp.twin.ui.stream import FrameServer  # noqa: E402


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.fixture
def server():
    state = {"t": 1.0}
    img = np.zeros((48, 64, 3), np.uint8)
    img[..., 0] = 200                                       # czerwony kadr RGB

    def frames():
        return {"brio": (img, state["t"]), "sim 1": (img[:24], state["t"] - 0.5)}

    srv = FrameServer(frames, host="127.0.0.1", port=free_port(), poll_s=0.005)
    srv.start()
    srv.state = state
    try:
        yield srv
    finally:
        srv.stop()


def get(url: str, timeout: float = 5.0):
    with urllib.request.urlopen(url, timeout=timeout) as r:
        return r.status, r.headers, r.read()


def test_lists_cameras_and_serves_the_last_frame_as_jpeg(server):
    status, headers, body = get(server.url + "cameras.json")
    assert status == 200
    cams = json.loads(body)["cameras"]
    assert [c["name"] for c in cams] == ["brio", "sim 1"]
    assert cams[0]["width"] == 64 and cams[0]["height"] == 48 and cams[0]["age_s"] >= 0

    status, headers, body = get(server.url + "brio.jpg")
    assert status == 200 and headers["Content-Type"] == "image/jpeg" and body[:2] == b"\xff\xd8"
    import cv2
    decoded = cv2.imdecode(np.frombuffer(body, np.uint8), cv2.IMREAD_COLOR)
    assert decoded.shape == (48, 64, 3) and decoded[0, 0, 2] > 150 and decoded[0, 0, 0] < 50   # BGR: czerwony wrocil

    status, headers, body = get(server.url + "sim%201.jpg")                                    # nazwa ze spacja
    assert status == 200 and body[:2] == b"\xff\xd8"

    with pytest.raises(urllib.error.HTTPError) as exc:
        get(server.url + "nie-ma.jpg")
    assert exc.value.code == 404

    status, headers, body = get(server.url)
    assert status == 200 and b"brio.mjpg" in body and b"sim 1.mjpg" in body


def test_mjpeg_stream_sends_a_part_per_new_frame_and_stops_on_close(server):
    with urllib.request.urlopen(server.url + "brio.mjpg", timeout=5.0) as r:
        assert r.headers["Content-Type"].startswith("multipart/x-mixed-replace")

        def part():
            assert r.readline().strip() == b"--kadr"
            headers = {}
            while (line := r.readline().strip()):
                k, v = line.decode().split(":", 1)
                headers[k.strip()] = v.strip()
            data = r.read(int(headers["Content-Length"]))
            assert r.readline() == b"\r\n"
            return data

        first = part()
        assert first[:2] == b"\xff\xd8"
        server.state["t"] = 2.0                             # nowy kadr -> nowa czesc
        second = part()
        assert second == first                               # ten sam obraz, ale drugi raz wyslany
        # bez nowego kadru nic nie przychodzi
        r.fp.raw._sock.settimeout(0.2)
        with pytest.raises(TimeoutError):
            r.readline()
    server.stop()                                            # drugi stop jest nieszkodliwy
    time.sleep(0.05)
