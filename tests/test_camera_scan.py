"""The camera scan finds every /dev/video node, not only the first few."""

from pathlib import Path

from machinekind_twin.cameras import video_indices


def test_every_video_node_is_scanned(tmp_path: Path):
    for name in ("video0", "video1", "video4", "video6", "video7", "video10", "videoX", "vchiq"):
        (tmp_path / name).touch()
    assert video_indices(tmp_path) == [0, 1, 4, 6, 7, 10]


def test_without_video_nodes_the_scan_tries_the_first_indices(tmp_path: Path):
    assert video_indices(tmp_path, max_index=6) == [0, 1, 2, 3, 4, 5]
