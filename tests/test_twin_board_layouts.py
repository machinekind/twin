"""The ChArUco layouts the lens step knows: each sheet is found by its own board and by no other."""

from __future__ import annotations

import pytest

cv2 = pytest.importorskip("cv2")

from machinekind_twin.calib.intrinsics import DEFAULT_LAYOUT, LAYOUTS, Board, Collector
from machinekind_twin.cli import main


def _sheet(name: str, dpi: int = 120):
    """The printable sheet of layout `name` as a camera would see it, head on."""
    return Board.of(name, 0.028).image(dpi=dpi)


@pytest.mark.parametrize("name", list(LAYOUTS))
def test_each_layout_is_found_by_its_own_board(name):
    sheet = _sheet(name)
    board = Board.of(name, 0.028)
    lay = LAYOUTS[name]
    assert board.squares == lay.squares and board.dictionary == lay.dictionary
    assert board.marker == pytest.approx(lay.marker_ratio * 0.028)
    det = Collector(board, sheet.shape[1::-1]).detect(sheet)
    assert det is not None
    corners = (lay.squares[0] - 1) * (lay.squares[1] - 1)
    assert len(det[0]) == corners, f"{name}: {len(det[0])} of {corners} corners"


def test_a_layout_is_not_found_by_another_board():
    """The old 7x5 print on the desk showed as 'board not visible' to a 9x6 detector: no frame was ever taken."""
    names = list(LAYOUTS)
    for name in names:
        other = next(n for n in names if n != name)
        sheet = _sheet(name)
        assert Collector(Board.of(other, 0.028), sheet.shape[1::-1]).detect(sheet) is None


def test_default_board_is_the_twins_sheet():
    assert DEFAULT_LAYOUT == "9x6"
    assert Board() == Board.of("9x6", 0.028)
    with pytest.raises(ValueError, match="nieznana tablica"):
        Board.of("8x8", 0.028)


def test_board_command_takes_a_layout(tmp_path, capsys):
    out = tmp_path / "board.png"
    assert main(["board", "--layout", "7x5", "--square-mm", "30", "--dpi", "120", "--out", str(out)]) == 0
    sheet = cv2.imread(str(out), cv2.IMREAD_GRAYSCALE)
    assert sheet is not None
    det = Collector(Board.of("7x5", 0.030), sheet.shape[1::-1]).detect(sheet)
    assert det is not None and len(det[0]) == 24
    assert Collector(Board.of("9x6", 0.030), sheet.shape[1::-1]).detect(sheet) is None
    assert "30.0 mm" in capsys.readouterr().out
