"""The grower on synthetic images: the colour mask, its blobs, and the blob nearest a pixel."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from machinekind_twin_seeded.grow import Blob, assign, blobs, box_distance, colour_mask, nearest, part, split
from machinekind_twin_seeded.settings import COLOURS, parse_hsv

W, H = 640, 480
RED = (200, 30, 30)
#: The red ranges, which wrap around H = 0.
RED_RANGES = parse_hsv(f"red={COLOURS['red']}")["red"]


def image(*squares: tuple[int, int, int, int]) -> np.ndarray:
    """A grey RGB image with red squares (u, v, width, height)."""
    img = np.full((H, W, 3), 120, np.uint8)
    for u, v, w, h in squares:
        img[v:v + h, u:u + w] = RED
    return img


def found(img: np.ndarray, min_area_px: int = 30) -> tuple[np.ndarray, list[Blob]]:
    return blobs(colour_mask(cv2.cvtColor(img, cv2.COLOR_RGB2HSV), RED_RANGES), min_area_px)


def test_a_red_square_is_one_blob_with_its_centroid_box_and_area():
    labels, (blob,) = found(image((300, 200, 40, 40)))

    assert blob.centroid == pytest.approx((319.5, 219.5), abs=0.5)
    assert blob.box == (300, 200, 339, 239)
    assert blob.area == 1600
    assert blob.side == 40
    assert int((labels == blob.index).sum()) == 1600
    assert blob.index != 0                       # the background is component 0 and never a blob


def test_the_colour_mask_is_uint8_and_misses_the_grey():
    mask = colour_mask(cv2.cvtColor(image((300, 200, 40, 40)), cv2.COLOR_RGB2HSV), RED_RANGES)

    assert mask.dtype == np.uint8 and mask.shape == (H, W)
    assert int((mask > 0).sum()) == 1600


def test_the_nearest_blob_within_the_radius_wins_and_a_far_one_is_ignored():
    _, blobs_ = found(image((100, 200, 40, 40), (290, 200, 40, 40)))
    first, second = sorted(blobs_, key=lambda b: b.box[0])
    anchor = (100 - 20.0, 219.5)                 # 20 px left of the first square

    assert nearest(blobs_, anchor, 30.0) == first
    assert nearest(blobs_, anchor, 10.0) is None
    assert nearest(blobs_, (300.0, 210.0), 10.0) == second


def test_a_tie_goes_to_the_larger_blob():
    _, blobs_ = found(image((100, 200, 20, 20), (200, 200, 60, 60)))

    small, large = sorted(blobs_, key=lambda b: b.area)
    assert nearest(blobs_, (159.5, 210.0), 50.0) == large   # both 40.5 px away
    assert small.area < large.area


def test_a_blob_too_large_for_one_block_is_rejected():
    _, (merged,) = found(image((300, 200, 40, 40), (340, 200, 40, 40)))   # two touching squares
    seed_side = 40

    assert merged.box == (300, 200, 379, 239) and merged.side == 80
    assert nearest([merged], (320.0, 220.0), 10.0, max_side_px=2.0 * seed_side) == merged
    assert nearest([merged], (320.0, 220.0), 10.0, max_side_px=1.5 * seed_side) is None


def test_specks_below_min_area_px_are_not_blobs():
    img = image()
    for u in (100, 200, 300, 400):
        img[100:103, u:u + 3] = RED
    hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, np.array(RED_RANGES[0][0], np.uint8), np.array(RED_RANGES[0][1], np.uint8))

    assert blobs(mask, 30)[1] == []
    assert len(blobs(mask, 5)[1]) == 4


def test_the_opening_removes_specks_from_the_colour_mask():
    img = image()
    img[100:102, 100:102] = RED

    assert found(img, 1)[1] == []


def test_box_distance_is_zero_inside_and_euclidean_outside():
    box = (10.0, 20.0, 30.0, 40.0)

    assert box_distance((15.0, 25.0), box) == 0.0
    assert box_distance((0.0, 30.0), box) == pytest.approx(10.0)
    assert box_distance((33.0, 44.0), box) == pytest.approx(5.0)


def test_a_blob_of_two_touching_squares_splits_into_one_part_per_square():
    labels, (blob,) = found(image((300, 200, 40, 40), (340, 200, 40, 40)))
    mask = labels == blob.index

    (left, right), centres = split(mask, [(305.0, 230.0), (350.0, 210.0)])

    assert not (left & right).any() and np.array_equal(left | right, mask)
    assert left[:, 300:340].sum() == right[:, 340:380].sum() == 40 * 40
    assert part(left) == ((319.5, 219.5), (300, 200, 339, 239), 40)
    assert centres == [pytest.approx((319.5, 219.5)), pytest.approx((359.5, 219.5))]
    assert part(np.zeros_like(mask)) is None


def test_a_flat_block_on_another_is_cut_across_not_along_its_length():
    # A long low blob: one flat block's top and front over the front of the one below it.
    labels, (blob,) = found(image((300, 200, 80, 40)))
    mask = labels == blob.index

    (top, under), centres = split(mask, [(338.0, 214.0), (342.0, 226.0)])

    assert top[200:207, 300:380].all() and under[234:240, 300:380].all()
    assert centres == [(338.0, 214.0), (342.0, 226.0)]


def test_a_free_centre_moves_along_the_line_from_the_one_that_stays():
    # The lower block's track had the whole blob to itself: it starts at the blob's centroid.
    labels, (blob,) = found(image((300, 200, 80, 40)))
    mask = labels == blob.index

    (top, under), (kept, moved) = split(mask, [(339.0, 207.0), (341.0, 220.0)], free=[False, True])

    assert kept == (339.0, 207.0) and moved[1] > 224.0
    assert top[200:210, 300:380].all() and under[232:240, 300:380].all()


def test_assign_pairs_the_cheapest_way_within_the_gate():
    assert assign(np.array([[1.0, 2.0], [1.5, 10.0]]), 20.0) == [(0, 1), (1, 0)]
    assert assign(np.array([[0.2], [0.1]]), 1.0) == [(1, 0)]
    assert assign(np.array([[5.0, 6.0]]), 1.0) == []
