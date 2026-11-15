import pytest

from tracker.iou import iou
from tracker.matching import iou_matrix, match

LEFT = [0, 0, 10, 10]
RIGHT = [100, 0, 10, 10]


def test_iou_matrix_shape_and_values():
    a = [LEFT, RIGHT, [5, 0, 10, 10]]
    b = [[1, 0, 10, 10], [100, 0, 10, 10]]
    m = iou_matrix(a, b)
    assert m.shape == (3, 2)
    for i in range(3):
        for j in range(2):
            assert m[i, j] == pytest.approx(iou(a[i], b[j]))


def test_match_finds_the_right_pairs_in_any_order():
    tracks = [LEFT, RIGHT]
    dets = [[101, 0, 10, 10], [1, 0, 10, 10]]   # same people, listed in the other order
    matches, unmatched_tracks, unmatched_dets = match(tracks, dets, iou_threshold=0.3)
    assert sorted(matches) == [(0, 1), (1, 0)]
    assert unmatched_tracks == []
    assert unmatched_dets == []


def test_match_rejects_pairs_below_threshold():
    tracks = [LEFT]
    dets = [[8, 0, 10, 10]]   # IoU = 20 / 180 = 0.11
    matches, unmatched_tracks, unmatched_dets = match(tracks, dets, iou_threshold=0.3)
    assert matches == []
    assert unmatched_tracks == [0]
    assert unmatched_dets == [0]


def test_match_more_boxes_than_tracks():
    matches, unmatched_tracks, unmatched_dets = match([LEFT], [[1, 0, 10, 10], RIGHT], 0.3)
    assert matches == [(0, 0)]
    assert unmatched_tracks == []
    assert unmatched_dets == [1]


def test_match_with_no_tracks_or_no_boxes():
    assert match([], [LEFT], 0.3) == ([], [], [0])
    assert match([LEFT], [], 0.3) == ([], [0], [])
