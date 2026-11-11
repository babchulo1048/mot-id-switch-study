import pytest

from tracker.iou import iou

A = [0, 0, 10, 10]  # tlwh: area 100


def test_same_box():
    assert iou(A, [0, 0, 10, 10]) == pytest.approx(1.0)


def test_half_to_the_right():
    # overlap 5 x 10 = 50, union 100 + 100 - 50 = 150
    assert iou(A, [5, 0, 10, 10]) == pytest.approx(1 / 3)


def test_diagonal():
    # overlap 5 x 5 = 25, union 100 + 100 - 25 = 175
    assert iou(A, [5, 5, 10, 10]) == pytest.approx(1 / 7)


def test_small_box_inside():
    # overlap = the small box = 16, union = the big box = 100
    assert iou(A, [2, 2, 4, 4]) == pytest.approx(0.16)


def test_touching_edges():
    # they share a line, but no area
    assert iou(A, [10, 0, 10, 10]) == 0.0


def test_far_away():
    assert iou(A, [20, 20, 5, 5]) == 0.0


def test_far_away_diagonal_no_fake_overlap():
    # width -10 and height -10 would multiply to +100 without max(0, ...)
    assert iou(A, [20, 20, 10, 10]) == 0.0


def test_both_zero_size():
    assert iou([0, 0, 0, 0], [0, 0, 0, 0]) == 0.0


def test_symmetry():
    b = [3, 4, 10, 8]
    assert iou(A, b) == pytest.approx(iou(b, A))
