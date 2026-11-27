import numpy as np

from tracker.stage2 import TwoRoundTracker


def ids(output):
    return sorted(track_id for track_id, _ in output)


def test_sure_box_with_no_track_gets_new_id():
    tracker = TwoRoundTracker()
    assert ids(tracker.update(1, np.array([[0, 0, 10, 10]]), np.array([0.9]))) == [1]


def test_unsure_box_with_no_track_is_thrown_away():
    # score 0.3: maybe a false box, so it may not start a new track
    tracker = TwoRoundTracker()
    assert tracker.update(1, np.array([[0, 0, 10, 10]]), np.array([0.3])) == []


def test_half_hidden_person_keeps_id_with_unsure_box():
    # a person walks behind a shelf: the detector is less sure (0.9 -> 0.4), but the box is still there
    tracker = TwoRoundTracker()
    for frame in range(1, 4):
        tracker.update(frame, np.array([[frame, 0, 10, 10]]), np.array([0.9]))
    out = tracker.update(4, np.array([[4, 0, 10, 10]]), np.array([0.4]))
    assert ids(out) == [1]                                      # round 2 kept ID 1


def test_sure_boxes_are_matched_first():
    # one track, two boxes at the same place: the sure one (0.9) wins round 1
    tracker = TwoRoundTracker()
    tracker.update(1, np.array([[0, 0, 10, 10]]), np.array([0.9]))
    out = tracker.update(2, np.array([[1, 0, 10, 10], [0, 0, 10, 10]]), np.array([0.9, 0.4]))
    assert {i: list(b) for i, b in out} == {1: [1, 0, 10, 10]}  # the unsure box is not used


def test_round_two_is_stricter():
    # IoU 0.43 is enough for a sure box (>= 0.3) but not for an unsure one (< 0.5)
    tracker = TwoRoundTracker()
    tracker.update(1, np.array([[0, 0, 10, 10]]), np.array([0.9]))
    assert tracker.update(2, np.array([[4, 0, 10, 10]]), np.array([0.4])) == []
