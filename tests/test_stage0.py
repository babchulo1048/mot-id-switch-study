import numpy as np

from tracker.stage0 import IoUTracker


def ids(output):
    return sorted(track_id for track_id, _ in output)


def test_slow_person_keeps_one_id():
    tracker = IoUTracker()
    for frame in range(1, 6):
        out = tracker.update(frame, np.array([[frame, 0, 10, 10]]))   # moves 1 px per frame
        assert ids(out) == [1]


def test_two_people_get_two_ids_and_keep_them():
    tracker = IoUTracker()
    assert ids(tracker.update(1, np.array([[0, 0, 10, 10], [100, 0, 10, 10]]))) == [1, 2]
    out = tracker.update(2, np.array([[101, 0, 10, 10], [1, 0, 10, 10]]))
    assert {i: list(b) for i, b in out} == {1: [1, 0, 10, 10], 2: [101, 0, 10, 10]}


def test_fast_jump_gets_a_new_id():
    # no motion prediction: a 9 px jump leaves IoU = 10 / 190 < 0.3, so stage 0 loses the person
    tracker = IoUTracker()
    tracker.update(1, np.array([[0, 0, 10, 10]]))
    assert ids(tracker.update(2, np.array([[9, 0, 10, 10]]))) == [2]


def test_missed_frame_within_max_age_keeps_id():
    tracker = IoUTracker(max_age=1)
    tracker.update(1, np.array([[0, 0, 10, 10]]))
    assert tracker.update(2, np.zeros((0, 4))) == []      # detector missed the person
    assert ids(tracker.update(3, np.array([[1, 0, 10, 10]]))) == [1]


def test_track_deleted_after_max_age():
    tracker = IoUTracker(max_age=1)
    tracker.update(1, np.array([[0, 0, 10, 10]]))
    tracker.update(2, np.zeros((0, 4)))
    tracker.update(3, np.zeros((0, 4)))                    # 2 frames missed > max_age
    assert ids(tracker.update(4, np.array([[1, 0, 10, 10]]))) == [2]
