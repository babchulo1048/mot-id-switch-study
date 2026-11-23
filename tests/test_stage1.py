import numpy as np

from tracker.stage0 import IoUTracker
from tracker.stage1 import SortTracker


def ids(output):
    return sorted(track_id for track_id, _ in output)


def walk(tracker, speed, frames):
    """One person walks right at `speed` px per frame. Returns the IDs seen in the last frame."""
    out = []
    for frame in range(1, frames + 1):
        out = tracker.update(frame, np.array([[speed * frame, 0, 10, 10]]))
    return ids(out)


def test_two_people_keep_their_ids():
    tracker = SortTracker()
    assert ids(tracker.update(1, np.array([[0, 0, 10, 10], [100, 0, 10, 10]]))) == [1, 2]
    out = tracker.update(2, np.array([[101, 0, 10, 10], [1, 0, 10, 10]]))
    assert {i: list(b) for i, b in out} == {1: [1, 0, 10, 10], 2: [101, 0, 10, 10]}


def test_fast_walker_stage0_loses_stage1_keeps():
    # 6 px per frame on a 10 px box: last box vs new box IoU = 4 / 16 = 0.25 < 0.3
    assert walk(IoUTracker(), speed=6, frames=10) != [1]     # stage 0: a new ID almost every frame
    tracker = SortTracker()
    tracker.update(1, np.array([[0, 0, 10, 10]]))
    tracker.update(2, np.array([[4, 0, 10, 10]]))            # a slower start, so the filter can learn the speed
    for frame in range(3, 11):
        out = tracker.update(frame, np.array([[4 + 6 * (frame - 2), 0, 10, 10]]))
    assert ids(out) == [1]                                    # stage 1: still ID 1


def test_hidden_person_found_further_on():
    # walks 3 px per frame, hidden in frames 7-8, comes back 9 px further on
    tracker = SortTracker(max_age=2)
    for frame in range(1, 7):
        tracker.update(frame, np.array([[3 * frame, 0, 10, 10]]))
    tracker.update(7, np.zeros((0, 4)))
    tracker.update(8, np.zeros((0, 4)))
    assert ids(tracker.update(9, np.array([[27, 0, 10, 10]]))) == [1]
