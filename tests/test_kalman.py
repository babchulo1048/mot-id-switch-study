import numpy as np
import pytest

from tracker.kalman import KalmanBox, cxcywh_to_tlwh, tlwh_to_cxcywh


def test_box_format_round_trip():
    # tlwh [100, 200, 50, 120] -> centre (125, 260)
    assert list(tlwh_to_cxcywh([100, 200, 50, 120])) == [125, 260, 50, 120]
    assert list(cxcywh_to_tlwh([125, 260, 50, 120])) == [100, 200, 50, 120]


def test_new_track_predicts_the_same_box():
    # no speed known yet: the first guess is "it stays where it is"
    kf = KalmanBox([100, 200, 50, 120])
    assert kf.predict() == pytest.approx([100, 200, 50, 120])


def test_update_lands_between_guess_and_detector():
    kf = KalmanBox([100, 200, 50, 120])
    kf.predict()                       # guess: left = 100
    kf.update([110, 200, 50, 120])     # detector: left = 110
    assert 100 < kf.box()[0] <= 110


def test_learns_the_speed():
    # a person walks right 5 px per frame; after 10 frames the filter has learned it
    kf = KalmanBox([100, 200, 50, 120])
    for step in range(1, 11):
        kf.predict()
        kf.update([100 + 5 * step, 200, 50, 120])
    assert kf.x[4] == pytest.approx(5, abs=0.5)               # vx about 5
    assert kf.predict()[0] == pytest.approx(155, abs=1)       # next guess: 150 + 5


def test_keeps_moving_while_hidden_and_gets_less_sure():
    kf = KalmanBox([100, 200, 50, 120])
    for step in range(1, 11):
        kf.predict()
        kf.update([100 + 5 * step, 200, 50, 120])
    sure = kf.P[0, 0]
    for _ in range(3):                 # hidden behind the shelf for 3 frames: predict only
        box = kf.predict()
    assert box[0] == pytest.approx(165, abs=1.5)              # 150 + 3 x 5
    assert kf.P[0, 0] > sure                                  # less sure than before
