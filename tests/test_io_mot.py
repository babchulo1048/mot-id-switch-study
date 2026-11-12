import pytest

from tracker.io_mot import load_detections, read_sequence_length, write_results

DET_TEXT = (
    "2,-1,10,20,30,40,0.9\n"   # rows are NOT sorted by frame in real det.txt files
    "1,-1,1,2,3,4,0.5\n"
    "2,-1,50,60,70,80,0.2\n"
)


def test_load_detections_groups_rows_by_frame(tmp_path):
    path = tmp_path / "det.txt"
    path.write_text(DET_TEXT)
    dets = load_detections(path)
    assert sorted(dets) == [1, 2]
    assert dets[2].shape == (2, 5)
    assert list(dets[1][0]) == pytest.approx([1, 2, 3, 4, 0.5])


def test_load_detections_drops_low_scores(tmp_path):
    path = tmp_path / "det.txt"
    path.write_text(DET_TEXT)
    dets = load_detections(path, min_score=0.5)
    assert dets[2].shape == (1, 5)   # the 0.2 box is gone
    assert dets[1].shape == (1, 5)   # 0.5 is kept: the rule is "score >= min_score"


def test_read_sequence_length(tmp_path):
    path = tmp_path / "seqinfo.ini"
    path.write_text("[Sequence]\nname=X\nframeRate=30\nseqLength=525\n")
    assert read_sequence_length(path) == 525


def test_write_results_uses_mot_format(tmp_path):
    path = tmp_path / "res.txt"
    write_results(path, [(1, 7, 10, 20, 30.5, 40)])
    assert path.read_text() == "1,7,10.00,20.00,30.50,40.00,1,-1,-1,-1\n"
