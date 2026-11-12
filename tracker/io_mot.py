"""Read MOT17 files and write tracker results in the MOTChallenge format."""
import configparser

import numpy as np


def load_detections(path, min_score=None):
    """Read a det.txt file.

    Returns a dict {frame: array of shape (N, 5)}, one row per box:
    [left, top, width, height, score]. Frames without boxes are missing.
    """
    data = np.loadtxt(path, delimiter=",", ndmin=2)
    detections = {}
    for row in data:
        frame = int(row[0])
        score = row[6]
        if min_score is not None and score < min_score:
            continue
        detections.setdefault(frame, []).append(row[2:7])
    return {frame: np.array(rows) for frame, rows in detections.items()}


def read_sequence_length(seqinfo_path):
    """Number of frames in the video, read from seqinfo.ini."""
    config = configparser.ConfigParser()
    config.read(seqinfo_path)
    return int(config["Sequence"]["seqLength"])


def write_results(path, rows):
    """Write tracker output. rows: list of (frame, track_id, left, top, width, height)."""
    with open(path, "w") as f:
        for frame, track_id, left, top, width, height in rows:
            f.write(f"{frame},{track_id},{left:.2f},{top:.2f},{width:.2f},{height:.2f},1,-1,-1,-1\n")
