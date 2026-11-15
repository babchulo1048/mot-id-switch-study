"""Match tracks to new boxes: IoU table + the Hungarian method."""
import numpy as np
from scipy.optimize import linear_sum_assignment

from tracker.iou import iou


def iou_matrix(boxes_a, boxes_b):
    """IoU of every box in boxes_a with every box in boxes_b (tlwh). Returns an (N, M) array."""
    matrix = np.zeros((len(boxes_a), len(boxes_b)))
    for i, box_a in enumerate(boxes_a):
        for j, box_b in enumerate(boxes_b):
            matrix[i, j] = iou(box_a, box_b)
    return matrix


def match(track_boxes, det_boxes, iou_threshold):
    """Give each track at most one box, and each box at most one track.

    Returns (matches, unmatched_tracks, unmatched_dets).
    matches is a list of (track_index, det_index) pairs.
    """
    if len(track_boxes) == 0 or len(det_boxes) == 0:
        return [], list(range(len(track_boxes))), list(range(len(det_boxes)))

    ious = iou_matrix(track_boxes, det_boxes)
    rows, cols = linear_sum_assignment(ious, maximize=True)

    matches = [(int(r), int(c)) for r, c in zip(rows, cols) if ious[r, c] >= iou_threshold]
    matched_tracks = {t for t, _ in matches}
    matched_dets = {d for _, d in matches}
    unmatched_tracks = [t for t in range(len(track_boxes)) if t not in matched_tracks]
    unmatched_dets = [d for d in range(len(det_boxes)) if d not in matched_dets]
    return matches, unmatched_tracks, unmatched_dets
