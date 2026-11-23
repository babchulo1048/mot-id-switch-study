"""Stage 1 (SORT idea): predict each track with a Kalman filter, then match the PREDICTED box by IoU."""
from tracker.kalman import KalmanBox
from tracker.matching import match


class Track:
    def __init__(self, track_id, box, frame):
        self.id = track_id
        self.kf = KalmanBox(box)
        self.last_frame = frame   # last frame where this track got a box


class SortTracker:
    def __init__(self, iou_threshold=0.3, max_age=1):
        self.iou_threshold = iou_threshold
        self.max_age = max_age    # frames a track may stay unmatched before it is deleted
        self.tracks = []
        self.next_id = 1

    def update(self, frame, det_boxes):
        """det_boxes: (N, 4) tlwh boxes of this frame.

        Returns [(track_id, box), ...] for every track that has a box in this frame.
        """
        # 0. NEW in stage 1: every track guesses where its person is now
        predicted = [t.kf.predict() for t in self.tracks]
        matches, _, unmatched_dets = match(predicted, det_boxes, self.iou_threshold)

        output = []
        # 1. matched: correct the guess with the detector's box
        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            track.kf.update(det_boxes[d_idx])
            track.last_frame = frame
            output.append((track.id, det_boxes[d_idx]))

        # 2. unmatched boxes: a new person gets a new ID
        for d_idx in unmatched_dets:
            track = Track(self.next_id, det_boxes[d_idx], frame)
            self.next_id += 1
            self.tracks.append(track)
            output.append((track.id, det_boxes[d_idx]))

        # 3. delete tracks that have been unmatched for too long
        #    (while they wait, predict() keeps moving them, so they can be found further on)
        self.tracks = [t for t in self.tracks if frame - t.last_frame <= self.max_age]
        return output
