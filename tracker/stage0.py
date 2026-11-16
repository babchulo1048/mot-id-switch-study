"""Stage 0: match each track's LAST box to the new boxes by IoU. No motion prediction."""
from tracker.matching import match


class Track:
    def __init__(self, track_id, box, frame):
        self.id = track_id
        self.box = box            # last matched box, tlwh
        self.last_frame = frame   # last frame where this track got a box


class IoUTracker:
    def __init__(self, iou_threshold=0.3, max_age=1):
        self.iou_threshold = iou_threshold
        self.max_age = max_age    # frames a track may stay unmatched before it is deleted
        self.tracks = []
        self.next_id = 1

    def update(self, frame, det_boxes):
        """det_boxes: (N, 4) tlwh boxes of this frame.

        Returns [(track_id, box), ...] for every track that has a box in this frame.
        """
        track_boxes = [t.box for t in self.tracks]
        matches, _, unmatched_dets = match(track_boxes, det_boxes, self.iou_threshold)

        output = []
        # 1. matched: same person, new position
        for t_idx, d_idx in matches:
            track = self.tracks[t_idx]
            track.box = det_boxes[d_idx]
            track.last_frame = frame
            output.append((track.id, track.box))

        # 2. unmatched boxes: a new person gets a new ID
        for d_idx in unmatched_dets:
            track = Track(self.next_id, det_boxes[d_idx], frame)
            self.next_id += 1
            self.tracks.append(track)
            output.append((track.id, track.box))

        # 3. delete tracks that have been unmatched for too long
        self.tracks = [t for t in self.tracks if frame - t.last_frame <= self.max_age]
        return output
