"""Stage 2 (ByteTrack idea): match the SURE boxes first, then give the UNSURE boxes a second chance."""
from tracker.matching import match
from tracker.stage1 import Track


class TwoRoundTracker:
    def __init__(self, iou_threshold=0.3, max_age=1, high_score=0.6, low_iou_threshold=0.5):
        self.iou_threshold = iou_threshold          # round 1 (sure boxes)
        self.low_iou_threshold = low_iou_threshold  # round 2 (unsure boxes): stricter, they are less trustworthy
        self.high_score = high_score                # a box is "sure" if its score >= high_score
        self.max_age = max_age
        self.tracks = []
        self.next_id = 1

    def update(self, frame, det_boxes, det_scores):
        """det_boxes: (N, 4) tlwh boxes, det_scores: (N,) detector scores of this frame.

        Returns [(track_id, box), ...] for every track that has a box in this frame.
        """
        high = [i for i in range(len(det_boxes)) if det_scores[i] >= self.high_score]
        low = [i for i in range(len(det_boxes)) if det_scores[i] < self.high_score]

        # 0. every track guesses where its person is now (same as stage 1)
        predicted = [t.kf.predict() for t in self.tracks]
        output = []

        # 1. ROUND 1: all tracks vs the SURE boxes
        matches, left_tracks, left_high = match(predicted, det_boxes[high], self.iou_threshold)
        for t_idx, j in matches:
            self._keep(self.tracks[t_idx], det_boxes[high[j]], frame, output)

        # 2. NEW in stage 2. ROUND 2: the tracks still without a box vs the UNSURE boxes
        left_predicted = [predicted[t] for t in left_tracks]
        matches2, _, _ = match(left_predicted, det_boxes[low], self.low_iou_threshold)
        for k, j in matches2:
            self._keep(self.tracks[left_tracks[k]], det_boxes[low[j]], frame, output)

        # 3. a SURE box with no track -> new ID.
        #    An unsure box with no track is thrown away (it is often a false box).
        for j in left_high:
            track = Track(self.next_id, det_boxes[high[j]], frame)
            self.next_id += 1
            self.tracks.append(track)
            output.append((track.id, det_boxes[high[j]]))

        # 4. a track with no box waits (and keeps moving) up to max_age, then is deleted
        self.tracks = [t for t in self.tracks if frame - t.last_frame <= self.max_age]
        return output

    @staticmethod
    def _keep(track, box, frame, output):
        """Matched: correct the guess with the box and keep the ID."""
        track.kf.update(box)
        track.last_frame = frame
        output.append((track.id, box))
