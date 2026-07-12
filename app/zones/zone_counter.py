import logging

import cv2
import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)


def _sign(p1, p2, p3):
    return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])


def _point_in_triangle(pt, v1, v2, v3):
    d1 = _sign(pt, v1, v2)
    d2 = _sign(pt, v2, v3)
    d3 = _sign(pt, v3, v1)
    has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
    has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
    return not (has_neg and has_pos)


class SimpleLineCounter:
    def __init__(self, start: sv.Point, end: sv.Point):
        self.start = start
        self.end = end
        self.tracker_state = {}
        self.in_count = 0
        self.out_count = 0

    def _side_of_line(self, point):
        return (self.end.x - self.start.x) * (point.y - self.start.y) - \
               (self.end.y - self.start.y) * (point.x - self.start.x)

    def trigger(self, detections: sv.Detections):
        if detections.tracker_id is None:
            return np.zeros(len(detections), dtype=bool), np.zeros(len(detections), dtype=bool)

        crossed_in = np.zeros(len(detections), dtype=bool)
        crossed_out = np.zeros(len(detections), dtype=bool)

        for i, tid in enumerate(detections.tracker_id):
            if tid is None:
                continue

            x1, y1, x2, y2 = detections.xyxy[i]
            center = sv.Point(int((x1 + x2) / 2), int((y1 + y2) / 2))
            side = self._side_of_line(center)

            if tid not in self.tracker_state:
                self.tracker_state[tid] = side
                continue

            prev_side = self.tracker_state[tid]
            if prev_side < 0 and side >= 0:
                self.in_count += 1
                crossed_in[i] = True
            elif prev_side >= 0 and side < 0:
                self.out_count += 1
                crossed_out[i] = True

            self.tracker_state[tid] = side

        return crossed_in, crossed_out


class ZoneCounter:
    """Free zone counting using supervision PolygonZone and custom line counter."""

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.dwell_times = {}
        self.zone_dwell = {}

        self.entry_zone = self._create_entry_zone()
        self.exit_zone = self._create_exit_zone()
        self.shelf_zone = self._create_shelf_zone()

        self.entry_counter = sv.PolygonZone(
            polygon=self.entry_zone,
            triggering_anchors=[sv.Position.CENTER],
        )
        self.exit_counter = sv.PolygonZone(
            polygon=self.exit_zone,
            triggering_anchors=[sv.Position.CENTER],
        )
        self.shelf_counter = sv.PolygonZone(
            polygon=self.shelf_zone,
            triggering_anchors=[sv.Position.CENTER],
        )

        self.entry_line = SimpleLineCounter(
            start=sv.Point(int(frame_width * 0.1), int(frame_height * 0.8)),
            end=sv.Point(int(frame_width * 0.9), int(frame_height * 0.8)),
        )

        self.entry_count = 0
        self.exit_count = 0
        self.shelf_count = 0

    def _create_entry_zone(self):
        w, h = self.frame_width, self.frame_height
        return np.array([
            [int(w * 0.05), int(h * 0.75)],
            [int(w * 0.95), int(h * 0.75)],
            [int(w * 0.95), int(h * 0.95)],
            [int(w * 0.05), int(h * 0.95)],
        ])

    def _create_exit_zone(self):
        w, h = self.frame_width, self.frame_height
        return np.array([
            [int(w * 0.05), int(h * 0.05)],
            [int(w * 0.95), int(h * 0.05)],
            [int(w * 0.95), int(h * 0.25)],
            [int(w * 0.05), int(h * 0.25)],
        ])

    def _create_shelf_zone(self):
        w, h = self.frame_width, self.frame_height
        return np.array([
            [int(w * 0.3), int(h * 0.3)],
            [int(w * 0.7), int(h * 0.3)],
            [int(w * 0.7), int(h * 0.7)],
            [int(w * 0.3), int(h * 0.7)],
        ])

    def update(self, detections: sv.Detections, frame_idx: int, fps: float) -> dict:
        person_mask = detections.class_id == 0
        person_detections = detections[person_mask]

        if len(person_detections) == 0:
            return self.get_stats()

        in_entry = self.entry_counter.trigger(detections=person_detections)
        in_exit = self.exit_counter.trigger(detections=person_detections)
        in_shelf = self.shelf_counter.trigger(detections=person_detections)

        self.entry_count = int(np.sum(in_entry))
        self.exit_count = int(np.sum(in_exit))
        self.shelf_count = int(np.sum(in_shelf))

        self.entry_line.trigger(detections=person_detections)

        current_time = frame_idx / fps
        for i, tid in enumerate(person_detections.tracker_id):
            if tid is None:
                continue
            if in_entry[i] or in_shelf[i]:
                if tid not in self.dwell_times:
                    self.dwell_times[tid] = current_time
            elif tid in self.dwell_times:
                dwell = current_time - self.dwell_times.pop(tid)
                self.zone_dwell["total"] = self.zone_dwell.get("total", 0) + dwell

        return self.get_stats()

    def annotate(self, frame: np.ndarray, detections: sv.Detections) -> np.ndarray:
        overlay = frame.copy()

        cv2.polylines(overlay, [self.entry_zone], True, (0, 255, 0), 2)
        cv2.polylines(overlay, [self.exit_zone], True, (0, 0, 255), 2)
        cv2.polylines(overlay, [self.shelf_zone], True, (255, 165, 0), 2)

        h, w = frame.shape[:2]
        cv2.line(overlay, (int(w * 0.1), int(h * 0.8)), (int(w * 0.9), int(h * 0.8)), (0, 255, 0), 2)

        cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, f"Entry: {self.entry_count}", (10, h - 10), font, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"Exit: {self.exit_count}", (10, 30), font, 0.6, (0, 0, 255), 2)
        cv2.putText(frame, f"Shelf: {self.shelf_count}", (10, h // 2), font, 0.6, (255, 165, 0), 2)

        return frame

    def get_stats(self) -> dict:
        avg_dwell = 0
        if self.entry_line.in_count > 0:
            avg_dwell = self.zone_dwell.get("total", 0) / self.entry_line.in_count
        return {
            "entry_zone_count": self.entry_count,
            "exit_zone_count": self.exit_count,
            "shelf_zone_count": self.shelf_count,
            "entry_line_count": self.entry_line.in_count,
            "exit_line_count": self.entry_line.out_count,
            "avg_dwell_seconds": round(avg_dwell, 1),
            "total_dwell_seconds": round(self.zone_dwell.get("total", 0), 1),
        }
