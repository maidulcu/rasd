import logging

import cv2
import numpy as np
import supervision as sv

logger = logging.getLogger(__name__)


class ZoneCounter:
    """Free zone counting using supervision PolygonZone and LineZone."""

    def __init__(self, frame_width: int, frame_height: int):
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.dwell_times = {}  # track_id -> entry_time
        self.zone_dwell = {}  # zone_name -> total_dwell_seconds

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

        self.entry_line = sv.LineZone(
            start=sv.Point(int(frame_width * 0.1), int(frame_height * 0.8)),
            end=sv.Point(int(frame_width * 0.9), int(frame_height * 0.8)),
        )
        self.exit_line = sv.LineZone(
            start=sv.Point(int(frame_width * 0.1), int(frame_height * 0.2)),
            end=sv.Point(int(frame_width * 0.9), int(frame_height * 0.2)),
        )

        self.zone_annotator = sv.PolygonZoneAnnotator(
            zone=self.entry_zone,
            color=sv.Color.GREEN,
            thickness=2,
            text_thickness=1,
            text_scale=0.5,
        )
        self.line_annotator = sv.LineZoneAnnotator(
            thickness=2,
            text_thickness=1,
            text_scale=0.5,
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
        import time
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

        crossing = self.entry_line.trigger(detections=person_detections)

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
        frame = self.zone_annotator.annotate(scene=frame, counter=self.entry_counter)
        frame = self.line_annotator.annotate(scene=frame, line=self.entry_line)

        overlay = frame.copy()
        cv2.polylines(overlay, [self.entry_zone], True, (0, 255, 0), 2)
        cv2.polylines(overlay, [self.exit_zone], True, (0, 0, 255), 2)
        cv2.polylines(overlay, [self.shelf_zone], True, (255, 165, 0), 2)
        cv2.addWeighted(overlay, 0.8, frame, 0.2, 0, frame)

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
