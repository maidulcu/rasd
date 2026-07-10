import logging

logger = logging.getLogger(__name__)

VALUABLE_CLASSES = {
    24: "backpack",
    26: "handbag",
    28: "suitcase",
    67: "cell phone",
    62: "laptop",
    39: "bottle",
    73: "book",
    44: "knife",
    45: "spoon",
    46: "bowl",
    47: "banana",
    76: "scissors",
    77: "teddy bear",
}


def iou(box1, box2):
    x1 = max(box1[0], box2[0])
    y1 = max(box1[1], box2[1])
    x2 = min(box1[2], box2[2])
    y2 = min(box1[3], box2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    union = area1 + area2 - inter
    return inter / union if union > 0 else 0


class ObjectTrack:
    def __init__(self, obj_id, bbox, label, class_id):
        self.obj_id = obj_id
        self.bbox = bbox
        self.label = label
        self.class_id = class_id
        self.misses = 0
        self.unattended_frames = 0
        self.last_near_person = False
        self.person_interacted = None
        self.concealed = False


class TheftDetector:
    def __init__(self, iou_threshold=0.3, max_misses=5,
                 unattended_threshold=30, overlap_threshold=0.3):
        self.iou_threshold = iou_threshold
        self.max_misses = max_misses
        self.unattended_threshold = unattended_threshold
        self.overlap_threshold = overlap_threshold
        self.tracks = {}
        self.next_id = 1
        self.unattended_alerts = []
        self.theft_alerts = []
        self.current_interactions = []

    def is_valuable(self, class_id):
        return class_id in VALUABLE_CLASSES

    def get_valuable_label(self, class_id):
        return VALUABLE_CLASSES.get(class_id, f"class_{class_id}")

    def update(self, all_detections, people_boxes, person_ids=None):
        valuable_detections = [
            det for det in all_detections if self.is_valuable(det.get("class_id", -1))
        ]

        matched = set()
        for track_id, track in list(self.tracks.items()):
            if track.concealed:
                continue
            best_iou = self.iou_threshold
            best_det = None
            best_det_idx = None
            for i, det in enumerate(valuable_detections):
                if i in matched:
                    continue
                if self.get_valuable_label(det.get("class_id", -1)) != track.label:
                    continue
                score = iou(track.bbox, det["bbox"])
                if score > best_iou:
                    best_iou = score
                    best_det = det
                    best_det_idx = i
            if best_det is not None:
                track.bbox = best_det["bbox"]
                track.misses = 0
                matched.add(best_det_idx)
            else:
                track.misses += 1

        for i, det in enumerate(valuable_detections):
            if i not in matched:
                label = self.get_valuable_label(det.get("class_id", -1))
                self.tracks[self.next_id] = ObjectTrack(
                    self.next_id, det["bbox"], label, det.get("class_id", -1)
                )
                self.next_id += 1

        self.current_interactions = []

        for track_id in list(self.tracks.keys()):
            track = self.tracks[track_id]

            if track.concealed:
                if track.misses >= self.max_misses * 2:
                    del self.tracks[track_id]
                else:
                    track.misses += 1
                continue

            if track.misses >= self.max_misses:
                if track.person_interacted is not None:
                    self.theft_alerts.append({
                        "track_id": track.obj_id,
                        "label": track.label,
                        "bbox": track.bbox,
                        "type": "concealed",
                        "person_location": track.bbox,
                        "person_id": track.person_interacted,
                    })
                    track.concealed = True
                    track.misses = 0
                    logger.warning("THEFT: %s concealed by person %s (track %d)",
                                   track.label, track.person_interacted, track.obj_id)
                else:
                    del self.tracks[track_id]
                continue

            cx = (track.bbox[0] + track.bbox[2]) / 2
            cy = (track.bbox[1] + track.bbox[3]) / 2
            near_person = False
            interacting_person_id = None

            for pi, pb in enumerate(people_boxes):
                cnt = ((track.bbox[0] + track.bbox[2]) // 2,
                       (track.bbox[1] + track.bbox[3]) // 2)
                pt_in = pb[0] <= cx <= pb[2] and pb[1] <= cy <= pb[3]
                box_overlap = iou(track.bbox, pb) > self.overlap_threshold

                if pt_in or box_overlap:
                    near_person = True
                    pid = person_ids[pi] if person_ids and pi < len(person_ids) else None
                    if box_overlap:
                        interacting_person_id = pid or f"person_{pi}"
                    break

            if interacting_person_id is not None:
                track.person_interacted = interacting_person_id
                self.current_interactions.append({
                    "track_id": track.obj_id,
                    "label": track.label,
                    "bbox": track.bbox,
                    "person_id": interacting_person_id,
                })

            if near_person:
                track.unattended_frames = 0
            else:
                track.unattended_frames += 1
                if track.unattended_frames == self.unattended_threshold:
                    self.unattended_alerts.append({
                        "track_id": track.obj_id,
                        "label": track.label,
                        "bbox": track.bbox,
                        "frames": track.unattended_frames,
                    })
                    logger.info("Unattended %s (track %d)", track.label, track.obj_id)

            track.last_near_person = near_person

    def get_results(self):
        return {
            "unattended_objects": self.unattended_alerts,
            "theft_alerts": self.theft_alerts,
            "current_interactions": self.current_interactions,
        }
