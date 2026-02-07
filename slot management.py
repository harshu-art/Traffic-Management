import cv2
import json
import numpy as np

with open("slots.json") as f:
    PARKING_SLOTS = {int(k): v for k, v in json.load(f).items()}

def bbox_center(box):
    x1, y1, x2, y2 = box
    return int((x1 + x2)/2), int((y1 + y2)/2)

def point_inside_polygon(point, polygon):
    return cv2.pointPolygonTest(
        np.array(polygon, dtype=np.int32),
        point,
        False
    ) >= 0

def check_slots(detections):
    status = {sid: "EMPTY" for sid in PARKING_SLOTS}

    for det in detections:
        x1, y1, x2, y2 = det
        cx, cy = bbox_center((x1, y1, x2, y2))

        for sid, poly in PARKING_SLOTS.items():
            if point_inside_polygon((cx, cy), poly):
                status[sid] = "FULL"

    return status
