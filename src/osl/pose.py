"""RTMPose cache keys and bounding-box-relative COCO keypoints."""
import hashlib
import json
import sqlite3
import numpy as np


def pose_key(record):
    fields = [record['sequence'], record['frame'], record['person_id'],
              record['image_path'], record['image_size'], record['bbox_xywh']]
    return hashlib.sha256(json.dumps(fields, separators=(',', ':')).encode()).hexdigest()


def relative_pose(points, scores, bbox):
    points = np.asarray(points, dtype=np.float32)
    scores = np.asarray(scores, dtype=np.float32)
    x, y, w, h = bbox
    if points.shape != (17, 2) or scores.shape != (17,) or w <= 0 or h <= 0:
        raise ValueError('Expected 17 keypoints and a positive bounding box')
    result = np.column_stack(((points[:, 0] - x) / w, (points[:, 1] - y) / h, scores))
    if not np.isfinite(result).all(): raise ValueError('Non-finite pose')
    return result.astype(np.float32)


def load_cache(path):
    # Read-only mode refuses to create a missing cache by accident.
    from pathlib import Path
    uri = Path(path).resolve().as_uri() + '?mode=ro'
    with sqlite3.connect(uri, uri=True) as db:
        rows = db.execute('SELECT key, pose FROM poses').fetchall()
    result = {}
    for key, blob in rows:
        pose = np.frombuffer(blob, dtype='<f4').copy()
        if pose.size != 51 or not np.isfinite(pose).all(): raise ValueError(f'Invalid cached pose: {key}')
        result[key] = pose.reshape(17, 3)
    return result
