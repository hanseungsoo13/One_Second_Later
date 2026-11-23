import sqlite3, sys, tempfile, unittest
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from osl.pose import load_cache, pose_key, relative_pose

class PoseCacheTests(unittest.TestCase):
    def test_relative_coordinates_and_cache_identity(self):
        record = dict(sequence='s', frame=1, person_id=3, image_path='a', image_size=[100, 80], bbox_xywh=[10, 20, 20, 40])
        points = np.tile([20., 40.], (17, 1))
        pose = relative_pose(points, np.ones(17), record['bbox_xywh'])
        np.testing.assert_allclose(pose[0], [.5, .5, 1.])
        key = pose_key(record)
        self.assertNotEqual(key, pose_key({**record, 'bbox_xywh': [10, 20, 21, 40]}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'pose.sqlite'
            with sqlite3.connect(path) as db:
                db.execute('CREATE TABLE poses (key TEXT PRIMARY KEY, pose BLOB)')
                db.execute('INSERT INTO poses VALUES (?, ?)', (key, pose.astype('<f4').tobytes()))
            np.testing.assert_allclose(load_cache(path)[key], pose)
            with self.assertRaises(sqlite3.OperationalError): load_cache(Path(tmp) / 'missing.sqlite')
    def test_reject_invalid_box(self):
        with self.assertRaises(ValueError): relative_pose(np.zeros((17, 2)), np.ones(17), [0, 0, 0, 2])
