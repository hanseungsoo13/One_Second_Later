import importlib.util
import tempfile
import unittest
from pathlib import Path

spec = importlib.util.spec_from_file_location('build_personpath22', Path(__file__).resolve().parents[1] / 'scripts/build_personpath22.py')
builder = importlib.util.module_from_spec(spec)
spec.loader.exec_module(builder)


class PreprocessingTests(unittest.TestCase):
    def test_existing_output_is_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            with self.assertRaises(FileExistsError):
                builder.prepare_output(Path(temp))

    def test_selected_ids_have_no_server_paths(self):
        ids = builder.load_sequence_ids(Path(__file__).resolve().parents[1] / 'data/personpath22_ground_view_ids.txt')
        self.assertEqual(len(ids), 68)
        self.assertEqual(len(set(ids)), 68)
        self.assertTrue(all(value.startswith('uid_vid_') and '/' not in value for value in ids))


class PairingTests(unittest.TestCase):
    @staticmethod
    def entity(frame, time, identity=1, labels=None):
        return {'id': identity, 'time': time, 'blob': {'frame_idx': frame},
                'bb': [10, 20, 30, 40], 'labels': labels if labels is not None else {'person': 1}}

    def test_selects_nearest_one_second_within_same_track(self):
        rows = [self.entity(0, 0), self.entity(24, 1001), self.entity(25, 1040),
                self.entity(24, 1000, identity=2)]
        self.assertEqual([(a['blob']['frame_idx'], b['blob']['frame_idx'])
                          for a, b in builder.make_pairs(rows)], [(0, 24)])

    def test_duplicate_track_frame_fails(self):
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            builder.make_pairs([self.entity(0, 0), self.entity(0, 0)])

class DatasetBuildTests(unittest.TestCase):
    def test_builds_images_and_training_records_from_video(self):
        import json
        import cv2
        import numpy as np
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            video = root / 'sequence.avi'
            writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*'MJPG'), 25.0, (32, 32))
            if not writer.isOpened():
                self.skipTest('OpenCV MJPG encoder is unavailable')
            for _ in range(50):
                writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
            writer.release()
            entities = [{'id': 1, 'time': frame * 40, 'blob': {'frame_idx': frame},
                         'bb': [2, 3, 10, 15], 'labels': {'person': 1}}
                        for frame in (0, 25)]
            annotation = root / 'annotation.json'
            annotation.write_text(json.dumps({'metadata': {'resolution': {'width': 32, 'height': 32},
                                                              'number_of_frames': 50, 'fps': 25},
                                               'entities': entities}))
            output = root / 'output'
            builder.prepare_output(output)
            result = builder.process({'video_id': 'uid_vid_00001', 'video_path': str(video),
                                      'amodal_annotation_path': str(annotation)}, output)
            self.assertEqual(result['train'], 1)
            record = json.loads((output / 'uid_vid_00001/train.jsonl').read_text())
            self.assertEqual(record['target_frame'], 25)
            self.assertEqual(record['target_bottom_center_xy'], [7.0, 18.0])
            self.assertTrue(Path(record['image_path']).is_file())
