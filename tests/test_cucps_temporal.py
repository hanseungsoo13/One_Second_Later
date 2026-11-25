import unittest
from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import numpy as np
import torch
from PIL import Image

from osl.cucps_temporal import add_temporal_prior, next_frame_indices
from cucps_temporal_experiment import (
    _attach_temporal_prior, _resolve_device, _install_memory_compatibility,
    _load_prw_probes_utf8, _install_numpy_detection_compatibility,
    _predict_relative_poses,
    _exclusive_epoch_end,
)


class ModernMemoryLayerTests(unittest.TestCase):
    def test_forward_backward_preserves_memory_update_and_input_gradient(self):
        memory = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        inputs = torch.tensor([[0.6, 0.8]], requires_grad=True)
        targets = torch.tensor([0])

        class LegacyMemory:
            pass

        instance = LegacyMemory()
        instance.mem = memory
        _install_memory_compatibility(type('RegressorModule', (), {'Memory': LegacyMemory}))
        instance.forward = LegacyMemory.forward.__get__(instance, LegacyMemory)
        instance.forward(inputs, targets).sum().backward()

        self.assertTrue(torch.allclose(inputs.grad, torch.tensor([[1.0, 1.0]])))
        expected_memory = torch.tensor([[0.8, 0.4], [0.0, 1.0]])
        expected_memory[0] /= expected_memory[0].norm()
        self.assertTrue(torch.allclose(memory, expected_memory))


class DetectionMetricCompatibilityTests(unittest.TestCase):
    def test_converts_numpy_arrays_for_legacy_empty_list_check(self):
        def legacy_metric(dataset, detections):
            return [bool(detection != []) for detection in detections]

        evaluator = SimpleNamespace(detection_performance_calc=legacy_metric)
        _install_numpy_detection_compatibility(evaluator)
        detections = [np.ones((2, 5)), np.empty((0, 5))]

        self.assertEqual(evaluator.detection_performance_calc(None, detections), [True, False])


class ProbeLoaderCompatibilityTests(unittest.TestCase):
    def test_loads_official_crlf_probe_list_as_text(self):
        class Dataset:
            def __init__(self, root):
                self.root = root

            def _get_cam_id(self, im_name):
                return int(im_name[1])

        with TemporaryDirectory() as temp_dir:
            Path(temp_dir, 'query_info.txt').write_bytes(
                b'42 10 20 30 40 c1s3_016471\r\n'
            )
            probes = _load_prw_probes_utf8(Dataset(temp_dir))

        self.assertEqual(probes[0]['im_name'], 'c1s3_016471.jpg')
        self.assertEqual(probes[0]['boxes'].tolist(), [[10, 20, 40, 60]])
        self.assertEqual(probes[0]['gt_pids'].tolist(), [42])


class DeviceResolutionTests(unittest.TestCase):
    def test_unindexed_cuda_resolves_to_visible_device_zero(self):
        self.assertEqual(_resolve_device('cuda'), torch.device('cuda:0'))

    def test_explicit_device_index_is_preserved(self):
        self.assertEqual(_resolve_device('cuda:2'), torch.device('cuda:2'))
        self.assertEqual(_resolve_device('cpu'), torch.device('cpu'))

    def test_requested_epoch_count_matches_exclusive_official_loop(self):
        self.assertEqual(len(range(1, _exclusive_epoch_end(25, 1))), 25)
        self.assertEqual(len(range(19, _exclusive_epoch_end(25, 19))), 25)


class PosePriorInputTests(unittest.TestCase):
    def test_pose_estimator_receives_bgr_and_returns_box_relative_keypoints(self):
        class Estimator:
            def __call__(self, image, boxes):
                self.image = image
                self.boxes = boxes
                points = np.tile(np.array([[[20., 30.]]], dtype=np.float32), (1, 17, 1))
                scores = np.ones((1, 17), dtype=np.float32)
                return points, scores

        estimator = Estimator()
        image = Image.new('RGB', (50, 60), color=(10, 20, 30))
        boxes = np.array([[10., 10., 30., 50.]], dtype=np.float32)
        poses = _predict_relative_poses(estimator, image, boxes)

        self.assertEqual(estimator.image[0, 0].tolist(), [30, 20, 10])
        self.assertEqual(estimator.boxes, [[10.0, 10.0, 30.0, 50.0]])
        self.assertEqual(poses.shape, (1, 17, 3))
        self.assertTrue(np.allclose(poses[0, 0], [.5, .5, 1.]))


class AddTemporalPriorTests(unittest.TestCase):
    def test_adds_only_directed_prior_entries_for_each_query(self):
        similarities = torch.tensor([[0.4, 0.3, 0.2], [0.1, 0.2, 0.3]])
        query_ids = torch.tensor([0, 2])
        prior_rows = [([1, 2], [0.5, 0.25]), ([], []), ([0], [0.8])]

        actual = add_temporal_prior(similarities, query_ids, prior_rows, 0.1)

        expected = torch.tensor([[0.4, 0.35, 0.225], [0.18, 0.2, 0.3]])
        self.assertTrue(torch.allclose(actual, expected))
        self.assertTrue(torch.equal(similarities, torch.tensor([[0.4, 0.3, 0.2], [0.1, 0.2, 0.3]])))

    def test_zero_weight_leaves_similarity_unchanged(self):
        similarities = torch.tensor([[0.4, 0.3]])
        actual = add_temporal_prior(similarities, torch.tensor([0]), [([1], [1.0]), ([], [])], 0.0)
        self.assertTrue(torch.equal(actual, similarities))

    def test_next_frames_are_ordered_only_within_camera_segments(self):
        records = [
            {'im_name': 'c1s1_000050.jpg'},
            {'im_name': 'c2s1_000025.jpg'},
            {'im_name': 'c1s1_000025.jpg'},
            {'im_name': 'c1s2_000010.jpg'},
            {'im_name': 'c1s1_000100.jpg'},
        ]
        self.assertEqual(next_frame_indices(records), {2: 0, 0: 4})

    def test_hnm_receives_temporal_similarity_before_original_mining(self):
        class Clustering:
            use_hnm = True

            def hard_negative_mining(self, similarities, coapp, pids, memory, threshold):
                return similarities

        clustering = Clustering()
        _attach_temporal_prior(clustering, [([1], [0.5]), ([], [])], 0.1)
        similarities = torch.tensor([[0.2, 0.3]])

        actual = clustering.hard_negative_mining(
            similarities, torch.zeros_like(similarities), torch.tensor([0]),
            torch.zeros((2, 1)), 0.6,
        )

        self.assertTrue(torch.allclose(actual, torch.tensor([[0.2, 0.35]])))


if __name__ == '__main__':
    unittest.main()
