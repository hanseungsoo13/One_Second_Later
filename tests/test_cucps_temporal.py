import unittest

import torch

from osl.cucps_temporal import add_temporal_prior, next_frame_indices
from cucps_temporal_experiment import (
    _attach_temporal_prior, _resolve_device, _install_memory_compatibility,
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


class DeviceResolutionTests(unittest.TestCase):
    def test_unindexed_cuda_resolves_to_visible_device_zero(self):
        self.assertEqual(_resolve_device('cuda'), torch.device('cuda:0'))

    def test_explicit_device_index_is_preserved(self):
        self.assertEqual(_resolve_device('cuda:2'), torch.device('cuda:2'))
        self.assertEqual(_resolve_device('cpu'), torch.device('cpu'))


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
