import unittest
import numpy as np
import torch

from osl.cucps_temporal import build_temporal_graph, refine_temporal_positives
from cucps.experiment import _attach_temporal_graph


class TemporalGraphTests(unittest.TestCase):
    def test_mutual_matching_rejects_ambiguous_and_low_appearance_pairs(self):
        memory = torch.tensor([[1., 0.], [0., 1.], [.99, .01], [.01, .99]])
        rows = [([2, 3], [.9, .1]), ([2, 3], [.1, .9]), ([], []), ([], [])]
        graph = build_temporal_graph(rows, memory, .5, .5, .7, .1)
        self.assertEqual(set(graph[0]), {2})
        self.assertEqual(set(graph[1]), {3})
        self.assertEqual(set(graph[2]), {0})
        ambiguous = [([2, 3], [.9, .9]), ([2, 3], [.9, .9]), ([], []), ([], [])]
        graph = build_temporal_graph(ambiguous, torch.ones(4, 2), .5, .5, .7, .1)
        self.assertFalse(any(graph))

    def test_propagates_both_sides_preserving_original_and_scene_uniqueness(self):
        # 0--1--2 and 3--4; 2 and 5 share a scene. 1--3 is an original positive.
        graph = [{1: .9}, {0: .9, 2: .8}, {1: .8}, {4: .9}, {3: .9}, {}]
        memory = torch.tensor([[1., 0.]] * 6)
        scenes = torch.tensor([0, 1, 2, 3, 4, 2])
        targets = torch.tensor([0])
        # Original 0--5 blocks the proposed 0--2, but must survive.
        def predict(mem, ids):
            result = torch.zeros(len(ids), 6)
            for row, i in enumerate(ids.tolist()):
                result[row, i] = 1
                if i == 0: result[row, 5] = 1
                if i == 1: result[row, 3] = 1
            return result
        result = refine_temporal_positives(memory, targets, graph, scenes, predict,
                                           hops=2, alpha=1., min_confidence=.5,
                                           min_appearance=.5)
        self.assertEqual(result[0, 5], 1)
        self.assertEqual(result[0, 2], 0)
        self.assertEqual(result[0, 3], 1)
        self.assertEqual(result[0, 4], 1)


class GraphIntegrationTests(unittest.TestCase):
    def test_wrapper_keeps_original_mining_result_and_expands_it(self):
        class Clustering:
            cnt2snum = torch.tensor([0, 1, 2])
            def predict(self, memory, targets):
                result = torch.zeros(len(targets), 3)
                for row, target in enumerate(targets.tolist()):
                    result[row, target] = 1
                    if target == 1:
                        result[row, 2] = 1
                return result
        clustering = Clustering()
        config = dict(location_weight=.5, min_edge_appearance=.5,
                      min_edge_combined=.5, min_edge_margin=.1, hops=1,
                      alpha=1., min_path_confidence=.5, min_positive_appearance=.5)
        _attach_temporal_graph(clustering, [([1], [1.]), ([], []), ([], [])], config)
        result = clustering.predict(torch.tensor([[1., 0.]] * 3), torch.tensor([0]))
        self.assertEqual(result.tolist(), [[1., 1., 1.]])
