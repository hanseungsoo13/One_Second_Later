import unittest

from experiments.prw_motion.evaluate_prw_motion import gap_group, summarize


class PRWMotionEvaluationTests(unittest.TestCase):
    def test_gap_boundaries(self):
        self.assertEqual([gap_group(x, [25, 50]) for x in (25, 26, 50, 51)],
                         ['1-25', '26-50', '26-50', '>50'])

    def test_same_samples_moving_and_height_normalization(self):
        rows = [
            {'frame_gap': 25, 'height_px': 100, 'displacement_px': 10,
             'errors': {'stationary': 10, 'pose_p2': 5}},
            {'frame_gap': 50, 'height_px': 50, 'displacement_px': 4,
             'errors': {'stationary': 4, 'pose_p2': 5}},
        ]
        result = summarize(rows, [25, 50])
        self.assertEqual(result['overall']['count'], 2)
        self.assertEqual(result['overall']['stationary']['fde_px'], 7)
        self.assertAlmostEqual(result['overall']['pose_p2']['fde_height'], .075)
        self.assertEqual(result['moving_0.1']['count'], 1)
        self.assertEqual(result['gap/1-25']['pose_p2']['fde_px'], 5)


if __name__ == '__main__':
    unittest.main()

class ModelNameTests(unittest.TestCase):
    def test_accepts_new_names_and_rejects_duplicate_aliases(self):
        from experiments.prw_motion.evaluate_prw_motion import resolve_evaluation_models
        models = resolve_evaluation_models(['full_bbox_c', 'full_bbox_pose_seg'])
        self.assertEqual([model.name for model in models.values()], ['full_bbox_c', 'full_bbox_pose_seg'])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            resolve_evaluation_models(['e2c', 'full_bbox_c'])
