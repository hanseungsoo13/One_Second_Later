import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from osl.variants import resolve_model_variant


class VariantTests(unittest.TestCase):
    def test_legacy_names_resolve_to_explicit_model_features(self):
        expected = {
            'e2a': ('full_bbox_a', False, False, None, False),
            'e2b': ('full_bbox_b', True, False, None, False),
            'e2c': ('full_bbox_c', True, True, None, False),
            'e2c_p1': ('full_bbox_pose1', True, True, 'p1', False),
            'e2c_p2': ('full_bbox_pose2', True, True, 'p2', False),
            'e2c_p2_s': ('full_bbox_pose_seg', True, True, 'p2', True),
        }
        for old, values in expected.items():
            with self.subTest(old=old):
                variant = resolve_model_variant(old)
                self.assertEqual((variant.name, variant.transformer, variant.person_branch,
                                  variant.pose_fusion, variant.semantic), values)
                self.assertEqual(resolve_model_variant(variant.name), variant)

    def test_unknown_name_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown model variant'):
            resolve_model_variant('unknown')

class CucpsVariantTests(unittest.TestCase):
    def test_cucps_names_resolve_to_model_features(self):
        from osl.variants import resolve_cucps_variant
        self.assertIsNone(resolve_cucps_variant('baseline'))
        self.assertEqual(resolve_cucps_variant('temporal').name, 'full_bbox_c')
        self.assertEqual(resolve_cucps_variant('pose_p1').name, 'full_bbox_pose1')
        self.assertEqual(resolve_cucps_variant('pose_p2').name, 'full_bbox_pose2')
        for name in ('full_bbox_a', 'full_bbox_b', 'full_bbox_c', 'full_bbox_pose1', 'full_bbox_pose2', 'full_bbox_pose_seg'):
            self.assertEqual(resolve_cucps_variant(name).name, name)
