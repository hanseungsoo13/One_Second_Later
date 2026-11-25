import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from osl.config import validate_osl_config


class ConfigValidationTests(unittest.TestCase):
    def test_pose_model_requires_pose_cache_configuration(self):
        config = {'data': {'size': [384, 640]}, 'training': {},
                  'model': {'variant': 'full_bbox_pose2', 'person_crop': { 'size': [224, 224] }}}
        with self.assertRaisesRegex(ValueError, 'model.pose'):
            validate_osl_config(config)

    def test_semantic_model_requires_grid_aligned_size(self):
        config = {'data': {'size': [383, 640]}, 'training': {},
                  'model': {'variant': 'full_bbox_pose_seg', 'person_crop': {'size': [224, 224]},
                            'pose': {'cache': {'train': 'a', 'val': 'b'}},
                            'semantic': {'model': 'a', 'revision': 'b', 'cache': 'c'}}}
        with self.assertRaisesRegex(ValueError, 'divisible by 16'):
            validate_osl_config(config)

class CucpsConfigValidationTests(unittest.TestCase):
    def test_pose_requires_its_own_checkpoint(self):
        from osl.config import validate_cucps_config
        cfg = {'motion': {'checkpoint': '/base.pt', 'pose': {}}, 'temporal_prior': {}}
        with self.assertRaisesRegex(ValueError, 'motion.pose_checkpoint'):
            validate_cucps_config(cfg, 'full_bbox_pose2')

    def test_segmentation_requires_semantic_model(self):
        from osl.config import validate_cucps_config
        cfg = {'motion': {'pose_checkpoint': '/pose.pt', 'pose': {}, 'person_crop': {'size': [224, 224]}}, 'temporal_prior': {}}
        with self.assertRaisesRegex(ValueError, 'motion.semantic'):
            validate_cucps_config(cfg, 'full_bbox_pose_seg')

    def test_baseline_needs_no_motion_checkpoint(self):
        from osl.config import validate_cucps_config
        self.assertIsNone(validate_cucps_config({}, 'baseline'))
