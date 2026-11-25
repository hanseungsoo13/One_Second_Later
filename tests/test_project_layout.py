import importlib
import subprocess
import sys
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT))


class ProjectLayoutTests(unittest.TestCase):
    def test_legacy_modules_share_implementation_with_new_packages(self):
        for legacy, canonical in (
            ('osl.cucps_temporal', 'integration.temporal'),
            ('osl.motion_inputs', 'integration.motion_inputs'),
            ('osl.cucps_compat', 'cucps.compat'),
        ):
            with self.subTest(legacy=legacy):
                self.assertIs(importlib.import_module(legacy), importlib.import_module(canonical))

    def test_new_entrypoints_offer_help(self):
        for script in ('scripts/osl/run.py', 'scripts/osl/cache_pose.py',
                       'scripts/osl/cache_semantic.py', 'scripts/osl/build_personpath22.py',
                       'scripts/cucps/experiment.py', 'experiments/prw_motion/evaluate_prw_motion.py',
                       'experiments/temporal_reranking/temporal_search.py'):
            with self.subTest(script=script):
                result = subprocess.run([sys.executable, str(ROOT / script), '--help'], cwd=ROOT,
                                        capture_output=True, text=True, timeout=60)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertIn('usage:', result.stdout)
