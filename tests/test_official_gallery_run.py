import sys,tempfile,unittest
from pathlib import Path
from unittest.mock import patch
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from experiments.temporal_reranking import official_gallery_search as app

class OfficialRunTests(unittest.TestCase):
 def test_fresh_run_extracts_then_compares(self):
  with tempfile.TemporaryDirectory() as d:
   cfg={'output_root':d,'fusion':{'weight':.05}}
   with patch.object(app,'extract') as extract,patch.object(app,'compare') as compare:
    app.run(cfg);extract.assert_called_once_with(cfg);compare.assert_called_once_with(cfg)
 def test_existing_cache_skips_extraction(self):
  with tempfile.TemporaryDirectory() as d:
   Path(d,'official_features.pkl').touch();cfg={'output_root':d,'fusion':{'weight':.05}}
   with patch.object(app,'extract') as extract,patch.object(app,'compare') as compare:
    app.run(cfg);extract.assert_not_called();compare.assert_called_once_with(cfg)
 def test_existing_results_and_invalid_weight_fail_before_work(self):
  with tempfile.TemporaryDirectory() as d:
   Path(d,'official_gallery_metrics.json').touch();cfg={'output_root':d,'fusion':{'weight':.05}}
   with patch.object(app,'extract') as extract,patch.object(app,'compare') as compare:
    with self.assertRaises(FileExistsError):app.run(cfg)
    extract.assert_not_called();compare.assert_not_called()
  with tempfile.TemporaryDirectory() as d:
   with self.assertRaises(ValueError):app.run({'output_root':d})
if __name__=='__main__':unittest.main()
