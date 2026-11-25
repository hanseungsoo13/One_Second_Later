import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from osl.temporal_search import location_scores,rank_candidates,search_metrics,select_weight

class TemporalSearchTests(unittest.TestCase):
 def test_footpoints_and_padding(self):
  h=np.zeros((5,5),dtype=np.float32);h[3,2]=1
  scores=location_scores(h,np.array([[1,1,3,3],[0,0,1,1],[-10,0,-5,3]],float),[1,1],[0,0],(5,5),radius=0)
  self.assertAlmostEqual(scores[0],1);self.assertAlmostEqual(scores[1],0);self.assertEqual(scores[2],0)
 def test_zero_weight_preserves_order(self):
  np.testing.assert_array_equal(rank_candidates([.8,.7],[0,1],0),[0,1]);np.testing.assert_array_equal(rank_candidates([.8,.7],[0,1],.2),[1,0])
 def test_misses_in_denominator(self):
  rows=[{'appearance':[.7,.8], 'position':[1,0], 'positive':[1,0]}, {'appearance':[], 'position':[], 'positive':[]}]
  m=search_metrics(rows,0);self.assertEqual(m['queries'],2);self.assertEqual(m['rank1'],0);self.assertEqual(m['mrr'],.25);self.assertEqual(m['detector_recall'],.5)
  self.assertEqual(search_metrics(rows,.2)['rank1'],.5)
 def test_select_weight_on_validation(self):
  rows=[{'appearance':[.7,.8], 'position':[1,0], 'positive':[1,0]}]
  self.assertEqual(select_weight(rows,[0,.2])['weight'],.2)
if __name__=='__main__':unittest.main()
