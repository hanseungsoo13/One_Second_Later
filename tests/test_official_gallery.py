import builtins,sys,unittest
from pathlib import Path
from types import SimpleNamespace
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from osl.official_gallery import next_frame_map,prepare_gallery,evaluate_queries,fixed_fusion_weight

class OfficialGalleryTests(unittest.TestCase):
 def test_next_only_same_segment_and_no_skipping(self):
  all_names=['c1s1_000025.jpg','c1s1_000050.jpg','c1s1_000100.jpg','c1s2_000075.jpg','c2s1_000050.jpg']
  got=next_frame_map(all_names,{'c1s1_000025.jpg','c1s1_000100.jpg','c1s2_000075.jpg','c2s1_000050.jpg'})
  self.assertNotIn('c1s1_000025.jpg',got)
  self.assertNotIn('c1s2_000075.jpg',got)
  self.assertEqual(got['c1s1_000050.jpg'],'c1s1_000100.jpg')
 def test_only_one_next_frame_receives_bias(self):
  records,det,feat,queries,qfeat=fixture();cache=prepare_gallery(records,det,feat,.9,True)
  base=evaluate_queries(records,cache,queries,qfeat,{},0)
  zero=evaluate_queries(records,cache,queries,qfeat,{0:('c1s1_000050.jpg',np.array([0.,1.,0.,0.,0.,0.]))},0)
  self.assertEqual(base['mAP'],zero['mAP']);self.assertEqual(base['top_k'],zero['top_k'])
  biased=evaluate_queries(records,cache,queries,qfeat,{0:('c1s1_000050.jpg',np.array([0.,1.,0.,0.,0.,0.]))},1.)
  self.assertNotEqual(base['mAP'],biased['mAP'])
 def test_fusion_weight_is_fixed_and_validated(self):
  self.assertEqual(fixed_fusion_weight({'fusion':{'weight':.05}}),.05)
  for cfg in ({},{'fusion':{'weight':float('nan')}},{'fusion':{'weight':-1}}):
   with self.assertRaises(ValueError):fixed_fusion_weight(cfg)
 def test_official_parity(self):
  if not (Path(__file__).resolve().parents[1]/'external/CUCPS_official/lib/datasets/prw.py').is_file():self.skipTest('Official CUCPS checkout not installed')
  sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'external/python-deps'))
  sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'external/CUCPS_official'))
  from lib.datasets import prw
  prw.xrange=range
  records,det,feat,queries,qfeat=fixture();gallery=SimpleNamespace(record=records,data_path='.',__len__=lambda self:len(records))
  class DS:
   def __init__(self,records):self.record=records;self.data_path='.'
   def __len__(self):return len(self.record)
  expected=prw.PRW.search_performance_calc(DS(records),DS(queries),det,feat,qfeat,det_thresh=.9,ignore_cam_id=True,remove_unlabel=True)
  actual=evaluate_queries(records,prepare_gallery(records,det,feat,.9,True),queries,qfeat,{},0)
  self.assertAlmostEqual(actual['mAP'],expected['mAP'],places=10);np.testing.assert_allclose(actual['top_k'],expected['accs'])

def fixture():
 records=[];dets=[];features=[]
 for n in range(3):
  name=f'c1s1_{25*(n+1):06}.jpg';boxes=np.array([[20+i*40,10,50+i*40,90] for i in range(6)],dtype=np.float32);ids=np.array([1,2,3,4,5,-2]);
  records.append({'im_name':name,'boxes':boxes,'gt_pids':ids,'cam_id':1});dets.append(np.column_stack([boxes,np.ones(6)]));
  features.append(np.array([[.9,.1],[.6,.4],[.1,.9],[.2,.8],[.3,.7],[.7,.3]],dtype=np.float32))
 queries=[{'im_name':'c1s1_000025.jpg','boxes':records[0]['boxes'][0:1],'gt_pids':np.array([1]),'cam_id':1}];qfeat=[np.array([[1.,0.]],dtype=np.float32)]
 return records,dets,features,queries,qfeat
if __name__=='__main__':unittest.main()
