import json, sys, tempfile, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from PIL import Image
from osl.data import AnnotationDataset

class AnnotationTests(unittest.TestCase):
    def test_uses_requested_jsonl_split_and_coordinates(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); seq=root/'MOT17-02-FRCNN'; seq.mkdir(); image=root/'frame.jpg'; Image.new('RGB',(100,50)).save(image)
            record={'sequence':seq.name,'person_id':1,'frame':1,'image_path':str(image),'image_size':[100,50],'bbox_xywh':[10,5,20,30],'target_bottom_center_xy':[50,40]}
            (seq/'train.jsonl').write_text(json.dumps(record)+'\n'+json.dumps(record)+'\n'); (seq/'val.jsonl').write_text(json.dumps(record)+'\n')
            ds=AnnotationDataset(root,[seq.name],'val',size=(64,64))
            self.assertEqual(len(ds),1); item=ds[0]
            self.assertEqual(item['original_target'].tolist(),[50,40]); self.assertAlmostEqual(item['target'][0].item(),32); self.assertAlmostEqual(item['target'][1].item(),41.6,places=4)
    def test_outside_target_is_clipped_only_for_heatmap(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); seq=root/'MOT17-02-FRCNN'; seq.mkdir(); image=root/'frame.jpg'; Image.new('RGB',(100,50)).save(image)
            record={'sequence':seq.name,'person_id':1,'frame':1,'image_path':str(image),'image_size':[100,50],'bbox_xywh':[10,5,20,30],'target_bottom_center_xy':[120,60]}
            (seq/'val.jsonl').write_text(json.dumps(record)+'\n')
            item=AnnotationDataset(root,[seq.name],'val',size=(64,64))[0]
            self.assertEqual(item['original_target'].tolist(),[120,60])
            self.assertAlmostEqual(item['target'][0].item(),63.36,places=4)
            self.assertAlmostEqual(item['target'][1].item(),47.36,places=4)
if __name__=='__main__': unittest.main()
