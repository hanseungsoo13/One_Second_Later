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
    def test_frame_sampling_preserves_all_people_and_val(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); seq=root/'MOT17-02-FRCNN'; seq.mkdir()
            records=[]
            for frame in (1,2,15,16,31):
                for person in (1,2):
                    records.append({'sequence':seq.name,'person_id':person,'frame':frame,'target_frame':frame+30,'time_delta_seconds':1.0})
            for split in ('train','val'):
                (seq/f'{split}.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records))
            train=AnnotationDataset(root,[seq.name],'train',sample_fps=2)
            val=AnnotationDataset(root,[seq.name],'val')
            self.assertEqual([r['frame'] for r in train.records], [1,1,16,16,31,31])
            self.assertEqual(len(val),10)
    def test_person_crop_uses_original_image_pixels(self):
        with tempfile.TemporaryDirectory() as td:
            root=Path(td); seq=root/'MOT17-02-FRCNN'; seq.mkdir(); image=root/'frame.png'
            im=Image.new('RGB',(100,50)); im.putpixel((50,25),(255,0,0)); im.save(image)
            record={'sequence':seq.name,'person_id':1,'frame':1,'image_path':str(image),'image_size':[100,50],'bbox_xywh':[49,24,2,2],'target_bottom_center_xy':[50,30]}
            (seq/'train.jsonl').write_text(json.dumps(record)+'\n')
            ds=AnnotationDataset(root,[seq.name],'train',size=(16,16),person_crop={'size':[32,32],'scale':2})
            self.assertEqual(ds[0]['person_crop'].shape,(3,32,32))
            self.assertGreater(ds[0]['person_crop'][0].max().item(),1.)
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
