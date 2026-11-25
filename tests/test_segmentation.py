import json, sqlite3, sys, tempfile, unittest
from pathlib import Path
import numpy as np
import torch
from torch import nn
from PIL import Image
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from osl.segmentation import SegmentationCache, image_key, cache_metadata
from osl.data import AnnotationDataset
from osl.model import Model

class Backbone(nn.Module):
    channels = 8
    def forward(self, x): return x.mean(1, keepdim=True).repeat(1, 8, 1, 1)[:, :, ::16, ::16]

class SegmentationTests(unittest.TestCase):
    def test_semantic_fusion_affects_output_and_receives_gradient(self):
        torch.manual_seed(5)
        model = Model(Backbone(), person_branch=True, pose_fusion='p2', semantic_channels=150).eval()
        image = torch.rand(1, 3, 64, 64); box = torch.tensor([[8., 8., 40., 50.]])
        pose = torch.rand(1, 17, 3); seg = torch.rand(1, 150, 4, 4)
        a = model(image, box, image, pose, seg)
        b = model(image, box, image, pose, torch.zeros_like(seg))
        self.assertFalse(torch.allclose(a, b))
        a.mean().backward()
        self.assertIsNotNone(model.semantic_projection.weight.grad)
        with self.assertRaises(ValueError): model(image, box, image, pose)

    def test_cache_round_trip_and_metadata_validation(self):
        with tempfile.TemporaryDirectory() as td:
            path = Path(td) / 'cache.sqlite'
            metadata = cache_metadata('model', 'revision', (64, 64), (4, 4))
            cache = SegmentationCache(path, metadata, writable=True)
            record = {'image_path': '/a', 'image_size': [100, 50]}
            pose = np.full((150, 4, 4), 1/150, dtype=np.float32)
            cache.put(image_key(record), pose); cache.commit(); cache.close()
            reader = SegmentationCache(path, metadata)
            np.testing.assert_allclose(reader.get(image_key(record)), pose, atol=1e-5)
            with self.assertRaises(KeyError): reader.get('missing')
            reader.close()
            with self.assertRaises(ValueError): SegmentationCache(path, cache_metadata('other', 'revision', (64,64), (4,4)))

    def test_dataset_reuses_image_cache_for_people(self):
        with tempfile.TemporaryDirectory() as td:
            root = Path(td); seq = root/'s'; seq.mkdir(); image = root/'im.png'
            Image.new('RGB', (100, 50)).save(image)
            record = dict(sequence='s', person_id=1, frame=1, image_path=str(image), image_size=[100,50], bbox_xywh=[10,5,20,30], target_bottom_center_xy=[50,40])
            (seq/'val.jsonl').write_text(json.dumps(record)+'\n'+json.dumps({**record,'person_id':2})+'\n')
            metadata = cache_metadata('model', 'revision', (64,64), (4,4))
            path = root/'cache.sqlite'; cache = SegmentationCache(path, metadata, writable=True)
            cache.put(image_key(record), np.full((150,4,4), 1/150, np.float32)); cache.commit(); cache.close()
            ds = AnnotationDataset(root, ['s'], 'val', size=(64,64), semantic_cache=str(path), semantic_metadata=metadata)
            self.assertEqual(ds[0]['semantic'].shape, (150,4,4))
            torch.testing.assert_close(ds[0]['semantic'], ds[1]['semantic'])
