import sys, unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import torch
from torch import nn
from osl.model import Model, spatial_prompt, heatmap_loss
from osl.evaluate import predictions
class Backbone(nn.Module):
    channels=8
    def forward(self,x): return torch.zeros(x.shape[0],8,x.shape[2]//16,x.shape[3]//16)
class PipelineTests(unittest.TestCase):
    def test_prompt_and_distribution(self):
        box=torch.tensor([[16.,16.,48.,48.]])
        p=spatial_prompt(box,4,4,(64,64)); self.assertEqual(p.shape,(1,4,4,4)); self.assertGreater(p[0,0].sum(),0)
        logits=Model(Backbone(),True)(torch.randn(1,3,64,64),box)
        self.assertEqual(logits.shape,(1,1,16,16))
        loss=heatmap_loss(logits,torch.tensor([[32.,48.]]),(64,64),2.5); loss.backward()
        self.assertTrue(torch.isfinite(loss)); self.assertIsNotNone(logits.grad_fn)
    def test_coordinate_centers(self):
        x=torch.full((1,1,2,4),-100.); x[0,0,1,2]=100
        arg,expected,p=predictions(x,(20,40)); self.assertTrue(torch.allclose(arg,torch.tensor([[25.,15.]]))); self.assertTrue(torch.allclose(arg,expected)); self.assertAlmostEqual(p.sum().item(),1.)
if __name__=='__main__': unittest.main()
