import sys, unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import torch
from torch import nn
from osl.model import Model

class Backbone(nn.Module):
    channels = 8
    def forward(self, x):
        return x.mean(1, keepdim=True).repeat(1, 8, 1, 1)[:, :, ::16, ::16]

class PoseModelTests(unittest.TestCase):
    def test_pose_variants_use_shared_encoder_and_receive_gradients(self):
        for fusion in ('p1', 'p2'):
            with self.subTest(fusion=fusion):
                torch.manual_seed(2)
                model = Model(Backbone(), person_branch=True, pose_fusion=fusion).eval()
                image = torch.randn(2, 3, 64, 64)
                box = torch.tensor([[4., 5., 30., 50.], [5., 6., 40., 55.]])
                pose = torch.rand(2, 17, 3)
                output = model(image, box, image, pose)
                self.assertEqual(output.shape, (2, 1, 16, 16))
                self.assertFalse(torch.allclose(output, model(image, box, image, torch.zeros_like(pose))))
                output.mean().backward()
                self.assertIsNotNone(model.pose_encoder.net[0].weight.grad)
                with self.assertRaises(ValueError): model(image, box, image)
