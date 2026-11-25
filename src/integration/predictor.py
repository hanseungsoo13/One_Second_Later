"""Load the frozen OSL predictor used to build a CUCPS prior."""
import torch

from osl.model import DinoBackbone, Model
from integration.variants import resolve_cucps_variant


def build_motion_model(cfg, device, paths, variant='temporal', *, backbone_class=DinoBackbone, model_class=Model):
    motion_cfg = cfg['motion']
    specification = resolve_cucps_variant(variant)
    if specification is None:
        raise ValueError('baseline has no motion model')
    pose_fusion = specification.pose_fusion
    model = model_class(
        backbone_class(str(paths[('motion', 'repo')]), str(paths[('motion', 'weights')]), motion_cfg['name']),
        transformer=specification.transformer,
        person_branch=specification.person_branch,
        pose_fusion=pose_fusion,
        semantic_channels=150 if specification.semantic else None,
    ).to(device)
    checkpoint_key = 'pose_checkpoint' if pose_fusion else 'checkpoint'
    checkpoint = torch.load(paths[('motion', checkpoint_key)], map_location='cpu', weights_only=False)
    model.load_state_dict(checkpoint['model'], strict=True)
    model.eval()
    return model


def make_pose_estimator(cfg, device):
    pose_cfg = cfg['motion'].get('pose', {})
    try:
        from rtmlib import RTMPose
    except ImportError as exc:
        raise RuntimeError('P1/P2 experiments require rtmlib (RTMPose) in the active environment') from exc
    model_url = pose_cfg.get(
        'model',
        'https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/'
        'rtmpose-m_simcc-body7_pt-body7_420e-256x192-e48f03d0_20230504.zip',
    )
    return RTMPose(model_url, model_input_size=tuple(pose_cfg.get('input_size', [192, 256])),
                   backend=pose_cfg.get('backend', 'onnxruntime'),
                   device=pose_cfg.get('device', str(device)))
