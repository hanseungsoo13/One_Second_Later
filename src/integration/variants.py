"""Map CUCPS experiment names to OSL model variants."""
from osl.variants import ModelVariant, resolve_model_variant

def resolve_cucps_variant(name: str) -> ModelVariant | None:
    if name == 'baseline':
        return None
    aliases = {'temporal': 'full_bbox_c', 'graph': 'full_bbox_c',
               'pose_p1': 'full_bbox_pose1', 'pose_p2': 'full_bbox_pose2'}
    return resolve_model_variant(aliases.get(name, name))
