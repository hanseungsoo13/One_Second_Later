"""Names and architecture features for the supported OSL models."""
from dataclasses import dataclass


@dataclass(frozen=True)
class ModelVariant:
    name: str
    transformer: bool
    person_branch: bool
    pose_fusion: str | None = None
    semantic: bool = False


_VARIANTS = {
    'e2a': ModelVariant('full_bbox_a', False, False),
    'e2b': ModelVariant('full_bbox_b', True, False),
    'e2c': ModelVariant('full_bbox_c', True, True),
    'e2c_p1': ModelVariant('full_bbox_pose1', True, True, 'p1'),
    'e2c_p2': ModelVariant('full_bbox_pose2', True, True, 'p2'),
    'e2c_p2_s': ModelVariant('full_bbox_pose_seg', True, True, 'p2', True),
}
_VARIANTS.update({variant.name: variant for variant in tuple(_VARIANTS.values())})


def resolve_model_variant(name: str) -> ModelVariant:
    try:
        return _VARIANTS[name]
    except KeyError as exc:
        raise ValueError(f'Unknown model variant: {name}') from exc



def resolve_cucps_variant(name: str):
    """Compatibility alias for the CUCPS integration variant resolver."""
    from integration.variants import resolve_cucps_variant as resolve
    return resolve(name)
