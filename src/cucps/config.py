"""Validate CUCPS experiment settings."""

def validate_cucps_config(config, variant_name):
    """Validate the chosen motion model's required settings before CUCPS loads."""
    from integration.variants import resolve_cucps_variant

    variant = resolve_cucps_variant(variant_name)
    if variant is None:
        return None
    motion = config.get('motion', {})
    checkpoint = 'pose_checkpoint' if variant.pose_fusion else 'checkpoint'
    if not motion.get(checkpoint):
        raise ValueError(f'{variant.name} requires motion.{checkpoint}')
    if variant.person_branch and not motion.get('person_crop'):
        raise ValueError(f'{variant.name} requires motion.person_crop')
    if variant.semantic:
        semantic = motion.get('semantic', {})
        if not all(semantic.get(key) for key in ('model', 'revision')):
            raise ValueError('full_bbox_pose_seg requires motion.semantic model and revision')
    return variant
