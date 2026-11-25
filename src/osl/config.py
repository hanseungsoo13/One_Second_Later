"""Validation shared by the OSL training and evaluation entry points."""
from .variants import resolve_model_variant


def validate_osl_config(config):
    """Validate model-dependent options before creating datasets or models."""
    for section in ('data', 'model', 'training'):
        if not isinstance(config.get(section), dict):
            raise ValueError(f'Configuration requires a {section} mapping')
    model = config['model']
    if 'variant' not in model:
        raise ValueError('Configuration requires model.variant')
    variant = resolve_model_variant(model['variant'])
    if variant.person_branch and not model.get('person_crop'):
        raise ValueError('Person branch requires model.person_crop configuration')
    if variant.pose_fusion and not model.get('pose'):
        raise ValueError('Pose variants require model.pose configuration')
    if variant.pose_fusion:
        cache = model['pose'].get('cache', {})
        if not all(cache.get(split) for split in ('train', 'val')):
            raise ValueError('model.pose.cache requires train and val paths')
    if variant.semantic:
        semantic = model.get('semantic')
        if not semantic or not all(semantic.get(key) for key in ('model', 'revision', 'cache')):
            raise ValueError('Semantic variants require model.semantic model, revision and cache')
        size = config['data'].get('size', ())
        if len(size) != 2 or any(not isinstance(n, int) or n <= 0 or n % 16 for n in size):
            raise ValueError('Scene dimensions must be positive and divisible by 16')
    return variant



def validate_cucps_config(config, variant_name):
    """Compatibility alias for CUCPS configuration validation."""
    from cucps.config import validate_cucps_config as validate
    return validate(config, variant_name)
