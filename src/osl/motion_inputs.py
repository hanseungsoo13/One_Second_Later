"""Compatibility alias for integration.motion_inputs."""
import sys
import integration.motion_inputs as _implementation
sys.modules[__name__] = _implementation
