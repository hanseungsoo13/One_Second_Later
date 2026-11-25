"""Compatibility alias for integration.location."""
import sys
import integration.location as _implementation
sys.modules[__name__] = _implementation
