"""Compatibility alias for integration.temporal."""
import sys
import integration.temporal as _implementation
sys.modules[__name__] = _implementation
