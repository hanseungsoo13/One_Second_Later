"""Compatibility alias for cucps.gallery."""
import sys
import cucps.gallery as _implementation
sys.modules[__name__] = _implementation
