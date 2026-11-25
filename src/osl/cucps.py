"""Compatibility alias for cucps.search."""
import sys
import cucps.search as _implementation
sys.modules[__name__] = _implementation
