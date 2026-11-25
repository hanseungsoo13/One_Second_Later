"""Compatibility alias for cucps.compat."""
import sys
import cucps.compat as _implementation
sys.modules[__name__] = _implementation
