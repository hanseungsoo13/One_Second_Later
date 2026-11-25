"""Compatibility entry point for scripts/osl/build_personpath22.py."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.osl.build_personpath22 import *

if __name__ == '__main__':
    main()
