"""conftest.py — pytest configuration for SafeGenome tests.

Ensures the project root is on sys.path so that ``from src.X import Y``
works regardless of how pytest is invoked.
"""
import sys
from pathlib import Path

# Project root is one level above this file (tests/)
ROOT = Path(__file__).parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
