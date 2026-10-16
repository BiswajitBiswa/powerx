"""Shared fixtures and mocks for test suite."""

import sys
from pathlib import Path
import pytest

# Ensure hybrid_gpu_mesh is in python path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))
