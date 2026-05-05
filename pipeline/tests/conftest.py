"""Pytest configuration — adds the pipeline root to sys.path so component imports resolve."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
