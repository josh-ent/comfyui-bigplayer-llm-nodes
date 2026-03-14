from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: tests that start ComfyUI")
    config.addinivalue_line("markers", "live: tests that hit the real Grok API")


if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

os.environ.setdefault("PYTHONPATH", str(REPO_ROOT))

