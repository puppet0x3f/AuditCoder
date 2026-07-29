import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent))


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: marks tests that require LLM API access"
    )
