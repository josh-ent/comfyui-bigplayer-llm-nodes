from __future__ import annotations

import pytest

from tests.comfyui_runtime import ensure_test_image


@pytest.fixture(scope="session", autouse=True)
def prepared_comfyui_test_image() -> None:
    ensure_test_image()
