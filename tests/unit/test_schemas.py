from __future__ import annotations

import pytest

from bigplayer_prompting.errors import MalformedProviderResponseError
from bigplayer_prompting.schemas import validate_result


def test_validate_simple_result_success():
    result = validate_result(
        "simple",
        {
            "positive_prompt": "cinematic cat portrait",
            "negative_prompt": "blurry, distorted",
            "comments": "Tailored for an SDXL-style checkpoint.",
        },
    )
    assert result.positive_prompt == "cinematic cat portrait"


def test_validate_split_result_success():
    result = validate_result(
        "split",
        {
            "text_l_positive": "portrait cat",
            "text_g_positive": "cinematic studio lighting, shallow depth of field",
            "text_l_negative": "blurry",
            "text_g_negative": "deformed anatomy",
            "comments": "Separated content and global style cues.",
        },
    )
    assert result.text_g_positive.startswith("cinematic")


def test_validate_result_rejects_missing_fields():
    with pytest.raises(MalformedProviderResponseError):
        validate_result(
            "simple",
            {
                "positive_prompt": "cat",
                "comments": "Missing negative prompt.",
            },
        )

