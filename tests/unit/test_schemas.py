from __future__ import annotations

import pytest

from bigplayer_prompting.capabilities import BASIC_PROMPT_CAPABILITY, KSAMPLER_CONFIG_CAPABILITY
from bigplayer_prompting.errors import MalformedProviderResponseError
from bigplayer_prompting.schemas import validate_result


def test_validate_composed_result_success():
    result = validate_result(
        {
            BASIC_PROMPT_CAPABILITY: {},
            KSAMPLER_CONFIG_CAPABILITY: {
                "sampler_names": ["euler", "ddim"],
                "scheduler_names": ["normal", "karras"],
            },
        },
        {
            "basic_prompt": {
                "positive_prompt": "cinematic cat portrait",
                "negative_prompt": "blurry, distorted",
                "comments": "Tailored for cinematic framing.",
            },
            "ksampler_config": {
                "steps": 24,
                "cfg": 7.5,
                "sampler_name": "euler",
                "scheduler": "karras",
                "denoise": 1.0,
                "comments": "Balanced quality and speed.",
            },
        },
    )
    assert result["basic_prompt"]["positive_prompt"] == "cinematic cat portrait"
    assert result["ksampler_config"]["sampler_name"] == "euler"


def test_validate_result_rejects_missing_top_level_capability():
    with pytest.raises(MalformedProviderResponseError):
        validate_result(
            {BASIC_PROMPT_CAPABILITY: {}},
            {},
        )


def test_validate_result_rejects_invalid_sampler_value():
    with pytest.raises(MalformedProviderResponseError):
        validate_result(
            {
                KSAMPLER_CONFIG_CAPABILITY: {
                    "sampler_names": ["euler"],
                    "scheduler_names": ["normal"],
                }
            },
            {
                "ksampler_config": {
                    "steps": 20,
                    "cfg": 8.0,
                    "sampler_name": "bad",
                    "scheduler": "normal",
                    "denoise": 1.0,
                    "comments": "Invalid sampler.",
                }
            },
        )
