from __future__ import annotations

from dataclasses import dataclass

from bigplayer_prompting.service import PromptGenerationRequest, PromptGenerationService


@dataclass
class FakeModel:
    cached_patcher_init = (object(), ("/models/sdxl-base-1.0.safetensors",))


class FakeProvider:
    def __init__(self):
        self.calls = []

    def generate_structured(self, request):
        self.calls.append(request)
        if request.schema_name.endswith("simple_prompt_result"):
            return {
                "positive_prompt": "cinematic cat portrait",
                "negative_prompt": "blurry",
                "comments": "Used SDXL framing conventions.",
            }
        return {
            "text_l_positive": "cat portrait",
            "text_g_positive": "cinematic studio light",
            "text_l_negative": "blurry",
            "text_g_negative": "deformed",
            "comments": "Split channels for local and global text encoders.",
        }


def test_service_uses_model_name_and_mode_in_request():
    provider = FakeProvider()
    service = PromptGenerationService(provider=provider)
    service.generate(
        PromptGenerationRequest(
            mode="simple",
            prose="A cinematic portrait of a cat.",
            api_key="secret-key",
            llm_model="grok-test",
            model=FakeModel(),
        )
    )
    sent = provider.calls[0]
    assert "sdxl-base-1.0.safetensors" in sent.user_prompt
    assert "Return `positive_prompt`, `negative_prompt`, and `comments`." in sent.user_prompt


def test_service_caches_when_assume_determinism_is_enabled():
    provider = FakeProvider()
    service = PromptGenerationService(provider=provider)
    request = PromptGenerationRequest(
        mode="simple",
        prose="A cinematic portrait of a cat.",
        api_key="secret-key",
        llm_model="grok-test",
        model=FakeModel(),
        assume_determinism=True,
    )
    first = service.generate(request)
    second = service.generate(request)
    assert first == second
    assert len(provider.calls) == 1


def test_service_reexecutes_when_assume_determinism_is_disabled():
    provider = FakeProvider()
    service = PromptGenerationService(provider=provider)
    request = PromptGenerationRequest(
        mode="simple",
        prose="A cinematic portrait of a cat.",
        api_key="secret-key",
        llm_model="grok-test",
        model=FakeModel(),
        assume_determinism=False,
    )
    service.generate(request)
    service.generate(request)
    assert len(provider.calls) == 2


def test_split_fallback_is_annotated():
    class DuplicatingProvider(FakeProvider):
        def generate_structured(self, request):
            return {
                "text_l_positive": "same",
                "text_g_positive": "same",
                "text_l_negative": "",
                "text_g_negative": "",
                "comments": "Provider could not distinguish channels.",
            }

    service = PromptGenerationService(provider=DuplicatingProvider())
    result = service.generate(
        PromptGenerationRequest(
            mode="split",
            prose="A cinematic portrait of a cat.",
            api_key="secret-key",
            llm_model="grok-test",
            model=FakeModel(),
        )
    )
    assert "Fallback used" in result.comments

