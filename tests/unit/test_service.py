from __future__ import annotations

from dataclasses import dataclass

from bigplayer_prompting.operations import OperationKind
from bigplayer_prompting.provider import ProviderConfig
from bigplayer_prompting.service import PromptGenerationRequest, PromptGenerationService


@dataclass
class FakeModel:
    cached_patcher_init = (object(), ("/models/sdxl-base-1.0.safetensors",))


class FakeProvider:
    def __init__(self):
        self.operations = []
        self.configs = []

    def invoke(self, operation, config):
        self.operations.append(operation)
        self.configs.append(config)
        if operation.response_schema_name.endswith("simple_prompt_result"):
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
    service = PromptGenerationService(providers={"xAI": provider})
    service.generate(
        PromptGenerationRequest(
            mode="simple",
            prose="A cinematic portrait of a cat.",
            api_key="secret-key",
            provider="xAI",
            provider_model="grok-4-latest",
            target_model=FakeModel(),
        )
    )
    operation = provider.operations[0]
    config = provider.configs[0]
    assert operation.kind is OperationKind.PROMPT_GENERATION
    assert operation.target_model_name == "sdxl-base-1.0.safetensors"
    assert operation.output_mode == "simple"
    assert operation.response_schema_name == "bigplayer_simple_prompt_result"
    assert config.provider == "xAI"
    assert config.provider_model == "grok-4-latest"


def test_service_caches_when_assume_determinism_is_enabled():
    provider = FakeProvider()
    service = PromptGenerationService(providers={"xAI": provider})
    request = PromptGenerationRequest(
        mode="simple",
        prose="A cinematic portrait of a cat.",
        api_key="secret-key",
        provider="xAI",
        provider_model="grok-4-latest",
        target_model=FakeModel(),
        assume_determinism=True,
    )
    first = service.generate(request)
    second = service.generate(request)
    assert first == second
    assert len(provider.operations) == 1


def test_service_reexecutes_when_assume_determinism_is_disabled():
    provider = FakeProvider()
    service = PromptGenerationService(providers={"xAI": provider})
    request = PromptGenerationRequest(
        mode="simple",
        prose="A cinematic portrait of a cat.",
        api_key="secret-key",
        provider="xAI",
        provider_model="grok-4-latest",
        target_model=FakeModel(),
        assume_determinism=False,
    )
    service.generate(request)
    service.generate(request)
    assert len(provider.operations) == 2


def test_split_fallback_is_annotated():
    class DuplicatingProvider(FakeProvider):
        def invoke(self, operation, config):
            return {
                "text_l_positive": "same",
                "text_g_positive": "same",
                "text_l_negative": "",
                "text_g_negative": "",
                "comments": "Provider could not distinguish channels.",
            }

    service = PromptGenerationService(providers={"xAI": DuplicatingProvider()})
    result = service.generate(
        PromptGenerationRequest(
            mode="split",
            prose="A cinematic portrait of a cat.",
            api_key="secret-key",
            provider="xAI",
            provider_model="grok-4-latest",
            target_model=FakeModel(),
        )
    )
    assert "Fallback used" in result.comments


def test_provider_interface_supports_future_operation_shapes():
    @dataclass(frozen=True)
    class FakeFutureOperation:
        kind: str = "future_operation"
        value: str = "payload"

    class FakeFutureProvider:
        def invoke(self, operation, config):
            return {"kind": operation.kind, "value": operation.value, "provider": config.provider}

    provider = FakeFutureProvider()
    payload = provider.invoke(
        FakeFutureOperation(),
        ProviderConfig(provider="xAI", provider_model="grok-4-latest", api_key="secret-key"),
    )
    assert payload == {"kind": "future_operation", "value": "payload", "provider": "xAI"}
