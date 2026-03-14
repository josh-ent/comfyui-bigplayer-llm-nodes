from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from .cache import CacheKey, DeterministicCache, stable_hash
from .capabilities import (
    CAPABILITY_DEFINITIONS,
    MODEL_CONTEXT_CAPABILITY,
    MODULE_CLASS_TO_CAPABILITY,
    model_context_to_text,
)
from .errors import ProviderError
from .operations import PromptGenerationOperation
from .provider import InvocationContext, ProviderConfig, REGISTERED_PROVIDERS, redact_secret
from .schemas import validate_result

SCHEMA_VERSION = "modular-v1"


@dataclass(frozen=True)
class LLMProviderBundle:
    api_key: str
    provider: str
    provider_model: str
    provider_base_url: str = ""
    assume_determinism: bool = True


@dataclass(frozen=True)
class LLMSessionHandle:
    session_id: str
    root_node_id: str
    cache_key: str


@dataclass(frozen=True)
class CapabilityInstance:
    node_id: str
    class_type: str
    capability_id: str
    normalized_config: dict[str, Any]


@dataclass(frozen=True)
class SessionRecord:
    capability_configs: dict[str, dict[str, Any]]
    payload: dict[str, dict[str, Any]]


class SessionRegistry:
    def __init__(self) -> None:
        self._data: dict[str, SessionRecord] = {}
        self._lock = Lock()

    def get(self, key: str) -> SessionRecord | None:
        with self._lock:
            return self._data.get(key)

    def set(self, key: str, value: SessionRecord) -> None:
        with self._lock:
            self._data[key] = value


class PromptGenerationService:
    def __init__(
        self,
        cache: DeterministicCache | None = None,
        providers: dict[str, Any] | None = None,
        sessions: SessionRegistry | None = None,
    ) -> None:
        self._cache = cache or DeterministicCache()
        self._providers = providers or {
            provider_id: definition.factory() for provider_id, definition in REGISTERED_PROVIDERS.items()
        }
        self._sessions = sessions or SessionRegistry()

    def begin_session(
        self,
        *,
        prose: str,
        provider_bundle: LLMProviderBundle,
        dynprompt: Any,
        root_node_id: str,
        invocation_context: InvocationContext | None = None,
    ) -> LLMSessionHandle:
        invocation_context = invocation_context or InvocationContext()
        self._validate_provider_bundle(provider_bundle)
        capability_instances = self._discover_capabilities(dynprompt, root_node_id)
        capability_configs = self._consolidate_capabilities(capability_instances)
        output_configs = {
            capability_id: config
            for capability_id, config in capability_configs.items()
            if CAPABILITY_DEFINITIONS[capability_id].produces_output
        }
        if not output_configs:
            raise ProviderError("LLM Root requires at least one attached output module before calling the provider.")

        cache_key = self._build_cache_key(prose, provider_bundle, capability_configs)
        session_handle = LLMSessionHandle(
            session_id=stable_hash({"root_node_id": root_node_id, "cache_key": cache_key.value}),
            root_node_id=str(root_node_id),
            cache_key=cache_key.value,
        )

        if provider_bundle.assume_determinism:
            cached = self._cache.get(cache_key)
            if cached is not None:
                invocation_context.report_status("Reusing cached modular LLM result.")
                self._sessions.set(session_handle.session_id, SessionRecord(capability_configs=capability_configs, payload=cached))
                return session_handle

        operation = self._build_operation(prose, capability_configs, output_configs)
        invocation_context.report_status(
            f"Calling {provider_bundle.provider} with model {provider_bundle.provider_model}..."
        )
        payload = self._providers[provider_bundle.provider].invoke(
            operation,
            ProviderConfig(
                provider=provider_bundle.provider,
                provider_model=provider_bundle.provider_model,
                api_key=provider_bundle.api_key,
                provider_base_url=provider_bundle.provider_base_url,
            ),
            invocation_context,
        )
        invocation_context.report_status("Validating structured provider response...")
        validated = validate_result(output_configs, payload)
        if provider_bundle.assume_determinism:
            self._cache.set(cache_key, validated)
        self._sessions.set(session_handle.session_id, SessionRecord(capability_configs=capability_configs, payload=validated))
        invocation_context.report_status("Modular prompt generation complete.")
        return session_handle

    def resolve_capability(self, session: LLMSessionHandle, capability_id: str):
        record = self._sessions.get(session.session_id)
        if record is None:
            raise ProviderError("No shared LLM result was recorded for this session.")
        config = record.capability_configs.get(capability_id)
        if config is None or not CAPABILITY_DEFINITIONS[capability_id].produces_output:
            raise ProviderError(f"The shared LLM session does not include capability `{capability_id}`.")
        payload = record.payload.get(capability_id)
        if payload is None:
            raise ProviderError(f"The shared LLM session did not return payload for `{capability_id}`.")
        return CAPABILITY_DEFINITIONS[capability_id].validate_payload(payload, config)

    def build_root_is_changed_token(self, *, prose: str, provider_bundle: LLMProviderBundle) -> float | str:
        if not provider_bundle.assume_determinism:
            return float("NaN")
        return stable_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "prose": prose.strip(),
                "provider": provider_bundle.provider,
                "provider_model": provider_bundle.provider_model,
                "api_key": redact_secret(provider_bundle.api_key),
                "provider_base_url": provider_bundle.provider_base_url.strip(),
            }
        )

    def _build_operation(
        self,
        prose: str,
        capability_configs: dict[str, dict[str, Any]],
        output_configs: dict[str, dict[str, Any]],
    ) -> PromptGenerationOperation:
        context_blocks: list[tuple[str, str]] = []
        for capability_id, config in capability_configs.items():
            if capability_id == MODEL_CONTEXT_CAPABILITY:
                prompt_text = model_context_to_text(config).strip()
                if prompt_text:
                    context_blocks.append(("Additional model context", prompt_text))

        return PromptGenerationOperation(
            prose=prose,
            context_blocks=tuple(context_blocks),
            requested_capabilities=tuple(sorted(output_configs)),
            capability_configs=output_configs,
        )

    def _validate_provider_bundle(self, provider_bundle: LLMProviderBundle) -> None:
        provider_definition = REGISTERED_PROVIDERS.get(provider_bundle.provider)
        if provider_definition is None:
            raise ProviderError(f"Unsupported provider `{provider_bundle.provider}`.")
        if provider_bundle.provider_model not in provider_definition.models:
            raise ProviderError(
                f"Provider `{provider_bundle.provider}` does not support model `{provider_bundle.provider_model}`."
            )
        if provider_definition.requires_api_key and not provider_bundle.api_key.strip():
            raise ProviderError("The api_key input cannot be empty.")

    def _discover_capabilities(self, dynprompt: Any, root_node_id: str) -> list[CapabilityInstance]:
        if dynprompt is None:
            raise ProviderError("LLM Root requires workflow metadata to discover attached modules.")
        prompt = dynprompt.get_original_prompt()
        if not isinstance(prompt, dict):
            raise ProviderError("LLM Root could not inspect the workflow graph.")

        discovered: list[CapabilityInstance] = []
        for node_id, node in prompt.items():
            if not isinstance(node, dict):
                continue
            class_type = node.get("class_type")
            capability_id = MODULE_CLASS_TO_CAPABILITY.get(str(class_type))
            if capability_id is None:
                continue
            inputs = node.get("inputs", {})
            session_input = inputs.get("session")
            if not self._is_direct_root_link(session_input, root_node_id):
                continue
            normalized_config = CAPABILITY_DEFINITIONS[capability_id].normalize_config(dict(inputs))
            discovered.append(
                CapabilityInstance(
                    node_id=str(node_id),
                    class_type=str(class_type),
                    capability_id=capability_id,
                    normalized_config=normalized_config,
                )
            )

        discovered.sort(key=lambda item: (item.capability_id, self._node_sort_key(item.node_id)))
        return discovered

    def _consolidate_capabilities(
        self,
        capability_instances: list[CapabilityInstance],
    ) -> dict[str, dict[str, Any]]:
        consolidated: dict[str, dict[str, Any]] = {}
        for instance in capability_instances:
            existing = consolidated.get(instance.capability_id)
            if existing is not None and existing != instance.normalized_config:
                raise ProviderError(
                    f"Conflicting `{instance.capability_id}` modules are attached to the same LLM Root."
                )
            if existing is None:
                consolidated[instance.capability_id] = CAPABILITY_DEFINITIONS[instance.capability_id].resolve_config(
                    instance.normalized_config
                )
        return consolidated

    def _build_cache_key(
        self,
        prose: str,
        provider_bundle: LLMProviderBundle,
        capability_configs: dict[str, dict[str, Any]],
    ) -> CacheKey:
        return CacheKey(
            stable_hash(
                {
                    "schema_version": SCHEMA_VERSION,
                    "prose": prose.strip(),
                    "provider": provider_bundle.provider,
                    "provider_model": provider_bundle.provider_model,
                    "api_key": redact_secret(provider_bundle.api_key),
                    "provider_base_url": provider_bundle.provider_base_url.strip(),
                    "capability_configs": capability_configs,
                }
            )
        )

    def _is_direct_root_link(self, value: Any, root_node_id: str) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return False
        return str(value[0]) == str(root_node_id) and int(value[1]) == 0

    def _node_sort_key(self, node_id: str) -> tuple[int, str]:
        return (0, f"{int(node_id):08d}") if str(node_id).isdigit() else (1, str(node_id))
