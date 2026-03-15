from __future__ import annotations

from dataclasses import dataclass
from threading import Lock
from typing import Any

from .cache import CacheKey, DeterministicCache, stable_hash
from .capabilities import CAPABILITY_DEFINITIONS, MODULE_CLASS_TO_CAPABILITY
from .operations import PromptGenerationOperation
from .schemas import validate_result
from ..errors import ProviderError
from ..providers.base import InvocationContext, ProviderConfig, ProviderDebugRecord, redact_secret
from ..providers.registry import REGISTERED_PROVIDERS
from ..state.preset import normalize_preset_config, render_preset_config, serialize_preset_config

SCHEMA_VERSION = "modular-v3"
PROMPT_DEBUG_CLASS_TYPE = "BigPlayerPromptDebug"


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
    debug: ProviderDebugRecord | None


@dataclass(frozen=True)
class CachedSessionValue:
    payload: dict[str, dict[str, Any]]
    debug: ProviderDebugRecord | None


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
        preset_config: Any = None,
        invocation_context: InvocationContext | None = None,
    ) -> LLMSessionHandle:
        invocation_context = invocation_context or InvocationContext()
        self._validate_provider_bundle(provider_bundle)
        prompt = self._extract_prompt(dynprompt)
        debug_enabled = self._has_prompt_debug(prompt, root_node_id)
        capability_instances = self._discover_capabilities(prompt, root_node_id)
        capability_configs = self._consolidate_capabilities(capability_instances)
        output_configs = {
            capability_id: config
            for capability_id, config in capability_configs.items()
            if CAPABILITY_DEFINITIONS[capability_id].produces_output
        }
        if not output_configs:
            raise ProviderError(
                "Natural Language Root requires at least one attached output module before calling the provider."
            )

        preset_bundle = normalize_preset_config(preset_config)
        context_blocks = self._build_context_blocks(preset_bundle=preset_bundle)
        cache_key = self._build_cache_key(
            prose,
            provider_bundle,
            capability_configs,
            preset_bundle=preset_bundle,
            debug_enabled=debug_enabled,
        )
        session_handle = LLMSessionHandle(
            session_id=stable_hash({"root_node_id": root_node_id, "cache_key": cache_key.value}),
            root_node_id=str(root_node_id),
            cache_key=cache_key.value,
        )

        if provider_bundle.assume_determinism:
            cached = self._cache.get(cache_key)
            if cached is not None:
                invocation_context.report_status("Reusing cached modular LLM result.")
                self._sessions.set(
                    session_handle.session_id,
                    SessionRecord(
                        capability_configs=capability_configs,
                        payload=cached.payload,
                        debug=cached.debug,
                    ),
                )
                return session_handle

        operation = self._build_operation(
            prose,
            output_configs,
            context_blocks=context_blocks,
        )
        debug_record = ProviderDebugRecord() if debug_enabled else None
        provider_context = InvocationContext(
            status_callback=invocation_context.status_callback,
            debug_record=debug_record,
        )
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
            provider_context,
        )
        invocation_context.report_status("Validating structured provider response...")
        validated = validate_result(output_configs, payload)
        if provider_bundle.assume_determinism:
            self._cache.set(
                cache_key,
                CachedSessionValue(payload=validated, debug=self._freeze_debug_record(debug_record)),
            )
        self._sessions.set(
            session_handle.session_id,
            SessionRecord(
                capability_configs=capability_configs,
                payload=validated,
                debug=self._freeze_debug_record(debug_record),
            ),
        )
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
        return self.build_root_change_token(
            prose=prose,
            provider_bundle=provider_bundle,
            dynprompt=None,
            root_node_id="",
            preset_config=None,
        )

    def build_root_change_token(
        self,
        *,
        prose: str,
        provider_bundle: LLMProviderBundle,
        dynprompt: Any,
        root_node_id: str,
        preset_config: Any = None,
    ) -> float | str:
        if not provider_bundle.assume_determinism:
            return float("NaN")

        capability_configs: dict[str, dict[str, Any]] = {}
        debug_enabled = False
        if dynprompt is not None:
            prompt = self._extract_prompt(dynprompt)
            debug_enabled = self._has_prompt_debug(prompt, root_node_id)
            capability_instances = self._discover_capabilities(prompt, root_node_id)
            capability_configs = self._consolidate_capabilities(capability_instances)

        preset_bundle = normalize_preset_config(preset_config)
        return stable_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "prose": prose.strip(),
                "provider": provider_bundle.provider,
                "provider_model": provider_bundle.provider_model,
                "api_key": redact_secret(provider_bundle.api_key),
                "provider_base_url": provider_bundle.provider_base_url.strip(),
                "capability_configs": capability_configs,
                "preset_config": serialize_preset_config(preset_bundle),
                "debug_enabled": debug_enabled,
            }
        )

    def _build_operation(
        self,
        prose: str,
        output_configs: dict[str, dict[str, Any]],
        *,
        context_blocks: tuple[tuple[str, str], ...],
    ) -> PromptGenerationOperation:
        ordered_capabilities = tuple(self._sort_capability_ids(output_configs))
        return PromptGenerationOperation(
            prose=prose,
            context_blocks=context_blocks,
            requested_capabilities=ordered_capabilities,
            capability_configs={capability_id: output_configs[capability_id] for capability_id in ordered_capabilities},
        )

    def _build_context_blocks(
        self,
        *,
        preset_bundle,
    ) -> tuple[tuple[str, str], ...]:
        blocks: list[tuple[str, str]] = []
        preset_text = render_preset_config(preset_bundle).strip()
        if preset_text:
            blocks.append(("Preset workflow config", preset_text))
        return tuple(blocks)

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

    def _extract_prompt(self, dynprompt: Any) -> dict[str, Any]:
        if dynprompt is None:
            raise ProviderError("Natural Language Root requires workflow metadata to discover attached modules.")
        prompt = dynprompt.get_original_prompt()
        if not isinstance(prompt, dict):
            raise ProviderError("Natural Language Root could not inspect the workflow graph.")
        return prompt

    def _discover_capabilities(self, prompt: dict[str, Any], root_node_id: str) -> list[CapabilityInstance]:
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

        discovered.sort(
            key=lambda item: (
                CAPABILITY_DEFINITIONS[item.capability_id].composition_priority,
                item.capability_id,
                self._node_sort_key(item.node_id),
            )
        )
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
                    f"Conflicting `{instance.capability_id}` modules are attached to the same Natural Language Root."
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
        *,
        preset_bundle,
        debug_enabled: bool,
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
                    "preset_config": serialize_preset_config(preset_bundle),
                    "debug_enabled": debug_enabled,
                }
            )
        )

    def resolve_debug(self, session: LLMSessionHandle) -> ProviderDebugRecord:
        record = self._sessions.get(session.session_id)
        if record is None:
            raise ProviderError("No shared LLM result was recorded for this session.")
        if record.debug is None:
            raise ProviderError("Prompt Debug was not attached to the Natural Language Root for this session.")
        return record.debug

    def _is_direct_root_link(self, value: Any, root_node_id: str) -> bool:
        if not isinstance(value, (list, tuple)) or len(value) < 2:
            return False
        return str(value[0]) == str(root_node_id) and int(value[1]) == 0

    def _node_sort_key(self, node_id: str) -> tuple[int, str]:
        return (0, f"{int(node_id):08d}") if str(node_id).isdigit() else (1, str(node_id))

    def _sort_capability_ids(self, capability_ids) -> list[str]:
        return sorted(
            capability_ids,
            key=lambda capability_id: (
                CAPABILITY_DEFINITIONS[capability_id].composition_priority,
                capability_id,
            ),
        )

    def _has_prompt_debug(self, prompt: dict[str, Any], root_node_id: str) -> bool:
        for node in prompt.values():
            if not isinstance(node, dict):
                continue
            if str(node.get("class_type")) != PROMPT_DEBUG_CLASS_TYPE:
                continue
            inputs = node.get("inputs", {})
            if self._is_direct_root_link(inputs.get("session"), root_node_id):
                return True
        return False

    def _freeze_debug_record(self, debug_record: ProviderDebugRecord | None) -> ProviderDebugRecord | None:
        if debug_record is None:
            return None
        return ProviderDebugRecord(
            request_text=debug_record.request_text,
            response_text=debug_record.response_text,
        )
