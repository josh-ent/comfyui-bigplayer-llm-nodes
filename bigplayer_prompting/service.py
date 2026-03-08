from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .cache import CacheKey, DeterministicCache, stable_hash
from .model_name import extract_model_name
from .prompts import SYSTEM_PROMPT, build_user_prompt
from .provider import REGISTERED_PROVIDERS, ProviderRequest, redact_secret
from .schemas import PromptMode, get_provider_schema, validate_result

SCHEMA_VERSION = "phase1-v2"


@dataclass(frozen=True)
class PromptGenerationRequest:
    mode: PromptMode
    prose: str
    api_key: str
    provider: str
    provider_model: str
    target_model: Any
    style_policy: str = ""
    provider_base_url: str = ""
    assume_determinism: bool = True


class PromptGenerationService:
    def __init__(self, cache: DeterministicCache | None = None, providers: dict[str, Any] | None = None) -> None:
        self._cache = cache or DeterministicCache()
        self._providers = providers or {
            provider_id: definition.factory() for provider_id, definition in REGISTERED_PROVIDERS.items()
        }

    def generate(self, request: PromptGenerationRequest):
        model_name = extract_model_name(request.target_model)
        cache_key = self._build_cache_key(request, model_name)
        if request.assume_determinism:
            cached = self._cache.get(cache_key)
            if cached is not None:
                return cached

        user_prompt = build_user_prompt(
            prose=request.prose,
            model_name=model_name,
            style_policy=request.style_policy,
            mode=request.mode,
        )
        payload = self._providers[request.provider].generate_structured(
            ProviderRequest(
                api_key=request.api_key,
                provider_model=request.provider_model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=user_prompt,
                schema_name=f"bigplayer_{request.mode}_prompt_result",
                schema=get_provider_schema(request.mode),
                provider_base_url=request.provider_base_url or None,
            )
        )
        result = validate_result(request.mode, payload)
        result = self._annotate_split_fallback(result, request.mode)
        if request.assume_determinism:
            self._cache.set(cache_key, result)
        return result

    def build_is_changed_token(
        self,
        *,
        prose: str,
        api_key: str,
        provider: str,
        provider_model: str,
        style_policy: str,
        provider_base_url: str,
        assume_determinism: bool,
        mode: PromptMode,
    ) -> float | str:
        if not assume_determinism:
            return float("NaN")
        return stable_hash(
            {
                "schema_version": SCHEMA_VERSION,
                "mode": mode,
                "prose": prose.strip(),
                "api_key": redact_secret(api_key),
                "provider": provider,
                "provider_model": provider_model,
                "style_policy": style_policy.strip(),
                "provider_base_url": provider_base_url.strip(),
            }
        )

    def _build_cache_key(self, request: PromptGenerationRequest, model_name: str) -> CacheKey:
        return CacheKey(
            stable_hash(
                {
                    "schema_version": SCHEMA_VERSION,
                    "mode": request.mode,
                    "prose": request.prose.strip(),
                    "api_key": redact_secret(request.api_key),
                    "provider": request.provider,
                    "provider_model": request.provider_model,
                    "model_name": model_name,
                    "style_policy": request.style_policy.strip(),
                    "provider_base_url": request.provider_base_url.strip(),
                }
            )
        )

    def _annotate_split_fallback(self, result, mode: PromptMode):
        if mode != "split":
            return result
        duplicated_positive = result.text_l_positive == result.text_g_positive
        duplicated_negative = result.text_l_negative == result.text_g_negative
        if duplicated_positive or duplicated_negative:
            suffix = " Fallback used: duplicated L/G channel text because the provider did not separate the channels."
            result.comments = f"{result.comments}{suffix}".strip()
        return result
