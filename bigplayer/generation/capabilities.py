from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from ..errors import MalformedProviderResponseError, ProviderError

BASIC_PROMPT_CAPABILITY = "basic_prompt"
SPLIT_PROMPT_CAPABILITY = "split_prompt"
KSAMPLER_CONFIG_CAPABILITY = "ksampler_config"
CHECKPOINT_PICKER_CAPABILITY = "checkpoint_picker"

BIGPLAYER_MODULE_CLASS_TYPES = {
    "BigPlayerBasicPrompt",
    "BigPlayerSplitPrompt",
    "BigPlayerKSamplerConfig",
    "BigPlayerCheckpointPicker",
}


def list_available_checkpoints() -> list[str]:
    try:
        import folder_paths
    except ImportError:
        return []
    return sorted(folder_paths.get_filename_list("checkpoints"))


def list_sampler_names() -> list[str]:
    try:
        import comfy.samplers
    except ImportError:
        return ["euler", "euler_ancestral", "dpmpp_2m", "dpmpp_sde", "ddim"]
    return list(comfy.samplers.KSampler.SAMPLERS)


def list_scheduler_names() -> list[str]:
    try:
        import comfy.samplers
    except ImportError:
        return ["normal", "karras", "simple", "sgm_uniform", "ddim_uniform"]
    return list(comfy.samplers.KSampler.SCHEDULERS)


class _BaseSchema(BaseModel):
    model_config = ConfigDict(extra="forbid")

    @field_validator("*", mode="before")
    @classmethod
    def _normalize_strings(cls, value: Any) -> Any:
        if isinstance(value, str):
            return value.strip()
        return value


class BasicPromptPayload(_BaseSchema):
    positive_prompt: str
    negative_prompt: str
    comments: str = Field(min_length=1)


class SplitPromptPayload(_BaseSchema):
    text_l_positive: str
    text_g_positive: str
    text_l_negative: str
    text_g_negative: str
    comments: str = Field(min_length=1)


class KSamplerConfigPayload(_BaseSchema):
    steps: int = Field(ge=1, le=10000)
    cfg: float = Field(ge=0.0, le=100.0)
    sampler_name: str
    scheduler: str
    denoise: float = Field(ge=0.0, le=1.0)
    comments: str = Field(min_length=1)


class CheckpointPickerPayload(_BaseSchema):
    checkpoint_name: str = Field(min_length=1)
    comments: str = Field(min_length=1)


def _validate_model(model_cls: type[BaseModel], payload: dict[str, Any]) -> BaseModel:
    try:
        return model_cls.model_validate(payload)
    except ValidationError as exc:
        raise MalformedProviderResponseError(f"Provider response failed schema validation: {exc}") from exc


def _validate_basic_prompt(payload: dict[str, Any], _: dict[str, Any]) -> BasicPromptPayload:
    return _validate_model(BasicPromptPayload, payload)


def _validate_split_prompt(payload: dict[str, Any], _: dict[str, Any]) -> SplitPromptPayload:
    return _validate_model(SplitPromptPayload, payload)


def _validate_ksampler(payload: dict[str, Any], config: dict[str, Any]) -> KSamplerConfigPayload:
    result = _validate_model(KSamplerConfigPayload, payload)
    if result.sampler_name not in config["sampler_names"]:
        raise MalformedProviderResponseError(
            f"Provider response failed schema validation: unsupported sampler `{result.sampler_name}`."
        )
    if result.scheduler not in config["scheduler_names"]:
        raise MalformedProviderResponseError(
            f"Provider response failed schema validation: unsupported scheduler `{result.scheduler}`."
        )
    return result


def _validate_checkpoint(payload: dict[str, Any], config: dict[str, Any]) -> CheckpointPickerPayload:
    result = _validate_model(CheckpointPickerPayload, payload)
    if result.checkpoint_name not in config["available_checkpoints"]:
        raise MalformedProviderResponseError(
            f"Provider response failed schema validation: unknown checkpoint `{result.checkpoint_name}`."
        )
    return result


def _normalize_empty(inputs: dict[str, Any]) -> dict[str, Any]:
    del inputs
    return {}


def _resolve_empty(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config)


def _resolve_ksampler(config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    resolved["sampler_names"] = list_sampler_names()
    resolved["scheduler_names"] = list_scheduler_names()
    return resolved


def _resolve_checkpoint(config: dict[str, Any]) -> dict[str, Any]:
    resolved = dict(config)
    resolved["available_checkpoints"] = list_available_checkpoints()
    if not resolved["available_checkpoints"]:
        raise ProviderError("Checkpoint Picker requires at least one available checkpoint in ComfyUI.")
    return resolved


@dataclass(frozen=True)
class CapabilityDefinition:
    capability_id: str
    module_class_type: str
    produces_output: bool
    composition_priority: int
    normalize_config: Callable[[dict[str, Any]], dict[str, Any]]
    resolve_config: Callable[[dict[str, Any]], dict[str, Any]]
    validate_payload: Callable[[dict[str, Any], dict[str, Any]], BaseModel]


CAPABILITY_DEFINITIONS: dict[str, CapabilityDefinition] = {
    BASIC_PROMPT_CAPABILITY: CapabilityDefinition(
        capability_id=BASIC_PROMPT_CAPABILITY,
        module_class_type="BigPlayerBasicPrompt",
        produces_output=True,
        composition_priority=300,
        normalize_config=_normalize_empty,
        resolve_config=_resolve_empty,
        validate_payload=_validate_basic_prompt,
    ),
    SPLIT_PROMPT_CAPABILITY: CapabilityDefinition(
        capability_id=SPLIT_PROMPT_CAPABILITY,
        module_class_type="BigPlayerSplitPrompt",
        produces_output=True,
        composition_priority=300,
        normalize_config=_normalize_empty,
        resolve_config=_resolve_empty,
        validate_payload=_validate_split_prompt,
    ),
    KSAMPLER_CONFIG_CAPABILITY: CapabilityDefinition(
        capability_id=KSAMPLER_CONFIG_CAPABILITY,
        module_class_type="BigPlayerKSamplerConfig",
        produces_output=True,
        composition_priority=200,
        normalize_config=_normalize_empty,
        resolve_config=_resolve_ksampler,
        validate_payload=_validate_ksampler,
    ),
    CHECKPOINT_PICKER_CAPABILITY: CapabilityDefinition(
        capability_id=CHECKPOINT_PICKER_CAPABILITY,
        module_class_type="BigPlayerCheckpointPicker",
        produces_output=True,
        composition_priority=100,
        normalize_config=_normalize_empty,
        resolve_config=_resolve_checkpoint,
        validate_payload=_validate_checkpoint,
    ),
}

MODULE_CLASS_TO_CAPABILITY = {
    definition.module_class_type: definition.capability_id for definition in CAPABILITY_DEFINITIONS.values()
}
