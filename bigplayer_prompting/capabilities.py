from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

from .errors import MalformedProviderResponseError, ProviderError

BASIC_PROMPT_CAPABILITY = "basic_prompt"
SPLIT_PROMPT_CAPABILITY = "split_prompt"
KSAMPLER_CONFIG_CAPABILITY = "ksampler_config"
CHECKPOINT_PICKER_CAPABILITY = "checkpoint_picker"
MODEL_CONTEXT_CAPABILITY = "model_context"

BIGPLAYER_MODULE_CLASS_TYPES = {
    "BigPlayerBasicPrompt",
    "BigPlayerSplitPrompt",
    "BigPlayerKSamplerConfig",
    "BigPlayerCheckpointPicker",
    "BigPlayerModelContext",
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
        return [
            "euler",
            "euler_ancestral",
            "dpmpp_2m",
            "dpmpp_sde",
            "ddim",
        ]
    return list(comfy.samplers.KSampler.SAMPLERS)


def list_scheduler_names() -> list[str]:
    try:
        import comfy.samplers
    except ImportError:
        return [
            "normal",
            "karras",
            "simple",
            "sgm_uniform",
            "ddim_uniform",
        ]
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


def _basic_prompt_schema(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["positive_prompt", "negative_prompt", "comments"],
        "properties": {
            "positive_prompt": {"type": "string"},
            "negative_prompt": {"type": "string"},
            "comments": {"type": "string"},
        },
    }


def _split_prompt_schema(_: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "text_l_positive",
            "text_g_positive",
            "text_l_negative",
            "text_g_negative",
            "comments",
        ],
        "properties": {
            "text_l_positive": {"type": "string"},
            "text_g_positive": {"type": "string"},
            "text_l_negative": {"type": "string"},
            "text_g_negative": {"type": "string"},
            "comments": {"type": "string"},
        },
    }


def _ksampler_schema(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["steps", "cfg", "sampler_name", "scheduler", "denoise", "comments"],
        "properties": {
            "steps": {"type": "integer", "minimum": 1, "maximum": 10000},
            "cfg": {"type": "number", "minimum": 0.0, "maximum": 100.0},
            "sampler_name": {"type": "string", "enum": config["sampler_names"]},
            "scheduler": {"type": "string", "enum": config["scheduler_names"]},
            "denoise": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "comments": {"type": "string"},
        },
    }


def _checkpoint_schema(config: dict[str, Any]) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["checkpoint_name", "comments"],
        "properties": {
            "checkpoint_name": {"type": "string", "enum": config["available_checkpoints"]},
            "comments": {"type": "string"},
        },
    }


def _basic_prompt_instruction(_: dict[str, Any]) -> str:
    return (
        "Capability `basic_prompt`:\n"
        "- Return `positive_prompt`, `negative_prompt`, and `comments`.\n"
        "- Keep prompts concise, specific, and directly usable.\n"
        "- Negative prompts should omit unwanted content instead of explaining policy."
    )


def _split_prompt_instruction(_: dict[str, Any]) -> str:
    return (
        "Capability `split_prompt`:\n"
        "- Return `text_l_positive`, `text_g_positive`, `text_l_negative`, `text_g_negative`, and `comments`.\n"
        "- Separate local subject/content cues from broader global/style cues when possible.\n"
        "- Do not duplicate the full positive prompt into both channels unless that fallback is necessary and noted in comments."
    )


def _ksampler_instruction(config: dict[str, Any]) -> str:
    return (
        "Capability `ksampler_config`:\n"
        "- Choose `steps`, `cfg`, `sampler_name`, `scheduler`, `denoise`, and `comments` for a standard ComfyUI KSampler.\n"
        f"- Allowed sampler names: {', '.join(config['sampler_names'])}\n"
        f"- Allowed scheduler names: {', '.join(config['scheduler_names'])}\n"
        "- Prefer practical, workflow-ready defaults consistent with the user's prose."
    )


def _checkpoint_instruction(config: dict[str, Any]) -> str:
    choices = ", ".join(config["available_checkpoints"])
    return (
        "Capability `checkpoint_picker`:\n"
        "- Choose exactly one checkpoint from the allowed list and explain the choice in `comments`.\n"
        f"- Allowed checkpoints: {choices}"
    )


def _model_context_instruction(config: dict[str, Any]) -> str:
    text = config["model_context"].strip()
    if not text:
        return ""
    return f"Additional model context supplied by the workflow:\n{text}"


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


def _validate_unused_payload(payload: dict[str, Any], config: dict[str, Any]) -> BaseModel:
    del payload, config
    raise ProviderError("This capability does not expose provider output payload.")


def _normalize_empty(inputs: dict[str, Any]) -> dict[str, Any]:
    del inputs
    return {}


def _resolve_empty(config: dict[str, Any]) -> dict[str, Any]:
    return dict(config)


def _normalize_model_context(inputs: dict[str, Any]) -> dict[str, Any]:
    value = inputs.get("model_context", "")
    if isinstance(value, (list, tuple)):
        raise ProviderError("Model Context does not support linked `model_context` inputs.")
    return {"model_context": str(value or "").strip()}


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
    normalize_config: Callable[[dict[str, Any]], dict[str, Any]]
    resolve_config: Callable[[dict[str, Any]], dict[str, Any]]
    build_prompt: Callable[[dict[str, Any]], str]
    build_schema: Callable[[dict[str, Any]], dict[str, Any] | None]
    validate_payload: Callable[[dict[str, Any], dict[str, Any]], BaseModel]


CAPABILITY_DEFINITIONS: dict[str, CapabilityDefinition] = {
    BASIC_PROMPT_CAPABILITY: CapabilityDefinition(
        capability_id=BASIC_PROMPT_CAPABILITY,
        module_class_type="BigPlayerBasicPrompt",
        produces_output=True,
        normalize_config=_normalize_empty,
        resolve_config=_resolve_empty,
        build_prompt=_basic_prompt_instruction,
        build_schema=_basic_prompt_schema,
        validate_payload=_validate_basic_prompt,
    ),
    SPLIT_PROMPT_CAPABILITY: CapabilityDefinition(
        capability_id=SPLIT_PROMPT_CAPABILITY,
        module_class_type="BigPlayerSplitPrompt",
        produces_output=True,
        normalize_config=_normalize_empty,
        resolve_config=_resolve_empty,
        build_prompt=_split_prompt_instruction,
        build_schema=_split_prompt_schema,
        validate_payload=_validate_split_prompt,
    ),
    KSAMPLER_CONFIG_CAPABILITY: CapabilityDefinition(
        capability_id=KSAMPLER_CONFIG_CAPABILITY,
        module_class_type="BigPlayerKSamplerConfig",
        produces_output=True,
        normalize_config=_normalize_empty,
        resolve_config=_resolve_ksampler,
        build_prompt=_ksampler_instruction,
        build_schema=_ksampler_schema,
        validate_payload=_validate_ksampler,
    ),
    CHECKPOINT_PICKER_CAPABILITY: CapabilityDefinition(
        capability_id=CHECKPOINT_PICKER_CAPABILITY,
        module_class_type="BigPlayerCheckpointPicker",
        produces_output=True,
        normalize_config=_normalize_empty,
        resolve_config=_resolve_checkpoint,
        build_prompt=_checkpoint_instruction,
        build_schema=_checkpoint_schema,
        validate_payload=_validate_checkpoint,
    ),
    MODEL_CONTEXT_CAPABILITY: CapabilityDefinition(
        capability_id=MODEL_CONTEXT_CAPABILITY,
        module_class_type="BigPlayerModelContext",
        produces_output=False,
        normalize_config=_normalize_model_context,
        resolve_config=_resolve_empty,
        build_prompt=_model_context_instruction,
        build_schema=lambda config: None,
        validate_payload=_validate_unused_payload,
    ),
}

MODULE_CLASS_TO_CAPABILITY = {
    definition.module_class_type: definition.capability_id for definition in CAPABILITY_DEFINITIONS.values()
}
