from __future__ import annotations

from dataclasses import dataclass, replace
import os
import re
from typing import Any

from .errors import ProviderError


LORA_SYNTAX_PATTERN = re.compile(r"<lora:([^:>]+):([-\d.]+)(?::([-\d.]+))?>", re.IGNORECASE)
NONE_OPTION = "<none>"


@dataclass(frozen=True)
class PresetLora:
    name: str
    relative_path: str
    model_strength: float
    clip_strength: float


@dataclass(frozen=True)
class PresetConfigBundle:
    checkpoint_name: str = ""
    refiner_checkpoint_name: str = ""
    loras: tuple[PresetLora, ...] = ()
    controlnets: tuple[str, ...] = ()


def normalize_preset_config(value: Any) -> PresetConfigBundle | None:
    if value is None:
        return None
    if isinstance(value, PresetConfigBundle):
        return value
    raise ProviderError("Preset Config must come from a BigPlayer state-indication node or be omitted.")


def list_available_controlnets() -> list[str]:
    try:
        import folder_paths
    except ImportError:
        return []
    return sorted(folder_paths.get_filename_list("controlnet"))


def with_checkpoint_state(
    bundle: PresetConfigBundle | None,
    *,
    checkpoint_name: str,
    refiner_checkpoint_name: str,
) -> PresetConfigBundle:
    base = bundle or PresetConfigBundle()
    return replace(
        base,
        checkpoint_name=_normalize_combo_value(checkpoint_name),
        refiner_checkpoint_name=_normalize_combo_value(refiner_checkpoint_name),
    )


def with_lora_state(
    bundle: PresetConfigBundle | None,
    *,
    manual_syntax: str,
    linked_syntax: Any = None,
    lora_stack: Any = None,
) -> PresetConfigBundle:
    base = bundle or PresetConfigBundle()
    loras: list[PresetLora] = []
    loras.extend(_parse_lora_syntax(manual_syntax))
    loras.extend(_parse_lora_syntax(_coerce_text_value(linked_syntax, "linked LoRA syntax")))
    loras.extend(_parse_lora_stack(lora_stack))
    return replace(base, loras=tuple(_dedupe_loras(loras)))


def with_controlnet_state(
    bundle: PresetConfigBundle | None,
    *,
    manual_controlnets: str,
    linked_controlnets: Any = None,
) -> PresetConfigBundle:
    base = bundle or PresetConfigBundle()
    manual = _parse_string_list(manual_controlnets)
    linked = _parse_string_list(_coerce_text_value(linked_controlnets, "linked ControlNet input"))
    return replace(base, controlnets=tuple(_dedupe_strings([*manual, *linked])))


def render_preset_config(bundle: PresetConfigBundle | None) -> str:
    if bundle is None:
        return ""

    lines: list[str] = []
    if bundle.checkpoint_name:
        lines.append(f"- Checkpoint: {bundle.checkpoint_name}")
    if bundle.refiner_checkpoint_name:
        lines.append(f"- Refiner checkpoint: {bundle.refiner_checkpoint_name}")
    if bundle.loras:
        lines.append("- LoRAs:")
        for lora in bundle.loras:
            line = f"  - {_format_lora_syntax(lora)}"
            if lora.relative_path:
                line += f" [path: {lora.relative_path}]"
            lines.append(line)
    if bundle.controlnets:
        lines.append("- ControlNets:")
        for controlnet in bundle.controlnets:
            lines.append(f"  - {controlnet}")

    if not lines:
        return ""
    return "Workflow preset state provided explicitly:\n" + "\n".join(lines)


def serialize_preset_config(bundle: PresetConfigBundle | None) -> dict[str, Any]:
    if bundle is None:
        return {}
    return {
        "checkpoint_name": bundle.checkpoint_name,
        "refiner_checkpoint_name": bundle.refiner_checkpoint_name,
        "loras": [
            {
                "name": lora.name,
                "relative_path": lora.relative_path,
                "model_strength": lora.model_strength,
                "clip_strength": lora.clip_strength,
            }
            for lora in bundle.loras
        ],
        "controlnets": list(bundle.controlnets),
    }


def _normalize_combo_value(value: str) -> str:
    text = str(value or "").strip()
    return "" if text == NONE_OPTION else text


def _parse_lora_syntax(text: str) -> list[PresetLora]:
    text = text.strip()
    if not text:
        return []

    matches = list(LORA_SYNTAX_PATTERN.finditer(text))
    if not matches:
        raise ProviderError(
            "LoRA State expects LoRA Manager syntax like `<lora:name:0.6>` or `<lora:name:0.6:0.4>`."
        )

    remainder = LORA_SYNTAX_PATTERN.sub(" ", text)
    remainder = re.sub(r"[\s,;|]+", " ", remainder).strip()
    if remainder:
        raise ProviderError(
            "LoRA State received invalid LoRA syntax. Use only LoRA Manager tags separated by spaces or punctuation."
        )

    parsed: list[PresetLora] = []
    for match in matches:
        raw_name = match.group(1).strip()
        model_strength = _parse_float(match.group(2), "LoRA model strength")
        clip_strength = _parse_float(match.group(3), "LoRA clip strength") if match.group(3) else model_strength
        parsed.append(
            PresetLora(
                name=_extract_lora_name(raw_name),
                relative_path="",
                model_strength=model_strength,
                clip_strength=clip_strength,
            )
        )
    return parsed


def _parse_lora_stack(value: Any) -> list[PresetLora]:
    if value is None:
        return []
    if not isinstance(value, (list, tuple)):
        raise ProviderError("LoRA State expected `lora_stack` to be a LORA_STACK tuple list.")

    parsed: list[PresetLora] = []
    for entry in value:
        if not isinstance(entry, (list, tuple)) or len(entry) < 2:
            raise ProviderError("LoRA State received an invalid LORA_STACK entry.")
        relative_path = str(entry[0] or "").strip()
        if not relative_path:
            raise ProviderError("LoRA State received an empty relative path in LORA_STACK.")
        model_strength = _parse_float(entry[1], "LoRA model strength")
        clip_strength = _parse_float(entry[2], "LoRA clip strength") if len(entry) > 2 else model_strength
        parsed.append(
            PresetLora(
                name=_extract_lora_name(relative_path),
                relative_path=relative_path,
                model_strength=model_strength,
                clip_strength=clip_strength,
            )
        )
    return parsed


def _parse_string_list(text: str) -> list[str]:
    parts = re.split(r"[\n,;]+", text or "")
    return [part.strip() for part in parts if part and part.strip()]


def _coerce_text_value(value: Any, label: str) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        fragments: list[str] = []
        for item in value:
            if not isinstance(item, str):
                raise ProviderError(f"{label} must be a string or list of strings.")
            fragments.append(item)
        return "\n".join(fragments)
    raise ProviderError(f"{label} must be a string or list of strings.")


def _dedupe_loras(loras: list[PresetLora]) -> list[PresetLora]:
    deduped: dict[str, PresetLora] = {}
    for lora in loras:
        key = lora.relative_path.lower() if lora.relative_path else lora.name.lower()
        deduped[key] = lora
    return list(deduped.values())


def _dedupe_strings(items: list[str]) -> list[str]:
    deduped: dict[str, str] = {}
    for item in items:
        deduped[item.lower()] = item
    return list(deduped.values())


def _extract_lora_name(value: str) -> str:
    basename = os.path.basename(value.strip())
    return os.path.splitext(basename)[0]


def _format_lora_syntax(lora: PresetLora) -> str:
    if abs(lora.model_strength - lora.clip_strength) > 0.001:
        return f"<lora:{lora.name}:{lora.model_strength}:{lora.clip_strength}>"
    return f"<lora:{lora.name}:{lora.model_strength}>"


def _parse_float(value: Any, label: str) -> float:
    try:
        return float(value)
    except (TypeError, ValueError) as exc:
        raise ProviderError(f"{label} must be numeric.") from exc
