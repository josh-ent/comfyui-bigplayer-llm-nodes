from __future__ import annotations

from typing import Any

from .capabilities import CAPABILITY_DEFINITIONS
from .errors import MalformedProviderResponseError


def get_provider_schema(capability_configs: dict[str, dict[str, Any]]) -> dict[str, Any]:
    properties: dict[str, Any] = {}
    required: list[str] = []
    for capability_id, config in capability_configs.items():
        schema_builder = CAPABILITY_DEFINITIONS[capability_id].build_schema
        schema = schema_builder(config)
        if schema is None:
            continue
        properties[capability_id] = schema
        required.append(capability_id)

    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def validate_result(
    capability_configs: dict[str, dict[str, Any]],
    payload: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    if not isinstance(payload, dict):
        raise MalformedProviderResponseError("Provider returned a non-object structured payload.")

    expected = set(capability_id for capability_id in capability_configs if CAPABILITY_DEFINITIONS[capability_id].produces_output)
    actual = set(payload)
    if actual != expected:
        raise MalformedProviderResponseError(
            f"Provider response failed schema validation: expected top-level keys {sorted(expected)}, got {sorted(actual)}."
        )

    validated: dict[str, dict[str, Any]] = {}
    for capability_id in expected:
        raw_value = payload.get(capability_id)
        if not isinstance(raw_value, dict):
            raise MalformedProviderResponseError(
                f"Provider response failed schema validation: `{capability_id}` must be an object."
            )
        model = CAPABILITY_DEFINITIONS[capability_id].validate_payload(raw_value, capability_configs[capability_id])
        validated[capability_id] = model.model_dump()
    return validated
