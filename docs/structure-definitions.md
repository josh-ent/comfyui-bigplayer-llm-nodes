# BigPlayer Structure Definitions

BigPlayer exposes three reusable workflow structures as explicit Python dataclasses rather than anonymous dictionaries.

These are the shared contracts other nodes and extensions should target.

## `provider_config`

Python type: `bigplayer.LLMProviderBundle`

Defined in [`bigplayer/generation/service.py`](../bigplayer/generation/service.py).

Fields:

```python
LLMProviderBundle(
    api_key: str,
    provider: str,
    provider_model: str,
    provider_base_url: str = "",
    assume_determinism: bool = True,
)
```

Produced by:

- `BigPlayer LLM Provider`

Consumed by:

- `BigPlayer Natural Language Root`

## `session`

Python type: `bigplayer.LLMSessionHandle`

Defined in [`bigplayer/generation/service.py`](../bigplayer/generation/service.py).

Fields:

```python
LLMSessionHandle(
    session_id: str,
    root_node_id: str,
    cache_key: str,
)
```

Produced by:

- `BigPlayer Natural Language Root`

Consumed by:

- `BigPlayer Basic Prompt`
- `BigPlayer Split Prompt`
- `BigPlayer KSampler Config`
- `BigPlayer Checkpoint Picker`

## `preset_config`

Python type: `bigplayer.PresetConfigBundle`

Defined in [`bigplayer/state/preset.py`](../bigplayer/state/preset.py).

Fields:

```python
PresetConfigBundle(
    checkpoint_name: str = "",
    refiner_checkpoint_name: str = "",
    loras: tuple[PresetLora, ...] = (),
    controlnets: tuple[str, ...] = (),
)
```

The nested LoRA entry type is:

```python
PresetLora(
    name: str,
    relative_path: str,
    model_strength: float,
    clip_strength: float,
)
```

Produced by:

- `BigPlayer Checkpoint State`
- `BigPlayer LoRA State`
- `BigPlayer ControlNet State`

Consumed by:

- `BigPlayer Natural Language Root`
- other BigPlayer state-indication nodes

Serialized form used for hashing and transport:

```python
{
    "checkpoint_name": str,
    "refiner_checkpoint_name": str,
    "loras": [
        {
            "name": str,
            "relative_path": str,
            "model_strength": float,
            "clip_strength": float,
        }
    ],
    "controlnets": [str],
}
```

## Public Imports

For external reuse, these contracts are re-exported from `bigplayer`:

```python
from bigplayer import LLMProviderBundle, LLMSessionHandle, PresetConfigBundle, PresetLora
```
