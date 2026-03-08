# ComfyUI LLM Prompt Nodes

A ComfyUI extension for LLM-assisted prompt generation, LoRA recommendation, and iterative prompt refinement.

This project converts freeform user prose into structured prompts for image and video generation workflows using an external LLM provider. Initial provider support targets **Grok**, but the architecture is intended to support additional providers later.

## Status

Phase 1 scaffold implemented.

Current project focus:
- Phase 1: prose to prompts
- Grok provider integration
- simple and split prompt output nodes under `BigPlayer/Prompting`
- strict schema-bound structured outputs
- deterministic caching option for repeatable workflows

## What this project does

This extension is intended to provide ComfyUI nodes that:
- convert natural-language prose into positive and negative prompts;
- support both standard prompt output and split prompt output;
- later recommend suitable LoRAs from the local ComfyUI environment;
- later refine prompts iteratively by reviewing generated images.

This is an **LLM-assisted** system. Prompt transformation and selection logic are intentionally delegated to the LLM. Local code is responsible for input preparation, schema enforcement, caching, validation, and execution of validated outputs.

## Planned nodes

### 1. Simple Prompt Node

Takes user prose plus a connected `MODEL` and returns:
- `positive_prompt`
- `negative_prompt`
- `comments`

### 2. Split Prompt Node

Takes user prose plus a connected `MODEL` and returns:
- `text_l_positive`
- `text_g_positive`
- `text_l_negative`
- `text_g_negative`
- `comments`

## Development setup

Create and use the project-local virtual environment before doing any work:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -r .integration/ComfyUI/requirements.txt
.venv/bin/python -m pip install -e .[dev]
```

This repository also keeps a local ComfyUI checkout at `.integration/ComfyUI` for integration testing. The working tree is symlinked into that checkout's `custom_nodes` directory so tests exercise the current source directly.

## Testing

Run the default unit suite:

```bash
.venv/bin/pytest tests/unit
```

Run the mocked ComfyUI integration suite:

```bash
.venv/bin/pytest tests/integration -m integration
```

Run opt-in real Grok validation:

```bash
BIGPLAYER_GROK_LIVE_TEST=1 \
BIGPLAYER_GROK_API_KEY=... \
.venv/bin/pytest tests/unit/test_live_provider.py -m live
```

### 3. LoRA-Aware Prompt Node

Planned extension of the prompt-generation flow.

Returns prompt outputs plus a structured LoRA recommendation list such as:

```json
[
  {
    "lora_name": "example_lora",
    "clip_strength": 1.0,
    "model_strength": 1.0
  }
]
```

### 4. LoRA Review Node

Consumes the structured LoRA recommendation list and presents it clearly for user review and manual handling.

### 5. LoRA Apply Node

Consumes validated LoRA recommendations and attempts to apply them in the workflow.

### 6. Prompt Refinement Node

Takes a generated image plus prior prompt context, asks the LLM to critique the result against the original intent, and returns revised prompt outputs.

## Design principles

- **Structured output first**  
  Operational outputs must use schema-bound structured responses wherever feasible.

- **Provider-native schema enforcement**  
  Where supported, the provider must be asked to return output matching a declared schema.

- **Local validation remains mandatory**  
  Even with provider-side structured output, local code still validates responses before use.

- **No fake magic**  
  This project does not claim to contain hidden internal prompt intelligence. Where the LLM is making substantive decisions, that is explicit.

- **Determinism is optional**  
  Each LLM-backed node will support an `assume_determinism` option so identical inputs can reuse cached outputs when desired.

- **Execution is separate from selection**  
  The LLM may choose prompts or LoRAs, but local code validates and executes those choices.

## Reliability expectations

This project is being built to avoid the usual fragile “ask an LLM for JSON and hope” pattern.

The intended behaviour is:
- require schema-bound structured responses;
- reject malformed or incomplete outputs;
- validate all required fields and types locally;
- use explicit network timeouts;
- fail clearly and predictably;
- avoid silent fallbacks that hide upstream errors.

## Secrets and safety

- API keys should not be treated as casual workflow text inputs if avoidable.
- Secrets must not be logged.
- UI comments must never echo secrets.
- Documentation should describe the system honestly and avoid inflated claims.

## Roadmap

### Phase 1
- Grok provider adapter
- internal response schemas
- shared prompt-generation service
- simple prompt node
- split prompt node
- caching and validation behaviour

### Phase 2
- LoRA inventory support
- schema-bound LoRA recommendation output
- LoRA review node
- LoRA application node

### Phase 3
- image-based prompt refinement
- structured critique output
- iterative refinement workflows

## Intended architecture

The codebase is expected to separate concerns cleanly, with boundaries roughly along these lines:
- `nodes/` — ComfyUI node classes
- `providers/` — LLM provider adapters
- `prompts/` — prompt templates and instructions
- `schemas/` or `models/` — structured response models
- `services/` — orchestration logic
- `utils/` — hashing, cache helpers, logging, error mapping

## Current direction

This repository is being built as a proper ComfyUI extension, not as a thin wrapper around a chat completion endpoint. The goal is a predictable, inspectable, schema-driven toolset that is useful in real prompt workflows and honest about where its intelligence actually lives.

## Development note

The detailed implementation intent, delivery order, and architectural constraints are defined in `PROJECT_BRIEF.md`.
