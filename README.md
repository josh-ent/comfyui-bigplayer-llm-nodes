# ComfyUI LLM Prompt Nodes

A ComfyUI extension for single-call, modular, LLM-assisted prompt and workflow configuration.

This project converts freeform user prose into structured prompt and workflow data for image generation pipelines using an external LLM provider. Initial provider support targets **Grok/xAI**, but the architecture is intended to support additional providers with their own prompt style and schema-fragment handling.

## Status

Modular root-plus-module scaffold implemented.

Current project focus:
- provider/root/module prompt architecture
- Grok provider integration
- provider-owned prompt/schema composition
- modular output nodes under `BigPlayer/Prompting`
- strict schema-bound structured outputs
- deterministic caching option for repeatable workflows

## What this project does

This extension is intended to provide ComfyUI nodes that:
- convert natural-language prose into prompt and workflow outputs;
- support standard prompt output, split prompt output, checkpoint selection, and KSampler config selection;
- later recommend suitable LoRAs from the local ComfyUI environment;
- later refine prompts iteratively by reviewing generated images.

This is an **LLM-assisted** system. Prompt transformation and selection logic are intentionally delegated to the LLM. Local code is responsible for input preparation, schema enforcement, caching, validation, and execution of validated outputs.

## Current node flow

The current workflow shape is:

`BigPlayer LLM Provider` -> `BigPlayer Natural Language Root` -> one or more module nodes

The root discovers attached modules, builds one provider request, validates one structured response, and publishes a shared session. Each module reads only its own validated result slice.

### Core nodes

#### 1. BigPlayer LLM Provider

Provides:
- `provider`
- `provider_model`
- `api_key`
- `provider_base_url`
- `assume_determinism`

Outputs:
- `provider_config`

#### 2. BigPlayer Natural Language Root

Takes:
- `prose`
- `provider_config`

Outputs:
- `session`

#### 3. BigPlayer Basic Prompt

Takes:
- `session`

Outputs:
- `positive_prompt`
- `negative_prompt`
- `comments`

#### 4. BigPlayer Split Prompt

Takes:
- `session`

Outputs:
- `text_l_positive`
- `text_g_positive`
- `text_l_negative`
- `text_g_negative`
- `comments`

#### 5. BigPlayer KSampler Config

Takes:
- `session`

Outputs:
- `steps`
- `cfg`
- `sampler_name`
- `scheduler`
- `denoise`
- `comments`

#### 6. BigPlayer Checkpoint Picker

Takes:
- `session`

Outputs:
- `checkpoint_name`
- `comments`

#### 7. BigPlayer Model Context

Takes:
- `session`
- `model_context`

This module contributes extra context to the provider request but does not expose its own final payload.

## Development setup

Create and activate a project-local virtual environment before doing any work, then install the repo's Python dependencies:

```bash
python -m venv .venv
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e .[dev]
npm install
npx playwright install chromium
```

Integration tests no longer depend on an untracked local ComfyUI checkout. They build a pinned ComfyUI Docker image on first use and mount the current working tree into the container's `custom_nodes` directory so tests always exercise the current source directly.

Container prerequisites:
- Docker Desktop or another local Docker runtime must be available.
- The first integration run will build the pinned ComfyUI test image, so it takes longer than subsequent runs.

## Testing

Run the default unit suite:

```bash
python -m pytest tests/unit
```

Run the mocked ComfyUI integration suite:

```bash
python -m pytest tests/integration -m integration
```

Run the Playwright UI suite against the same Docker-backed ComfyUI test runtime:

```bash
npm run test:ui
```

Run the full integration pass:

```bash
npm run test:full
```

Run opt-in real Grok validation after setting `BIGPLAYER_GROK_LIVE_TEST=1` and `BIGPLAYER_GROK_API_KEY` in your shell environment:

```bash
python -m pytest tests/unit/test_live_provider.py -m live
```

## Composition model

- Modules contribute normalized contracts and local environment-derived config.
- Providers own prompt fragment text and provider-facing schema fragments.
- The root performs exactly one provider call for all attached modules.
- Duplicate identical modules are allowed and return the same shared result.
- Conflicting duplicate module configs on the same root fail before the provider call.

## Planned nodes

### 1. LoRA-Aware Prompt Module

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

### 2. LoRA Review Node

Consumes the structured LoRA recommendation list and presents it clearly for user review and manual handling.

### 3. LoRA Apply Node

Consumes validated LoRA recommendations and attempts to apply them in the workflow.

### 4. Prompt Refinement Node

Takes a generated image plus prior prompt context, asks the LLM to critique the result against the original intent, and returns revised prompt outputs.

## Design principles

- **Structured output first**  
  Operational outputs must use schema-bound structured responses wherever feasible.

- **Provider-native schema enforcement**  
  Where supported, the provider must be asked to return output matching a declared schema assembled from provider-owned fragments.

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
- modular capability contracts
- shared prompt-generation service
- provider node
- root session node
- basic prompt module
- split prompt module
- KSampler config module
- checkpoint picker module
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
- `prompts/` — provider-owned prompt fragments and instructions
- `schemas/` or `models/` — provider result validation models
- `services/` — orchestration logic
- `utils/` — hashing, cache helpers, logging, error mapping

## Current direction

This repository is being built as a proper ComfyUI extension, not as a thin wrapper around a chat completion endpoint. The goal is a predictable, inspectable, schema-driven toolset that is useful in real prompt workflows and honest about where its intelligence actually lives.

## Development note

The detailed implementation intent, delivery order, and architectural constraints are defined in `PROJECT_BRIEF.md`.
