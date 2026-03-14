# Prompt Node Project Brief

## Implementation note

This brief describes the original phase ordering and product intent.

The current implementation has since moved to a provider/root/module architecture:
- one provider node builds provider config;
- one root node performs a single LLM call for all attached modules;
- module nodes expose individual outputs from the shared validated response;
- provider adapters own prompt fragment text and provider-facing schema composition.

Treat the remainder of this document as the original delivery brief rather than a line-by-line description of the current node surface.

## Purpose

This repository will implement a ComfyUI extension that uses an LLM to transform user prose into structured prompts suitable for image and video generation workflows.

Initial provider integration will target Grok first, but the architecture must not hard-code Grok-specific assumptions beyond the first provider adapter.

The project is intended to be:
- technically credible;
- explicit about where behaviour is deterministic vs LLM-derived;
- safe with respect to secrets handling;
- modular enough to support later provider expansion;
- useful in real ComfyUI workflows, not merely a thin wrapper around a chat completion API.

---

## Product goals

The extension should:
- convert natural-language prose into usable positive and negative prompts;
- support both a simple prompt format and a split prompt format;
- understand the target generation model and tailor output accordingly;
- later recommend and integrate suitable LoRAs from the local ComfyUI environment;
- later support iterative refinement by reviewing generated image outputs and proposing improved prompts.

The extension must not pretend to contain internal prompt intelligence where it is in fact merely delegating to an LLM. The system must be honest in both implementation and documentation.

---

## Non-goals for the initial release

The first release is not expected to:
- support every LLM provider;
- provide perfect prompt generation for every model family;
- guarantee that the LLM output is semantically correct;
- automate every part of the ComfyUI graph;
- perform deep image critique comparable to a specialised vision pipeline.

The first release must instead focus on clean architecture, reliable node behaviour, and predictable outputs.

---

## Delivery order

Features should be implemented in this order.

### Phase 1 — Prose to prompts

Implement LLM-backed prompt generation from freeform prose.

#### Core requirements

The node must:
- accept a prose description from the user;
- accept metadata describing the target model or target prompt format;
- send the prompt plus model-context instructions to the LLM;
- return structured prompt outputs in a validated format;
- expose comments or explanation for the user.

#### Node variants

Implement two node types for the same underlying function.

##### 1. Simple prompt node

Inputs:
- user prose;
- target model information;
- optional style or policy inputs if useful;
- provider configuration as required.

Outputs:
- `positive_prompt`
- `negative_prompt`
- `comments`

##### 2. Split prompt node

Inputs:
- user prose;
- target model information;
- optional style or policy inputs if useful;
- provider configuration as required.

Outputs:
- `text_l_positive`
- `text_g_positive`
- `text_l_negative`
- `text_g_negative`
- `comments`

#### Behavioural requirements

- The same underlying transformation logic should power both nodes.
- The split node must not simply duplicate the full prompt into both channels unless that is an explicit fallback.
- The target model definition must materially affect prompt shaping.
- The implementation must validate the LLM response before returning outputs.
- The node must fail predictably when the LLM response is malformed or incomplete.

---

### Phase 2 — LoRA awareness and integration

Extend the Phase 1 nodes so that they can consider locally available LoRAs and return structured LoRA recommendations.

#### Core requirements

The system should:
- accept or discover the list of available LoRAs in the ComfyUI environment;
- provide that list to the LLM in a controlled form;
- have the LLM select appropriate LoRAs for the requested prompt;
- return both the prompt text and a structured LoRA recommendation list.

#### Structured LoRA output

The first LoRA-aware prompt node should output an array of objects in a form equivalent to:

```json
[
  {
    "lora_name": "example_lora",
    "clip_strength": 1.0,
    "model_strength": 1.0
  }
]
