from __future__ import annotations

from .schemas import PromptMode


SYSTEM_PROMPT = """You convert user prose into production-ready ComfyUI prompts.

You are not allowed to invent deterministic local logic that the node does not have.
Return only the schema requested by the caller.
Use the provided model name to tailor the prompt. When useful, use web search to look up
that model on CivitAI or HuggingFace before deciding on prompt shape.
Keep prompt text practical and immediately usable in image or video workflows.
Comments must explain how the model name influenced the result and note any fallback.
"""


def build_user_prompt(*, prose: str, model_name: str, style_policy: str, mode: PromptMode) -> str:
    if mode == "simple":
        output_instructions = (
            "Return `positive_prompt`, `negative_prompt`, and `comments`."
        )
    else:
        output_instructions = (
            "Return `text_l_positive`, `text_g_positive`, `text_l_negative`, "
            "`text_g_negative`, and `comments`. Do not duplicate the full prompt into both "
            "positive channels unless you explicitly treat that as a fallback and say so in comments."
        )

    style_section = style_policy.strip() or "No extra style policy supplied."

    return f"""User prose:
{prose.strip()}

Target model name:
{model_name}

Style or policy guidance:
{style_section}

Output mode:
{mode}

Requirements:
- Tailor the prompt to the target model name.
- Keep the output concise, specific, and workflow-ready.
- Negative prompts should omit content instead of explaining policy.
- {output_instructions}
- Return only data matching the requested schema.
"""

