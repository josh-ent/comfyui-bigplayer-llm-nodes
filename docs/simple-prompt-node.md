# BigPlayer Basic Prompt

Reads the `basic_prompt` portion of a shared structured LLM response.

This node does not call the provider on its own. It consumes the shared `session` emitted by `BigPlayer LLM Root`. The root is responsible for discovering this module, including the `basic_prompt` capability in the provider request, making one LLM call, and validating the combined structured response.

Input:
- `session`

Outputs:
- `positive_prompt`
- `negative_prompt`
- `comments`

Notes:
- Multiple `BigPlayer Basic Prompt` nodes may attach to the same root and will return the same validated result.
- If no compatible root session exists, the node fails rather than making an extra provider call.
