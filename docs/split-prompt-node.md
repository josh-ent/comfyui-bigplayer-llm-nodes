# BigPlayer Split Prompt

Reads the `split_prompt` portion of a shared structured LLM response.

This node does not call the provider directly. It consumes the shared `session` emitted by `BigPlayer Natural Language Root`. The root discovers this module, asks the provider for the `split_prompt` capability, and validates the combined response before this node reads its own slice.

Input:
- `session`

Outputs:
- `text_l_positive`
- `text_g_positive`
- `text_l_negative`
- `text_g_negative`
- `comments`

Notes:
- The split output is intended for workflows that distinguish local and global text channels.
- Multiple `BigPlayer Split Prompt` nodes may attach to the same root and will return the same validated result.
