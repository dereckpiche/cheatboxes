"""The tool-call parser for `envs.commons.tool`'s format, registered with
verl on import as `bash`, which `train.yaml` names under
`multi_turn.format`.
"""

from __future__ import annotations

import json
import re

from verl.experimental.agent_loop.tool_parser import FunctionCall, ToolParser

from envs.commons import tool

REASONING_END = re.compile(r"</think>")
TURN_ENDS = ("<|im_end|>",)
"""Qwen's end-of-turn token."""


@ToolParser.register("bash")
class BashTagParser(ToolParser):
    """Reads the command a turn ends with as a call to the bash tool."""

    @property
    def stop_token_ids(self) -> list[int]:
        """Stop at the end of the turn."""
        vocabulary = self.tokenizer.get_vocab()
        return [vocabulary[t] for t in TURN_ENDS if t in vocabulary]

    async def extract_tool_calls(self, responses_ids, tools=None):
        """Split a turn into its text before the command and the command,
        which counts only after the reasoning.
        """
        text = self.tokenizer.decode(responses_ids, skip_special_tokens=False)
        for end in TURN_ENDS:
            text = text.removesuffix(end)
        ends = list(REASONING_END.finditer(text))
        cut = ends[-1].end() if ends else 0
        before, command = tool.command(text[cut:])
        if command is None:
            return text, []
        arguments = json.dumps({"command": command}, ensure_ascii=False)
        call = FunctionCall(name="bash", arguments=arguments)
        return text[:cut] + before, [call]
