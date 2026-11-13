"""How the agent runs a command, the same for every model: it ends its turn
with the command in `<bash>` tags. Strict: one command, at the very end of
the turn; anything else is not a call, and a turn without one ends the
episode. The models' own tool formats cost a 150-260 token preamble and
11-29 tokens a call; this one costs a line and 6.
"""

import re

FORMAT = (
    "Run a command by ending your turn with it in bash tags, e.g. "
    "<bash>ls</bash>. Its output comes back as the next message. One "
    "command per turn; a turn that does not end with one ends the episode."
)
"""What the system prompt says about it, the whole of what the model is told."""
CALL = re.compile(r"(.*?)<bash>(.*?)</bash>\s*", re.DOTALL)


def call(command: str) -> str:
    """Write a command as the agent does."""
    return f"<bash>{command}</bash>"


def command(answer: str) -> tuple[str, str | None]:
    """Split a turn's answer (what follows its reasoning) into the text
    before its command and the command; None unless the answer ends with
    exactly one `<bash>...</bash>`.
    """
    found = CALL.fullmatch(answer)
    if not found or "<bash>" in found[2] or "</bash>" in found[1]:
        return answer, None
    return found[1], found[2].strip()
