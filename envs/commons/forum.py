"""The forum: one host folder shared by every episode of a run, mounted at
`/forum` in each sandbox, so agents can leave each other notes. training.run
points FORUM at the run directory.
"""

import os
from pathlib import Path


def mount() -> str:
    """The forum as a sandbox mount, its host folder created if needed."""
    folder = Path(os.environ.get("FORUM", "/tmp/cheatboxes-forum"))
    folder.mkdir(parents=True, exist_ok=True)
    return f"{folder}:/forum"
