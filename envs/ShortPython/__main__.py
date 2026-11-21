"""Fetch the datasets into the dataset directory, once, on Mila's login
node: `cluv run mila python -m envs.ShortPython`. Then, in an allocation,
never on a login node, compute what each reference returns, `python -m
envs.ShortPython expect`, then run each Python and JavaScript solution
the agent edits on its calls and keep those that return it all, `python
-m envs.ShortPython verify` (in sandboxes: node is in their image).
"""

import sys

from envs.ShortPython.data import FILES, expect, fetch, verify

if sys.argv[1:] == ["expect"]:
    for split in FILES:
        expect(split)
elif sys.argv[1:] == ["verify"]:
    for split in FILES:
        verify(split)
else:
    fetch()
