"""Fetch the dataset into the dataset directory, once, on Mila's login
node: `cluv run mila python -m envs.ShortPython`. Then compute what each
reference returns, in an allocation, never on a login node:
`python -m envs.ShortPython expect`.
"""

import sys

from envs.ShortPython.data import FILES, expect, fetch

if sys.argv[1:] == ["expect"]:
    for split in FILES:
        expect(split)
else:
    fetch()
