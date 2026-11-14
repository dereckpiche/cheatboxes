"""Fetch the dataset into the dataset directory, once, on Mila's login
node: `cluv run mila python -m envs.ShortPython`.
"""

from envs.ShortPython.data import fetch

fetch()
