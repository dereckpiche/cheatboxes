"""Where datasets live, as cluv syncs them (pyproject.toml `[tool.cluv]`).

Their home is Mila (`data_source`); `cluv sync` copies them to another
cluster's `datasets_path` when a job goes there. In a job,
`cluv.job.get_datasets_path()` gives the right one for the cluster;
`training.run` exports it as `DATA_DIR` so every worker reads it.
"""

from __future__ import annotations

import os
from pathlib import Path


def root() -> Path:
    """The dataset directory: `DATA_DIR`, else cluv's `datasets_path`."""
    if found := os.environ.get("DATA_DIR"):
        return Path(found)
    from cluv.job import get_datasets_path

    found = get_datasets_path()
    if found is None:
        raise RuntimeError("no datasets_path in [tool.cluv]")
    return found
