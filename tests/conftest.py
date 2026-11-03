"""Leave out the training tests when the `train` extra is not installed."""

import importlib.util

collect_ignore = (
    []
    if importlib.util.find_spec("verl")
    else ["test_training.py", "test_methods.py"]
)
