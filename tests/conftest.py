import importlib.util

# The optional ML extension needs PyTorch; skip its tests where it isn't installed.
collect_ignore = [] if importlib.util.find_spec("torch") else ["ml"]
