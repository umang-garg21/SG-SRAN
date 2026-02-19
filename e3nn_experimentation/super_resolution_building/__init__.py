import sys
import os

_project_root = "/home/warren/projects/Reynolds-QSR/"
_paths = [
    _project_root,
    os.path.join(_project_root, "utils"),
    os.path.join(_project_root, "e3nn_experimentation"),
]
for _p in _paths:
    if _p not in sys.path:
        sys.path.insert(0, _p)
