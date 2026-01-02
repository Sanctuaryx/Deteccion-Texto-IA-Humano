from __future__ import annotations

import pathlib
from typing import Any, Dict

import yaml


def load_yaml(path: str | pathlib.Path) -> Dict[str, Any]:
    path = pathlib.Path(path)
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)
