from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any


class ConfigError(Exception):
    """Raised when config parsing or validation fails."""


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in updates.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _parse_scalar(raw: str) -> Any:
    raw = raw.strip()
    if raw == "":
        return None
    if raw.lower() in {"true", "false"}:
        return raw.lower() == "true"
    if raw.startswith("[") and raw.endswith("]"):
        inside = raw[1:-1].strip()
        if not inside:
            return []
        return [_parse_scalar(part.strip()) for part in inside.split(",")]
    try:
        if "." in raw:
            return float(raw)
        return int(raw)
    except ValueError:
        return raw.strip('"').strip("'")


def _parse_simple_yaml(text: str) -> dict[str, Any]:
    root: dict[str, Any] = {}
    stack: list[tuple[int, dict[str, Any]]] = [(-1, root)]
    for lineno, line in enumerate(text.splitlines(), start=1):
        if not line.strip() or line.strip().startswith("#"):
            continue
        indent = len(line) - len(line.lstrip(" "))
        if indent % 2 != 0:
            raise ConfigError(f"Invalid indentation at line {lineno}")
        content = line.strip()
        if ":" not in content:
            raise ConfigError(f"Invalid line {lineno}: {line}")
        key, value = content.split(":", 1)
        while stack and indent <= stack[-1][0]:
            stack.pop()
        current = stack[-1][1]
        value = value.strip()
        if value == "":
            child: dict[str, Any] = {}
            current[key] = child
            stack.append((indent, child))
        else:
            current[key] = _parse_scalar(value)
    return root


def load_config(config_path: Path) -> dict[str, Any]:
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")
    text = config_path.read_text(encoding="utf-8")
    data = _parse_simple_yaml(text)

    defaults: dict[str, Any] = {
        "run_name": "baseline",
        "data": {"path": "city_day.csv", "target_column": None},
        "split": {"test_size": 0.2, "random_state": 42},
        "model": {
            "task": "auto",
            "type": "mlp",
            "params": {"hidden_layer_sizes": [50], "activation": "relu", "alpha": 0.0001, "max_iter": 500},
        },
        "output": {"results_dir": "results"},
    }
    return _deep_merge(defaults, data)
