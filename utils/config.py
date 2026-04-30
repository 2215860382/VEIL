"""Config loader supporting `inherit:` chains and `${path.dot.notation}` interpolation."""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import yaml

_INTERP = re.compile(r"\$\{([^}]+)\}")


def _deep_merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def _resolve(value: Any, root: dict) -> Any:
    if isinstance(value, str):
        def repl(m):
            path = m.group(1).split(".")
            cur = root
            for p in path:
                cur = cur[p]
            return str(cur)
        new = _INTERP.sub(repl, value)
        return new
    if isinstance(value, dict):
        return {k: _resolve(v, root) for k, v in value.items()}
    if isinstance(value, list):
        return [_resolve(v, root) for v in value]
    return value


def load_config(path: str | Path) -> dict:
    """Load a YAML config, recursively resolving `inherit:` and ${...} interpolation."""
    path = Path(path).resolve()
    with open(path) as f:
        cfg = yaml.safe_load(f) or {}

    parent_name = cfg.pop("inherit", None)
    if parent_name:
        parent_path = path.parent / parent_name
        parent = load_config(parent_path)
        cfg = _deep_merge(parent, cfg)

    # Resolve ${a.b.c} interpolations until stable.
    for _ in range(10):
        new = _resolve(cfg, cfg)
        if new == cfg:
            break
        cfg = new
    return cfg
