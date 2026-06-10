"""Write a human-readable record of how a memory bank directory was produced."""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

FILENAME = "MEMORY_BUILD_README.txt"


def write_memory_build_readme(
    out_dir: str | Path,
    *,
    title: str,
    lines: Iterable[str],
    filename: str = FILENAME,
) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    body = [title.strip(), f"generated_utc: {stamp}", ""] + [ln.rstrip() for ln in lines]
    (out / filename).write_text("\n".join(body) + "\n", encoding="utf-8")
