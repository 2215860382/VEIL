#!/usr/bin/env python3
"""Drop all records for one ``pipeline`` from a run_experiments JSONL (for resume).

After stripping ``veil_coarse8_27b`` rows, restart eval with the same ``--out`` and
``--pipelines veil_coarse8_27b`` only; other pipelines stay skipped (keys still present).

Usage:
  python scripts/strip_pipeline_from_jsonl.py outputs/results/videomme/foo.jsonl veil_coarse8_27b
  python scripts/strip_pipeline_from_jsonl.py path/to.jsonl veil_coarse8_27b --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from tempfile import NamedTemporaryFile


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("jsonl", type=Path, help="Path to .jsonl written by run_experiments.py")
    ap.add_argument("pipeline", help="Pipeline name to remove, e.g. veil_coarse8_27b")
    ap.add_argument("--dry-run", action="store_true", help="Print counts only; do not write")
    args = ap.parse_args()
    path: Path = args.jsonl
    if not path.is_file():
        print(f"error: not a file: {path}", file=sys.stderr)
        sys.exit(1)

    kept = dropped = bad = 0
    lines_out: list[str] = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                bad += 1
                continue
            if rec.get("pipeline") == args.pipeline:
                dropped += 1
                continue
            lines_out.append(json.dumps(rec, ensure_ascii=False) + "\n")
            kept += 1

    print(f"kept={kept} dropped={dropped} bad_lines={bad} path={path}")
    if args.dry_run:
        return

    if dropped == 0:
        print("nothing to do (no matching records)")
        return

    text = "".join(lines_out)
    with NamedTemporaryFile("w", encoding="utf-8", delete=False, dir=path.parent) as tmp:
        tmp.write(text)
        tmp_path = Path(tmp.name)
    tmp_path.replace(path)
    print(f"wrote {kept} records in place")


if __name__ == "__main__":
    main()
