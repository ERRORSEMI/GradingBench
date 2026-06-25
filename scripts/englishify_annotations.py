#!/usr/bin/env python3
"""Convert data/annotations/L{1,2,3}.json marks to English field names."""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "workpy"))
from common.annotation_schema import normalize_marks

ANN = REPO / "data" / "annotations"


def convert_file(path: Path):
    print(f"Converting {path} ...", flush=True)
    records = json.loads(path.read_text(encoding="utf-8"))
    for rec in records:
        rec["marks"] = normalize_marks(rec.get("marks", []))
    path.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {len(records)} records", flush=True)


def main():
    for name in ("L1.json", "L2.json", "L3.json"):
        p = ANN / name
        if p.is_file():
            convert_file(p)
    print("Done.")


if __name__ == "__main__":
    main()
