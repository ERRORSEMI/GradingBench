#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Export benchmark data:

  data/
    images/L{1,2,3}/{Subject}/*.jpg
    annotations/L1.json
    annotations/L2.json
    annotations/L3.json
    manifest/summary.json
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO / "workpy"))
from common.annotation_schema import normalize_marks
DATA_ROOT = REPO / "data"
IMAGES_ROOT = DATA_ROOT / "images"
ANNOTATIONS_ROOT = DATA_ROOT / "annotations"
MANIFEST_ROOT = DATA_ROOT / "manifest"

SUBJECTS = {
    "math": "Mathematics",
    "chinese": "Chinese",
    "english": "English",
    "science": "Science",
    "liberal_arts": "Humanities",
}

LEVELS = ("L1", "L2", "L3")

LEGACY_SUBJECT_CN = {
    "math": "数学",
    "chinese": "语文",
    "english": "英语",
    "science": "理综",
    "liberal_arts": "文综",
}

LEGACY_LEVEL_DIRS = {
    "L1": {"image": "L1-单题裁剪图", "label": "L1-单题作答框", "ext": ".jpg"},
    "L2": {"image": "L2-整页原图", "label": "L2-单题作答框", "ext": ".jpg"},
    "L3": {"image": "L3-整页原图", "label": "L3-整页标注", "ext": ".jpg"},
}


def _list_json(dir_path: Path) -> list[Path]:
    if not dir_path.is_dir():
        return []
    out = []
    try:
        with os.scandir(dir_path) as it:
            for entry in it:
                if entry.is_file() and entry.name.endswith(".json"):
                    out.append(dir_path / entry.name)
    except OSError:
        return []
    return sorted(out)


def _image_filename(level_key: str, sample_id: str, ext: str) -> str:
    if level_key == "L2":
        last_us = sample_id.rfind("_")
        page_id = sample_id[:last_us] if last_us != -1 else sample_id
        return page_id + ext
    return sample_id + ext


def _ensure_file(src: Path, dst: Path, mode: str):
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.is_file() and not dst.is_symlink():
        try:
            if dst.stat().st_size == src.stat().st_size:
                return False
        except OSError:
            pass
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if mode == "link":
        os.symlink(src.resolve(), dst)
    else:
        shutil.copy2(src, dst)
    return True


def copy_images_for_level(level_key: str, src_root: Path, mode: str) -> dict:
    """Copy/link images using existing annotations/Lx.json (skip if already present)."""
    ann_path = ANNOTATIONS_ROOT / f"{level_key}.json"
    if not ann_path.is_file():
        raise FileNotFoundError(f"Missing {ann_path}; run annotations-only first")

    legacy = LEGACY_LEVEL_DIRS[level_key]
    en_to_cn = {SUBJECTS[k]: LEGACY_SUBJECT_CN[k] for k in SUBJECTS}
    records = json.loads(ann_path.read_text(encoding="utf-8"))

    copied = skipped = missing = 0
    seen: set[tuple[str, str]] = set()

    for rec in records:
        subj_en = rec["subject"]
        subj_cn = en_to_cn.get(subj_en)
        if not subj_cn:
            continue
        image_name = os.path.basename(rec.get("image", ""))
        if not image_name:
            continue
        key = (subj_en, image_name)
        if key in seen:
            continue
        seen.add(key)

        src_img = src_root / subj_cn / legacy["image"] / image_name
        dest_img = IMAGES_ROOT / level_key / subj_en / image_name

        if not src_img.is_file():
            missing += 1
            continue
        if _ensure_file(src_img, dest_img, mode):
            copied += 1
        else:
            skipped += 1

    stats = {"copied": copied, "skipped": skipped, "missing": missing, "unique_images": len(seen)}
    print(
        f"  {level_key}: {stats['unique_images']} images, "
        f"copied={copied}, skipped={skipped}, missing={missing}",
        flush=True,
    )
    return stats


def export_level(level_key: str, src_root: Path, mode: str) -> list[dict]:
    legacy = LEGACY_LEVEL_DIRS[level_key]
    level_records: list[dict] = []
    linked_pages: set[tuple[str, str]] = set()

    for subj_key, subj_en in SUBJECTS.items():
        subj_cn = LEGACY_SUBJECT_CN[subj_key]
        src_subj = src_root / subj_cn
        lbl_dir = src_subj / legacy["label"]
        img_dir = src_subj / legacy["image"]
        if not lbl_dir.is_dir():
            continue

        dest_img_dir = IMAGES_ROOT / level_key / subj_en
        dest_img_dir.mkdir(parents=True, exist_ok=True)

        for lbl_path in _list_json(lbl_dir):
            with open(lbl_path, encoding="utf-8") as f:
                raw = json.load(f)

            sample_id = lbl_path.stem
            image_name = _image_filename(level_key, sample_id, legacy["ext"])
            src_img = img_dir / image_name
            has_image = src_img.is_file()

            if has_image and mode in ("link", "copy-all"):
                page_key = (subj_en, image_name)
                if page_key not in linked_pages:
                    if _ensure_file(src_img, dest_img_dir / image_name, mode):
                        linked_pages.add(page_key)
                    elif (dest_img_dir / image_name).is_file():
                        linked_pages.add(page_key)

            level_records.append(
                {
                    "id": sample_id,
                    "subject": subj_en,
                    "level": level_key,
                    "image": f"{subj_en}/{image_name}",
                    "width": raw.get("width"),
                    "height": raw.get("height"),
                    "marks": normalize_marks(raw.get("marks", [])),
                }
            )

        n_subj = sum(1 for r in level_records if r["subject"] == subj_en)
        print(f"  {level_key}/{subj_en}: {n_subj} annotations", flush=True)

    ANNOTATIONS_ROOT.mkdir(parents=True, exist_ok=True)
    out_path = ANNOTATIONS_ROOT / f"{level_key}.json"
    out_path.write_text(
        json.dumps(level_records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"  -> {out_path} ({len(level_records)} records)", flush=True)
    return level_records


def write_summary(stats: dict):
    MANIFEST_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "dataset": "GradingBench: Evaluating End-to-End Compositional Reasoning of MLLMs for Automated Exam Grading",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "layout": {
            "images": "data/images/{L1|L2|L3}/{Subject}/*.jpg",
            "annotations": "data/annotations/L1.json, L2.json, L3.json",
        },
        "subjects": SUBJECTS,
        "levels": list(LEVELS),
        "statistics": stats,
    }
    (MANIFEST_ROOT / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="Legacy test root")
    parser.add_argument(
        "--mode",
        choices=["annotations-only", "link", "copy-all"],
        default="link",
        help="annotations-only | link images | copy-all images",
    )
    parser.add_argument("--levels", nargs="*", choices=list(LEVELS), default=list(LEVELS))
    parser.add_argument(
        "--images-only",
        action="store_true",
        help="Only copy/link images (requires existing annotations/Lx.json)",
    )
    args = parser.parse_args()

    src_root = Path(args.source)
    if not src_root.is_dir():
        raise SystemExit(f"Source not found: {src_root}")

    if args.images_only:
        if args.mode == "annotations-only":
            raise SystemExit("--images-only requires --mode link or copy-all")
        img_mode = "link" if args.mode == "link" else "copy-all"
        img_stats = {}
        for level_key in args.levels:
            print(f"Copy images {level_key} ...", flush=True)
            img_stats[level_key] = copy_images_for_level(level_key, src_root, img_mode)
        print("\nDone.", flush=True)
        return

    img_mode = "link" if args.mode == "link" else ("copy-all" if args.mode == "copy-all" else None)
    stats = {"by_level": {}, "by_subject": defaultdict(lambda: defaultdict(int))}

    for level_key in args.levels:
        print(f"Export {level_key} ...", flush=True)
        records = export_level(level_key, src_root, img_mode or "annotations-only")
        stats["by_level"][level_key] = {"num_samples": len(records)}
        for rec in records:
            stats["by_subject"][rec["subject"]][level_key] += 1

    stats["by_subject"] = {k: dict(v) for k, v in stats["by_subject"].items()}
    write_summary(stats)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
