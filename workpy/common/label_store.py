"""Load GT from data/annotations/L{1,2,3}.json (images under data/images/)."""
from __future__ import annotations

import json
import os
from functools import lru_cache

from common.label_parser import parse_label_data


def _record_key(record: dict) -> tuple[str, str] | None:
    subject = record.get("subject")
    sample_id = record.get("id") or os.path.splitext(record.get("image", ""))[0]
    if subject and sample_id:
        return subject, sample_id
    return None


@lru_cache(maxsize=8)
def _load_level_index(annotations_file: str) -> dict[tuple[str, str], dict]:
    if not os.path.isfile(annotations_file):
        return {}
    with open(annotations_file, encoding="utf-8") as f:
        data = json.load(f)
    records = data if isinstance(data, list) else data.get("samples", [])
    index: dict[tuple[str, str], dict] = {}
    for record in records:
        key = _record_key(record)
        if key:
            index[key] = record
    return index


def _lookup(subject_config: dict, sample_id: str) -> dict | None:
    ann_file = subject_config.get("annotations_file")
    if not ann_file:
        return None
    subject = subject_config.get("subject_name")
    return _load_level_index(ann_file).get((subject, sample_id))


def annotation_exists(subject_config: dict, sample_id: str) -> bool:
    return _lookup(subject_config, sample_id) is not None


def load_label_record(subject_config: dict, sample_id: str) -> dict | None:
    return _lookup(subject_config, sample_id)


def iter_sample_ids(subject_config: dict) -> list[str]:
    ann_file = subject_config.get("annotations_file")
    subject = subject_config.get("subject_name")
    if not ann_file or not subject:
        return []
    index = _load_level_index(ann_file)
    return sorted(sid for (subj, sid) in index if subj == subject)


def resolve_image_path(subject_config: dict, sample_id: str, level: str | None = None) -> str:
    record = load_label_record(subject_config, sample_id)
    lvl = level or subject_config.get("level", "L1")
    if record and record.get("image"):
        rel = record["image"].replace("\\", "/")
        level_dir = os.path.dirname(subject_config["image_dir"])
        return os.path.join(level_dir, rel)

    image_dir = subject_config["image_dir"]
    if lvl == "L2":
        last_us = sample_id.rfind("_")
        page_id = sample_id[:last_us] if last_us != -1 else sample_id
        return os.path.join(image_dir, f"{page_id}.jpg")
    return os.path.join(image_dir, f"{sample_id}.jpg")


def parse_annotations(subject_config: dict, sample_id: str):
    record = load_label_record(subject_config, sample_id)
    if record is None:
        raise FileNotFoundError(f"annotation not found: {sample_config.get('subject_name')}/{sample_id}")
    return parse_label_data(record)


def infer_image_size(subject_config: dict, sample_id: str) -> tuple[int, int] | tuple[None, None]:
    record = load_label_record(subject_config, sample_id)
    if not record:
        return None, None
    width, height = record.get("width"), record.get("height")
    if width and height:
        return int(width), int(height)
    return None, None
