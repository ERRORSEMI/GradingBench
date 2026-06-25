"""按 level 配置迭代学科样本。"""
from __future__ import annotations

import os
from typing import Iterator

from gradingbench.data.label_store import iter_sample_ids
from gradingbench.config.level_specs import LevelSpec


def iter_subject_base_names(subject_config: dict, spec: LevelSpec) -> Iterator[str]:
    if spec.iter_mode == "sample_ids":
        yield from iter_sample_ids(subject_config)
        return

    image_dir = subject_config["image_dir"]
    for filename in sorted(os.listdir(image_dir)):
        if filename.lower().endswith(".jpg"):
            yield os.path.splitext(filename)[0]
