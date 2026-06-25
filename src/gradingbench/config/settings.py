"""Evaluation run configuration (output dirs, subject layout)."""
from __future__ import annotations

from gradingbench.paths import get_base_dir, results_json_dir, subject_dirs

BASE_DIR = get_base_dir()


def create_config(level: str, model_name: str, need_answer: bool = False) -> dict:
    """Return output directory and per-subject config for a level."""
    return {
        "level": level,
        "text_output_dir": results_json_dir(level, need_answer, model_name),
        "subject_config": subject_dirs(level),
    }
