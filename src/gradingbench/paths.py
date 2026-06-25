"""Repository paths and environment-backed configuration."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

DATA_ROOT = Path(os.environ.get("PIGAI_DATA_ROOT", REPO_ROOT / "data"))
IMAGES_ROOT = DATA_ROOT / "images"
ANNOTATIONS_ROOT = DATA_ROOT / "annotations"
RESULTS_ROOT = Path(os.environ.get("PIGAI_RESULTS_ROOT", REPO_ROOT / "results"))

FILTER_MODEL_PATH = os.environ.get(
    "PIGAI_FILTER_MODEL_PATH",
    os.environ.get("QWEN3_8B_PATH", "<path/to/Qwen3-8B>"),
)

SUBJECTS = {
    "math": "Mathematics",
    "chinese": "Chinese",
    "english": "English",
    "science": "Science",
    "liberal_arts": "Humanities",
}

LEVELS = ("L1", "L2", "L3")


def answer_mode(need_answer: bool) -> str:
    return "answer-based" if need_answer else "answer-free"


def get_base_dir() -> str:
    return str(REPO_ROOT)


def level_annotations_file(level: str) -> str:
    if level not in LEVELS:
        raise ValueError(f"unknown level: {level}")
    return str(ANNOTATIONS_ROOT / f"{level}.json")


def subject_image_dir(level: str, subject_name: str) -> str:
    return str(IMAGES_ROOT / level / subject_name)


def subject_dirs(level: str) -> dict:
    if level not in LEVELS:
        raise ValueError(f"unknown level: {level}")
    ann_file = level_annotations_file(level)
    out = {}
    for key, name in SUBJECTS.items():
        out[key] = {
            "level": level,
            "subject_key": key,
            "subject_name": name,
            "image_dir": subject_image_dir(level, name),
            "annotations_file": ann_file,
        }
    return out


def results_json_dir(level: str, need_answer: bool, model_name: str) -> str:
    return str(
        RESULTS_ROOT
        / "predictions"
        / level
        / answer_mode(need_answer)
        / model_name
    )


def results_predictions_root(level: str, need_answer: bool) -> str:
    return str(RESULTS_ROOT / "predictions" / level / answer_mode(need_answer))


def loc_metrics_path(level: str, need_answer: bool = False, ext: str = "csv") -> str:
    tag = f"{level}_{answer_mode(need_answer)}_localization"
    return str(RESULTS_ROOT / "metrics" / f"{tag}.{ext}")


def get_api_credentials(model_name: str | None = None) -> tuple[str, str]:
    if model_name == "doubao-1.5-vision-pro":
        app_id = os.environ.get("PIGAI_API_APP_ID_DOUBAO", os.environ.get("PIGAI_API_APP_ID", ""))
        app_key = os.environ.get("PIGAI_API_APP_KEY_DOUBAO", os.environ.get("PIGAI_API_APP_KEY", ""))
    elif model_name == "qwen3-vl-plus":
        app_id = os.environ.get("PIGAI_API_APP_ID_QWEN3", os.environ.get("PIGAI_API_APP_ID", ""))
        app_key = os.environ.get("PIGAI_API_APP_KEY_QWEN3", os.environ.get("PIGAI_API_APP_KEY", ""))
    else:
        app_id = os.environ.get("PIGAI_API_APP_ID", "")
        app_key = os.environ.get("PIGAI_API_APP_KEY", "")
    return app_id, app_key


def get_api_endpoint() -> str:
    return os.environ.get(
        "PIGAI_API_ENDPOINT",
        "https://your-api-endpoint/v1/chat/completions",
    )
