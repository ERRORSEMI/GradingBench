import os
import sys

_WORKPY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKPY not in sys.path:
    sys.path.insert(0, _WORKPY)

from paths import SUBJECTS, get_base_dir, results_json_dir, subject_dirs

BASE_DIR = get_base_dir()


def create_config(model_name, need_answer=False):
    """L1 single-question: answer-free / answer-based via need_answer."""
    level = "L1"
    return {
        "text_output_dir": results_json_dir(level, need_answer, model_name),
        "subject_config": subject_dirs(level),
    }
