"""统一 API 推理入口（L1/L2/L3）。"""
from __future__ import annotations

import argparse
import os
import re
import time

from tqdm import tqdm

from gradingbench.data.label_parser import group_boxes_by_question
from gradingbench.data.label_store import annotation_exists, parse_annotations, resolve_image_path
from gradingbench.config.level_specs import get_level_spec
from gradingbench.inference.model_init import call_api
from gradingbench.config.settings import create_config
from gradingbench.prompts.builder import build_prompt
from gradingbench.data.sample_iter import iter_subject_base_names
from gradingbench.utils.io import save_to_jsonfile


def _extract_json_text(generated_text: str) -> str:
    if "```json" in generated_text and "```" in generated_text:
        match = re.search(r"```json(.*?)```", generated_text, re.DOTALL)
        if match:
            return match.group(1).strip()
    return generated_text


def batch_process_subject(level, subject, config, model_name, need_answer):
    level_spec = get_level_spec(level)
    subject_config = config["subject_config"][subject]
    output_dir = config["text_output_dir"]
    base_names = list(iter_subject_base_names(subject_config, level_spec))
    print(f"[{level}/{subject}] {len(base_names)} samples")

    success_count = 0
    for base_name in tqdm(base_names, desc=subject):
        subject_base_name = f"{subject}_{base_name}"
        image_path = resolve_image_path(subject_config, base_name, level)
        result_path = os.path.join(output_dir, f"{subject_base_name}_raw.json")
        if os.path.exists(result_path):
            continue

        if not annotation_exists(subject_config, base_name):
            continue

        context_boxes, answer_boxes = parse_annotations(subject_config, base_name)
        if not answer_boxes:
            continue

        grouped_boxes = group_boxes_by_question(answer_boxes, level_spec.multi_question)
        prompt = build_prompt(
            level,
            image_path,
            grouped_boxes,
            model_name,
            need_answer,
            context_boxes if level_spec.needs_context else None,
        )

        generated_text = None
        max_retries = level_spec.api_max_retries
        for retry_count in range(1, max_retries + 1):
            generated_text = call_api(image_path, prompt, model_name)
            if generated_text is not None:
                break
            print(f"[{subject_base_name}] API retry {retry_count}/{max_retries}")
            time.sleep(level_spec.api_retry_sleep_sec)

        if generated_text is None:
            continue

        save_to_jsonfile(
            f"{subject_base_name}_raw.json",
            _extract_json_text(generated_text),
            output_dir,
        )
        success_count += 1

    print(f"[{level}/{subject}] saved {success_count} responses")


def process_all_subjects(args):
    level = args.level
    model_name = args.model
    need_answer = args.need_answer
    subjects = args.subjects.split(",")
    config = create_config(level, model_name, need_answer)
    os.makedirs(config["text_output_dir"], exist_ok=True)

    for subject in subjects:
        print(f"--- [{level}] {subject} | model={model_name} ---")
        batch_process_subject(level, subject, config, model_name, need_answer)

    print(f"Done. Output: {config['text_output_dir']}")


def main():
    parser = argparse.ArgumentParser(description="Unified API inference (L1/L2/L3)")
    parser.add_argument("--level", type=str, required=True, choices=["L1", "L2", "L3"])
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--subjects",
        type=str,
        required=True,
        help="Comma-separated: chinese,science,liberal_arts,math,english",
    )
    parser.add_argument("--need_answer", type=lambda x: x.lower() == "true", default=False)
    process_all_subjects(parser.parse_args())


if __name__ == "__main__":
    main()
