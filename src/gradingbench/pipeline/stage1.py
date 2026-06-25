"""Stage1：解析 raw 响应并与 GT 匹配。"""
from __future__ import annotations

import argparse
import json
import os

from gradingbench.coords.bbox import bbox_translate
from gradingbench.data.label_parser import group_boxes_by_question
from gradingbench.data.label_store import annotation_exists, parse_annotations, resolve_image_path
from gradingbench.config.level_specs import get_level_spec
from gradingbench.eval.metrics import match_model_results_with_gt
from gradingbench.config.settings import create_config
from gradingbench.eval.result_parser import parse_response_to_objects
from gradingbench.utils.io import categorize_files_by_subject


def process_single_raw_file(level, model_name, subject, config, filename, global_results):
    level_spec = get_level_spec(level)
    subject_config = config["subject_config"][subject]
    text_output_dir = config["text_output_dir"]

    original_base_name = filename.replace("_raw.json", "")[len(subject) + 1 :]
    image_path = resolve_image_path(subject_config, original_base_name, level)
    raw_file_path = os.path.join(text_output_dir, filename)

    if not annotation_exists(subject_config, original_base_name):
        return 0, 0, 0, 0

    _context, answer_boxes = parse_annotations(subject_config, original_base_name)
    if not answer_boxes:
        return 0, 0, 0, 0

    question_groups = group_boxes_by_question(answer_boxes, level_spec.multi_question)
    total_questions = len(question_groups)

    try:
        with open(raw_file_path, "r", encoding="utf-8") as f:
            raw_content = json.load(f)
        if isinstance(raw_content, dict):
            raw_content = [raw_content]
        if isinstance(raw_content, str):
            raise ValueError("parsed response is a string")
    except Exception:
        with open(raw_file_path, "r", encoding="utf-8") as f:
            raw_content = f.read()
        raw_content = parse_response_to_objects(raw_content)
    if raw_content is None:
        return total_questions, 0, 0, 0

    model_results = bbox_translate(image_path, raw_content, model_name)

    sorted_question_groups = sorted(
        question_groups.items(),
        key=lambda item: item[1][0]["bbox"][1],
    )
    sorted_model_results = sorted(
        model_results,
        key=lambda item: item["bbox_2d"][0][1] if item.get("bbox_2d") else float("inf"),
    )

    matched_groups = {qid: False for qid in question_groups}
    (
        _correct_boxes,
        _incorrect_boxes,
        _ocr_unmatched_boxes,
        _updated_matched_groups,
        num_iou,
        num_ocr,
        _result_content,
        _valid_count,
        updated_file_global_results,
    ) = match_model_results_with_gt(
        sorted_model_results,
        sorted_question_groups,
        matched_groups,
        [],
        subject,
    )

    global_results.extend(updated_file_global_results)
    return total_questions, num_iou, num_ocr, max(
        0, len(question_groups) - len(sorted_model_results)
    )


def generate_raw_results(args):
    level = args.level
    model_name = args.model
    need_answer = args.need_answer
    config = create_config(level, model_name, need_answer)
    text_output_dir = config["text_output_dir"]

    raw_files = [f for f in os.listdir(text_output_dir) if f.endswith("_raw.json")]
    print(f"[{level}] Stage1: parsing {len(raw_files)} raw responses")

    global_results = []
    total_group_count = 0
    subject_files = categorize_files_by_subject(raw_files)

    sum_num_iou = 0
    sum_num_ocr = 0
    sum_num_center = 0

    for subject, files in subject_files.items():
        for filename in files:
            group_count, num_iou, num_ocr, num_center = process_single_raw_file(
                level, model_name, subject, config, filename, global_results
            )
            total_group_count += group_count
            sum_num_iou += num_iou
            sum_num_ocr += num_ocr
            sum_num_center += num_center

    raw_results_path = os.path.join(text_output_dir, "0-global_results_raw.json")
    with open(raw_results_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "global_results": global_results,
                "total_group_count": total_group_count,
                "sum_num_iou": sum_num_iou,
                "sum_num_ocr": sum_num_ocr,
                "sum_num_center": total_group_count - sum_num_center,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    print(f"[{level}] Stage1 done: {len(global_results)} matched records -> {raw_results_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage1: parse raw model responses")
    parser.add_argument("--level", type=str, required=True, choices=["L1", "L2", "L3"])
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--need_answer",
        type=lambda x: x.lower() == "true",
        default=False,
        required=False,
    )
    generate_raw_results(parser.parse_args())


if __name__ == "__main__":
    main()
