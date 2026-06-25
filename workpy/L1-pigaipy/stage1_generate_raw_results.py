import os
import json
import argparse

from config import create_config
from common.utils import save_to_textfile, categorize_files_by_subject
from common.label_store import annotation_exists, parse_annotations, resolve_image_path
from common.label_parser import group_boxes_by_question
from common.result_parser import parse_response_to_objects
from common.bbox_utils import bbox_translate
from common.metrics import match_model_results_with_gt


def process_single_raw_file(model_name, subject, config, filename, global_results):
    subject_config = config["subject_config"][subject]
    text_output_dir = config["text_output_dir"]

    base_name = filename.replace("_raw.json", "")
    original_base_name = base_name[len(subject) + 1 :]

    image_path = resolve_image_path(subject_config, original_base_name, "L1")
    raw_file_path = os.path.join(text_output_dir, filename)

    if not annotation_exists(subject_config, original_base_name):
        return 0, 0, 0, 0

    _, answer_boxes = parse_annotations(subject_config, original_base_name)
    if not answer_boxes:
        return 0, 0, 0, 0

    question_groups = group_boxes_by_question(answer_boxes)
    unique_question_ids = list(question_groups.keys())
    total_questions = len(unique_question_ids)

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

    result_content = [
        f"Image: {original_base_name}.jpg",
        f"Question groups: {total_questions}",
        "\nModel outputs:",
    ]

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
        updated_result_content,
        valid_count,
        updated_file_global_results,
    ) = match_model_results_with_gt(
        sorted_model_results,
        sorted_question_groups,
        matched_groups,
        result_content,
        subject,
    )

    global_results.extend(updated_file_global_results)
    updated_result_content.append(f"\nValid groups: {valid_count}/{total_questions}")
    save_to_textfile(
        f"{base_name}_parsed.txt",
        "\n".join(updated_result_content),
        text_output_dir,
    )
    return total_questions, num_iou, num_ocr, max(
        0, len(question_groups) - len(sorted_model_results)
    )


def generate_raw_results(args):
    model_name = args.model
    need_answer = args.need_answer
    config = create_config(model_name, need_answer)
    text_output_dir = config["text_output_dir"]

    raw_files = [f for f in os.listdir(text_output_dir) if f.endswith("_raw.json")]
    print(f"Stage1: parsing {len(raw_files)} raw responses")

    global_results = []
    total_group_count = 0
    subject_files = categorize_files_by_subject(raw_files)

    sum_num_iou = 0
    sum_num_ocr = 0
    sum_num_center = 0

    for subject, files in subject_files.items():
        for filename in files:
            group_count, num_iou, num_ocr, num_center = process_single_raw_file(
                model_name, subject, config, filename, global_results
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

    print(f"Stage1 done: {len(global_results)} matched records -> {raw_results_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage1: parse raw model responses")
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--need_answer",
        type=lambda x: x.lower() == "true",
        default=False,
        required=False,
    )
    args = parser.parse_args()
    generate_raw_results(args)
