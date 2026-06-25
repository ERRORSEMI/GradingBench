import os
import re
import time
import argparse
from tqdm import tqdm

from config import create_config
from api import build_prompt_L3
from common.model_init import call_api
from common.utils import save_to_jsonfile
from common.label_store import annotation_exists, parse_annotations, resolve_image_path
from common.label_parser import group_boxes_by_question


def batch_process_subject(subject, config, model_name, need_answer):
    subject_config = config["subject_config"][subject]
    image_dir = subject_config["image_dir"]
    output_dir = config["text_output_dir"]

    image_files = [f for f in os.listdir(image_dir) if f.lower().endswith(".jpg")]
    print(f"[{subject}] {len(image_files)} pages")

    success_count = 0
    for filename in tqdm(image_files, desc=subject):
        base_name = os.path.splitext(filename)[0]
        subject_base_name = f"{subject}_{base_name}"
        image_path = resolve_image_path(subject_config, base_name, "L3")
        result_path = os.path.join(output_dir, f"{subject_base_name}_raw.json")
        if os.path.exists(result_path):
            continue

        if not annotation_exists(subject_config, base_name):
            continue

        _, answer_boxes = parse_annotations(subject_config, base_name)
        if not answer_boxes:
            continue

        grouped_boxes = group_boxes_by_question(answer_boxes, True)
        prompt = build_prompt_L3(image_path, grouped_boxes, model_name, need_answer)

        generated_text = None
        for retry_count in range(1, 3):
            generated_text = call_api(image_path, prompt, model_name)
            if generated_text is not None:
                break
            print(f"[{subject_base_name}] API retry {retry_count}/2")
            time.sleep(8)

        if generated_text is None:
            continue
        if "```json" in generated_text and "```" in generated_text:
            match = re.search(r"```json(.*?)```", generated_text, re.DOTALL)
            if match:
                generated_text = match.group(1).strip()

        save_to_jsonfile(f"{subject_base_name}_raw.json", generated_text, output_dir)
        success_count += 1

    print(f"[{subject}] saved {success_count} responses")


def process_all_subjects(args):
    model_name = args.model
    need_answer = args.need_answer
    subjects = args.subjects.split(",")
    config = create_config(model_name, need_answer)
    os.makedirs(config["text_output_dir"], exist_ok=True)

    for subject in subjects:
        print(f"--- {subject} | model={model_name} ---")
        batch_process_subject(subject, config, model_name, need_answer)

    print(f"Done. Output: {config['text_output_dir']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L3 inference (API)")
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--subjects",
        type=str,
        required=True,
        help="Comma-separated: chinese,science,liberal_arts,math,english",
    )
    parser.add_argument("--need_answer", type=lambda x: x.lower() == "true", default=False)
    args = parser.parse_args()
    process_all_subjects(args)
