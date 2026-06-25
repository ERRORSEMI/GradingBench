"""统一 vLLM 推理入口（L1/L2/L3）。"""
from __future__ import annotations

import argparse
import os
import re

os.environ.setdefault("SETUPTOOLS_USE_DISTUTILS", "local")

from PIL import Image
from tqdm import tqdm

from gradingbench.data.label_parser import group_boxes_by_question
from gradingbench.data.label_store import annotation_exists, parse_annotations, resolve_image_path
from gradingbench.config.level_specs import get_level_spec
from gradingbench.inference.model_init import get_llm_inputs, init_open_source_model
from gradingbench.config.settings import create_config
from gradingbench.prompts.builder import build_prompt
from gradingbench.data.sample_iter import iter_subject_base_names
from gradingbench.utils.io import save_to_jsonfile


def _extract_json_text(generated_text: str, model_name: str, strip_kimi: bool) -> str:
    if strip_kimi and model_name.startswith("Kimi"):
        end_marker = re.escape("<|im_end|>")
        pattern = r"◁/think▷\s*(.*?)\s*" + end_marker
        match = re.search(pattern, generated_text, re.DOTALL)
        if match:
            generated_text = match.group(1).strip()
    if "```json" in generated_text and "```" in generated_text:
        match = re.search(r"```json(.*?)```", generated_text, re.DOTALL)
        if match:
            generated_text = match.group(1).strip()
    return generated_text


def batch_process_subject(
    level,
    subject,
    config,
    model_name,
    processor,
    vllm_inputs_batch,
    image_list_name,
    need_answer,
):
    level_spec = get_level_spec(level)
    subject_config = config["subject_config"][subject]
    base_names = list(iter_subject_base_names(subject_config, level_spec))
    print(f"[{level}/{subject}] {len(base_names)} samples")

    for base_name in tqdm(base_names, desc=subject):
        subject_base_name = f"{subject}_{base_name}"
        image_path = resolve_image_path(subject_config, base_name, level)
        result_path = os.path.join(config["text_output_dir"], f"{subject_base_name}_raw.json")
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
        image = Image.open(image_path).convert("RGB")
        llm_inputs = get_llm_inputs(
            model_name=model_name,
            prompt=prompt,
            image=image,
            image_path=image_path,
            processor=processor,
        )
        vllm_inputs_batch.append(llm_inputs)
        image_list_name.append(subject_base_name)

    print(f"[{level}/{subject}] queued {len(vllm_inputs_batch)} requests")


def process_all_subjects(args):
    level = args.level
    level_spec = get_level_spec(level)
    model_name = args.model
    need_answer = args.need_answer
    subjects = args.subjects.split(",")
    config = create_config(level, model_name, need_answer)
    os.makedirs(config["text_output_dir"], exist_ok=True)

    llm, sampling_params, processor = init_open_source_model(
        model_name=args.model,
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    for subject in subjects:
        print(f"--- [{level}] {subject} | model={model_name} ---")
        vllm_inputs_batch = []
        image_list_name = []
        batch_process_subject(
            level,
            subject,
            config,
            model_name,
            processor,
            vllm_inputs_batch,
            image_list_name,
            need_answer,
        )

        batch_size = 25
        num_batches = (len(vllm_inputs_batch) + batch_size - 1) // batch_size
        for i in tqdm(
            range(0, len(vllm_inputs_batch), batch_size),
            total=num_batches,
            desc=f"{subject} infer",
        ):
            batch = vllm_inputs_batch[i : i + batch_size]
            outputs = llm.generate(batch, sampling_params=sampling_params)

            for idx, output in enumerate(outputs):
                generated_text = _extract_json_text(
                    output.outputs[0].text,
                    model_name,
                    level_spec.strip_kimi_think_vllm,
                )
                subject_base_name = image_list_name[i + idx]
                save_to_jsonfile(
                    f"{subject_base_name}_raw.json",
                    generated_text,
                    config["text_output_dir"],
                )


def main():
    parser = argparse.ArgumentParser(description="Unified vLLM inference (L1/L2/L3)")
    parser.add_argument("--level", type=str, required=True, choices=["L1", "L2", "L3"])
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--tensor_parallel_size", type=int, default=1)
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
