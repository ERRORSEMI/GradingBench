import os
import re
import argparse
from PIL import Image
from tqdm import tqdm

from config import create_config
from api import build_prompt_L2
from common.model_init import init_open_source_model, get_llm_inputs
from common.utils import save_to_jsonfile
from common.label_store import annotation_exists, iter_sample_ids, parse_annotations, resolve_image_path
from common.label_parser import group_boxes_by_question


def batch_process_subject(subject, config, model_name, processor, vllm_inputs_batch, image_list_name, need_answer):
    subject_config = config["subject_config"][subject]
    sample_ids = iter_sample_ids(subject_config)
    print(f"[{subject}] {len(sample_ids)} samples")

    for base_name in tqdm(sample_ids, desc=subject):
        subject_base_name = f"{subject}_{base_name}"
        image_path = resolve_image_path(subject_config, base_name, "L2")
        result_path = os.path.join(config["text_output_dir"], f"{subject_base_name}_raw.json")
        if os.path.exists(result_path):
            continue

        if not annotation_exists(subject_config, base_name):
            continue

        context_boxes, answer_boxes = parse_annotations(subject_config, base_name)
        if not answer_boxes:
            continue

        grouped_boxes = group_boxes_by_question(answer_boxes)
        prompt = build_prompt_L2(image_path, grouped_boxes, context_boxes, model_name, need_answer)
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

    print(f"[{subject}] queued {len(vllm_inputs_batch)} requests")


def process_all_subjects(args):
    model_name = args.model
    need_answer = args.need_answer
    subjects = args.subjects.split(",")
    config = create_config(model_name, need_answer)
    os.makedirs(config["text_output_dir"], exist_ok=True)

    llm, sampling_params, processor = init_open_source_model(
        model_name=args.model,
        model_path=args.model_path,
        tensor_parallel_size=args.tensor_parallel_size,
    )

    for subject in subjects:
        print(f"--- {subject} | model={model_name} ---")
        vllm_inputs_batch = []
        image_list_name = []
        batch_process_subject(
            subject, config, model_name, processor, vllm_inputs_batch, image_list_name, need_answer
        )

        batch_size = 25
        num_batches = (len(vllm_inputs_batch) + batch_size - 1) // batch_size
        for i in tqdm(range(0, len(vllm_inputs_batch), batch_size), total=num_batches, desc=f"{subject} infer"):
            batch = vllm_inputs_batch[i : i + batch_size]
            outputs = llm.generate(batch, sampling_params=sampling_params)

            for idx, output in enumerate(outputs):
                generated_text = output.outputs[0].text
                if "```json" in generated_text and "```" in generated_text:
                    match = re.search(r"```json(.*?)```", generated_text, re.DOTALL)
                    if match:
                        generated_text = match.group(1).strip()

                subject_base_name = image_list_name[i + idx]
                save_to_jsonfile(f"{subject_base_name}_raw.json", generated_text, config["text_output_dir"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="L2 inference (vLLM)")
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
    args = parser.parse_args()
    process_all_subjects(args)
