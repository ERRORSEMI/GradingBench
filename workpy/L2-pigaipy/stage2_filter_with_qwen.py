import os
import sys
_WORKPY = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _WORKPY not in sys.path:
    sys.path.insert(0, os.path.dirname(_WORKPY))
from paths import FILTER_MODEL_PATH
import json
import argparse
import re
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from config import create_config

_qwen_llm = None
_qwen_tokenizer = None
_qwen_sampling_params = None


def init_qwen_text_model(model_path, tensor_parallel_size=1):
    global _qwen_llm, _qwen_tokenizer, _qwen_sampling_params
    if _qwen_llm is None:
        _qwen_tokenizer = AutoTokenizer.from_pretrained(model_path)
        _qwen_sampling_params = SamplingParams(
            temperature=0.0,
            top_p=1.0,
            repetition_penalty=1.1,
            max_tokens=100,
            skip_special_tokens=True,
            stop_token_ids=[_qwen_tokenizer.eos_token_id],
        )
        _qwen_llm = LLM(
            model=model_path,
            gpu_memory_utilization=0.5,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=1024,
            max_num_seqs=32,
            trust_remote_code=True,
        )
        print(f"Filter model loaded: {model_path}")


def batch_judge_equivalence(batch_items, need_answer):
    prompts = []
    for item in batch_items:
        prompt = f"""
你是专业批改助手，需判断学生答案与参考答案是否语义等价，输出格式严格为：
"判断结果: 是/否，置信度: 0.0-1.0"
不允许任何额外内容或格式变化。

判断规则：
1. 核心信息（关键概念、结论、逻辑）完全一致，允许表述顺序/同义替换 → 是；
2. 核心信息缺失、偏差或矛盾 → 否。
3. 置信度表示你对判断结果的确定程度（1.0=完全确定，0.5=完全不确定）。

学生答案：{item['answer']}
参考答案：{item['ground_truth']}
""".strip()
        text_prompt = _qwen_tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        prompts.append(text_prompt)

    outputs = _qwen_llm.generate(prompts, sampling_params=_qwen_sampling_params)
    results = []
    for output in outputs:
        pred_text = output.outputs[0].text.strip()
        match = re.match(r"判断结果: (是|否)，置信度: (\d+\.\d+)", pred_text)
        if match:
            result = match.group(1)
            confidence = float(match.group(2))
            is_equivalent = (result == "是") or (result == "否" and confidence <= 0.7)
        elif re.match(r"(是|否)，置信度:(\d+\.\d+)", pred_text):
            match = re.match(r"(是|否)，置信度:(\d+\.\d+)", pred_text)
            result = match.group(1)
            confidence = float(match.group(2))
            is_equivalent = (result == "是") and (confidence > 0.7)
        else:
            is_equivalent = pred_text == "是"
        if need_answer:
            is_equivalent = True
        results.append(is_equivalent)
    return results


def filter_with_qwen_text(args):
    config = create_config(args.model, args.need_answer)
    text_output_dir = config["text_output_dir"]
    raw_results_path = os.path.join(text_output_dir, "0-global_results_raw.json")
    if not os.path.exists(raw_results_path):
        raise FileNotFoundError(f"Missing stage1 output: {raw_results_path}")

    with open(raw_results_path, encoding="utf-8") as f:
        raw_data = json.load(f)
    global_results = raw_data["global_results"]

    init_qwen_text_model(args.model_path, args.tensor_parallel_size)

    batch_size = 32
    filtered_results = []
    total = len(global_results)
    for i in range(0, total, batch_size):
        batch = global_results[i : i + batch_size]
        for item, is_matched in zip(batch, batch_judge_equivalence(batch, args.need_answer)):
            if is_matched:
                filtered_results.append(item)

    filtered_path = os.path.join(text_output_dir, "0-global_results_filtered.json")
    with open(filtered_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "filtered_results": filtered_results,
                "total_group_count": raw_data["total_group_count"],
                "sum_num_iou": raw_data["sum_num_iou"],
                "sum_num_ocr": raw_data["sum_num_ocr"],
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    match_rate = len(filtered_results) / total if total else 0.0
    print(f"Stage2 done: {len(filtered_results)}/{total} kept (match rate {match_rate:.2%})")
    print(f"Saved: {filtered_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage2: OCR filter with Qwen text model")
    parser.add_argument("--model", type=str, default="Qwen3-8B")
    parser.add_argument("--need_answer", type=lambda x: x.lower() == "true", default=False)
    parser.add_argument("--model_path", type=str, default=FILTER_MODEL_PATH)
    parser.add_argument("--tensor_parallel_size", type=int, default=2)
    args = parser.parse_args()
    filter_with_qwen_text(args)
