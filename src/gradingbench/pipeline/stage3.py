"""Stage3：计算并保存最终指标。"""
from __future__ import annotations

import argparse
import json
import os

from gradingbench.eval.metrics import calculate_and_save_global_metrics
from gradingbench.config.settings import create_config


def calculate_final_metrics(args):
    config = create_config(args.level, args.model, args.need_answer)
    text_output_dir = config["text_output_dir"]
    filtered_path = os.path.join(text_output_dir, "0-global_results_filtered.json")
    if not os.path.exists(filtered_path):
        raise FileNotFoundError(f"Missing stage2 output: {filtered_path}")

    with open(filtered_path, "r", encoding="utf-8") as f:
        filtered_data = json.load(f)

    calculate_and_save_global_metrics(
        filtered_data["filtered_results"],
        filtered_data["total_group_count"],
        text_output_dir,
        filtered_data["sum_num_iou"],
        filtered_data["sum_num_ocr"],
    )

    print(f"[{args.level}] Stage3 done: metrics saved to {text_output_dir}")


def main():
    parser = argparse.ArgumentParser(description="Stage3: compute final metrics")
    parser.add_argument("--level", type=str, required=True, choices=["L1", "L2", "L3"])
    parser.add_argument("--model", type=str, default="Qwen2.5-VL-7B-Instruct")
    parser.add_argument(
        "--need_answer",
        type=lambda x: x.lower() == "true",
        default=False,
        required=False,
    )
    calculate_final_metrics(parser.parse_args())


if __name__ == "__main__":
    main()
