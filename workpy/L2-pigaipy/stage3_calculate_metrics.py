import os
import json
import argparse
from common.metrics import calculate_and_save_global_metrics
from config import create_config

def calculate_final_metrics(args):
    model_name = args.model
    need_answer = args.need_answer
    config = create_config(model_name, need_answer)
    text_output_dir = config["text_output_dir"]
    # 读取阶段2的筛选结果
    filtered_path = os.path.join(text_output_dir, "0-global_results_filtered.json")
    if not os.path.exists(filtered_path):
        raise FileNotFoundError(f"Missing stage2 output: {filtered_path}")
    
    with open(filtered_path, 'r', encoding='utf-8') as f:
        filtered_data = json.load(f)
    filtered_results = filtered_data["filtered_results"]
    total_group_count = filtered_data["total_group_count"]
    sum_num_iou = filtered_data["sum_num_iou"]
    sum_num_ocr = filtered_data["sum_num_ocr"]

    # 计算并保存最终指标（复用原指标计算函数）
    calculate_and_save_global_metrics(
        filtered_results,  # 仅使用筛选后的label对
        total_group_count,
        text_output_dir,
        sum_num_iou,
        sum_num_ocr
    )
    
    print(f"Stage3 done: metrics saved to {text_output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage3: compute final metrics")
    parser.add_argument('--model', type=str, default='Qwen2.5-VL-7B-Instruct')
    parser.add_argument("--need_answer", type=lambda x: x.lower() == 'true', default=False, required=False)
    args = parser.parse_args()
    calculate_final_metrics(args)