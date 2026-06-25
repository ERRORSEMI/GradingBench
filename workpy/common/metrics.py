from .utils import save_to_textfile
from .result_parser import process_question_group, calculate_OCR
from common.bbox_utils import calculate_iou

def match_model_results_with_gt(
    sorted_model_results, sorted_question_groups, initial_matched_groups,
    initial_result_content, subject
):
    """
    处理模型预测结果与GT标注框的匹配逻辑
    返回：所有计算结果和更新后的变量
    """
    # 1. 在函数内部初始化所有需要使用的变量（解决NameError）
    correct_boxes = []
    incorrect_boxes = []
    ocr_unmatched_boxes = []
    matched_groups = initial_matched_groups.copy()  # 复制初始状态，避免修改原字典
    num_iou = 0
    num_ocr = 0
    result_content = initial_result_content.copy()  # 复制初始结果内容
    valid_count = 0
    file_global_results = []
    # print(len(sorted_model_results), len(sorted_question_groups))
    # 2. 核心匹配逻辑（不变）
    for model_item in sorted_model_results:
        best_match = None
        min_distance = float('inf')
        max_valid_iou = 0  # 记录满足IoU>0.2的最大IoU值

        # --------------------------
        # 第一步：优先寻找IoU>0.2的未匹配GT
        # --------------------------
        for qid, boxes in sorted_question_groups:
            if matched_groups[qid]:  # 跳过已匹配的GT题目组
                continue
            
            # 计算当前模型框与该GT题目组的IoU
            true_bbox_list = [box["bbox"] for box in boxes]
            current_iou = calculate_iou(true_bbox_list, model_item['bbox_2d'])
            
            # 若IoU>0.2，且当前IoU大于已记录的最大有效IoU，更新最佳匹配
            if current_iou > 0.2 and current_iou > max_valid_iou:
                max_valid_iou = current_iou
                best_match = (qid, boxes)  # 优先选择IoU最大的GT

        # --------------------------
        # 第二步：若无IoU>0.2的GT，寻找中心点最近的未匹配GT
        # --------------------------
        if best_match is None:  # 仅在第一步未找到有效匹配时执行
            for qid, boxes in sorted_question_groups:
                if matched_groups[qid]:
                    continue
                
                # 计算GT框中心点
                gt_box = boxes[0]["bbox"]
                gt_center_x = (gt_box[0] + gt_box[2]) / 2
                gt_center_y = (gt_box[1] + gt_box[3]) / 2
                
                # 计算模型框中心点
                model_box = model_item['bbox_2d'][0]
                model_center_x = (model_box[0] + model_box[2]) / 2
                model_center_y = (model_box[1] + model_box[3]) / 2
                
                # 计算欧氏距离
                distance = ((gt_center_x - model_center_x) ** 2 + 
                        (gt_center_y - model_center_y) ** 2) ** 0.5
                
                # 更新最近距离的匹配
                if distance < min_distance:
                    min_distance = distance
                    best_match = (qid, boxes)

        # --------------------------
        # 处理匹配结果
        # --------------------------
        if best_match:
            qid, boxes = best_match
            num_ocr += 1

            # 计算IOU并更新计数
            true_bbox_list = [box["bbox"] for box in boxes]
            iou = calculate_iou(true_bbox_list, model_item['bbox_2d'])
            if iou > 0.2:
                num_iou += 1

            # 计算OCR匹配度
            # print(boxes[0]['hand_written'], model_item['text'])

            text = model_item.get('text', '')
            if text and text[0] == '(' and text[-1] == ')' and boxes[0]['hand_written'] and boxes[0]['hand_written'][0] != '(':
                model_item['text'] = text[1:-1]

            best_iou_ocr = calculate_OCR(boxes[0]['hand_written'], model_item['text'])
            if best_iou_ocr > 0.7:
                num_ocr -= 1
                ocr_unmatched_boxes.append(model_item)
                continue

            # 更新匹配状态和结果
            matched_groups[qid] = True
            group_content, is_valid, result_dict = process_question_group(qid, boxes, model_item, subject)
            result_content.extend(group_content)

            if is_valid:
                
                valid_count += 1
                file_global_results.append(result_dict)
                if result_dict["true_label"] != result_dict["predicted_label"]:
                    incorrect_boxes.append((qid, boxes, model_item))
                else:
                    correct_boxes.append((qid, boxes, model_item))
        else:
            ocr_unmatched_boxes.append(model_item)

    # 3. 返回所有需要在主函数中使用的结果
    return (correct_boxes, incorrect_boxes, ocr_unmatched_boxes,
            matched_groups, num_iou, num_ocr, result_content, valid_count, file_global_results)
            
def calculate_metrics(matched_results, total_labeled_count):
    """
    计算精确率、召回率、正确率，并新增 FN / FP
    注：matched_results为答案已匹配的样本，即r["answer"]与r["ground_truth"]匹配
    :param matched_results: [(true_label, predicted_label), ...] 答案匹配的样本
    :param total_labeled_count: 总标注数量
    """
    # 答案匹配后，再判断label是否相等（正确计数）
    correct_count = sum(1 for t, p in matched_results if t == p)
    # 答案匹配的样本数（用于计算召回率）
    matched_count = len(matched_results)
    labeled_count = total_labeled_count

    # 计算FN和FP（基于答案匹配后的label判断）
    fn, fp = _count_fn_fp(matched_results)

    # 召回率：答案匹配的样本数 / 总标注数（体现模型找到正确答案的能力）
    recall = matched_count / labeled_count if labeled_count > 0 else 0
    # 正确率：答案匹配且label正确的样本数 / 总标注数（体现最终正确性）
    accuracy = correct_count / labeled_count if labeled_count > 0 else 0

    return {
        "recall": recall,          # 答案匹配率（原召回率逻辑调整）
        "accuracy": accuracy,      # 答案匹配且label正确的比例
        "correct_count": correct_count,  # 答案匹配且label正确的数量
        "matched_count": matched_count,  # 答案匹配的总数量
        "labeled_count": labeled_count,
        "fn": fn,   # 答案匹配但true_label=1而predicted_label=0（漏判）
        "fp": fp    # 答案匹配但true_label=0而predicted_label=1（误判）
    }


def calculate_metrics_by_category(global_results, total_labeled_count):
    # ======================== 新增：四个维度的分类统计 ========================
    
    # 初始化各维度的统计字典
    dim1_stats = {}
    dim2_stats = {} 
    dim3_stats = {}
    subject_stats = {}
    
    # 遍历所有结果，按维度分类统计
    for result in global_results:
        # 获取维度值，如果不存在则设为"unknown"
        dim1 = result.get('dim1', 'unknown')
        dim2 = result.get('dim2', 'unknown')
        dim3 = result.get('dim3', 'unknown')
        subject = result.get('subject', 'unknown')
        
        # 获取标签信息
        true_label = result.get('true_label')
        predicted_label = result.get('predicted_label')
        
        # 统计dim1
        if dim1 not in dim1_stats:
            dim1_stats[dim1] = {"total": 0, "correct": 0}
        dim1_stats[dim1]["total"] += 1
        if true_label == predicted_label:
            dim1_stats[dim1]["correct"] += 1
            
        # 统计dim2
        if dim2 not in dim2_stats:
            dim2_stats[dim2] = {"total": 0, "correct": 0}
        dim2_stats[dim2]["total"] += 1
        if true_label == predicted_label:
            dim2_stats[dim2]["correct"] += 1
            
        # 统计dim3
        if dim3 not in dim3_stats:
            dim3_stats[dim3] = {"total": 0, "correct": 0}
        dim3_stats[dim3]["total"] += 1
        if true_label == predicted_label:
            dim3_stats[dim3]["correct"] += 1
            
        # 统计subject
        if subject not in subject_stats:
            subject_stats[subject] = {"total": 0, "correct": 0}
        subject_stats[subject]["total"] += 1
        if true_label == predicted_label:
            subject_stats[subject]["correct"] += 1

    # 2. 计算全局指标（基于答案匹配后的结果）
    # 提取标签对用于全局计算
    label_pairs = [(r.get('true_label'), r.get('predicted_label')) for r in global_results]
    global_metrics = calculate_metrics(label_pairs, total_labeled_count)

    return {
        "global": global_metrics,
        "by_dim1": dim1_stats,
        "by_dim2": dim2_stats,
        "by_dim3": dim3_stats,
        "by_subject": subject_stats
    }


from common.annotation_schema import (
    DIM1_ORDER,
    DIM2_ORDER,
    FIXED_TOTALS,
    STAGE_ORDER,
)


def calculate_and_save_global_metrics(global_results, total_group_count, text_output_dir, num_IOU, num_OCR):
    metrics = calculate_metrics_by_category(global_results, total_group_count)

    summary = f"""
    ============ Global Metrics ============
    Total question groups: {metrics['global']['labeled_count']}
    Matched groups: {metrics['global']['matched_count']}
    Correct groups: {metrics['global']['correct_count']}
    -----------------------------------
    
    Localization match rate (diagnostic): {num_IOU / total_group_count:.2%}
    Text Recognition recall (CER-filtered): {num_OCR / total_group_count:.2%}
    Reasoning recall: {metrics['global']['recall']:.2%}
    End-to-End Accuracy: {metrics['global']['accuracy']:.2%}

    FN (right->wrong): {metrics['global']['fn']}
    FP (wrong->right): {metrics['global']['fp']}
    """

    summary += "\n\n============ By Dimension ============\n"

    subject_order = ["science", "math", "chinese", "english", "liberal_arts"]

    summary += "\n--- dimension1 ---\n"
    for dim_value in DIM1_ORDER:
        if dim_value in metrics["by_dim1"]:
            stats = metrics["by_dim1"][dim_value]
            fixed_total = FIXED_TOTALS[f"dimension1-{dim_value}"]
            percentage = stats["correct"] / fixed_total if fixed_total > 0 else 0
            summary += f"dimension1-{dim_value}: {stats['correct']}/{fixed_total} ({percentage:.2%})\n"

    summary += "\n--- dimension2 ---\n"
    for dim_value in DIM2_ORDER:
        if dim_value in metrics["by_dim2"]:
            stats = metrics["by_dim2"][dim_value]
            fixed_total = FIXED_TOTALS[f"dimension2-{dim_value}"]
            percentage = stats["correct"] / fixed_total if fixed_total > 0 else 0
            summary += f"dimension2-{dim_value}: {stats['correct']}/{fixed_total} ({percentage:.2%})\n"

    summary += "\n--- educational_stage ---\n"
    for dim_value in STAGE_ORDER:
        if dim_value in metrics["by_dim3"]:
            stats = metrics["by_dim3"][dim_value]
            fixed_total = FIXED_TOTALS[f"educational_stage-{dim_value}"]
            percentage = stats["correct"] / fixed_total if fixed_total > 0 else 0
            summary += f"educational_stage-{dim_value}: {stats['correct']}/{fixed_total} ({percentage:.2%})\n"

    summary += "\n--- subject ---\n"
    for subject in subject_order:
        if subject in metrics["by_subject"]:
            stats = metrics["by_subject"][subject]
            fixed_total = FIXED_TOTALS[f"subject-{subject}"]
            percentage = stats["correct"] / fixed_total if fixed_total > 0 else 0
            summary += f"subject-{subject}: {stats['correct']}/{fixed_total} ({percentage:.2%})\n"

    summary += "\n===================================="

    save_to_textfile("1-global_metrics.txt", summary, text_output_dir)
    print(summary)

    return metrics

def _count_fn_fp(pairs):
    """返回 (FN 数量, FP 数量)"""
    fn = 0  # 正确被判成错误
    fp = 0  # 错误被判成正确
    for t, p in pairs:
        if t and not p:
            fn += 1
        elif not t and p:
            fp += 1
    return fn, fp