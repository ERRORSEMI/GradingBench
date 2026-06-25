# parser.py
import re
import ast
import json
import Levenshtein


def fix_common_json_issues(json_str):
    """修复JSON字符串中的常见问题"""
    # 修复尾随逗号
    json_str = re.sub(r',\s*([}\]])', r'\1', json_str)
    # 标准化布尔值和null
    json_str = re.sub(r':\s*True\b', ': true', json_str)
    json_str = re.sub(r':\s*False\b', ': false', json_str)
    json_str = re.sub(r':\s*None\b', ': null', json_str)
    # 修复键的引号
    # json_str = re.sub(r"(\w+)\s*:", r'"\1":', json_str)
    # 确保值有引号
    # json_str = re.sub(r':\s*([a-zA-Z_][\w\s]*)\s*([,}])', r': "\1"\2', json_str)
    return json_str

def extract_key_fields(json_str):
    # print(json_str)
    """针对你的JSON片段，修复提取逻辑"""
    result = {}
    
    # 1. 提取bbox_2d：匹配键名，捕获数组内容并转为列表
    # 正则解释：匹配 "bbox_2d" : [任意内容]，允许冒号前后有空格
    bbox_pattern = r'\\"bbox_2d\\"\s*:\s*(\[.*?\])'
    bbox_match = re.search(bbox_pattern, json_str, re.DOTALL)  # re.DOTALL让.匹配换行
    if bbox_match:
        bbox_str = bbox_match.group(1).strip()
        try:
            # 转为Python列表（解决原函数字符串问题）
            result["bbox_2d"] = json.loads(bbox_str)
        except json.JSONDecodeError:
            result["bbox_2d"] = bbox_str  # 极端情况保留字符串
    
    # 2. 提取text：匹配键名，捕获双引号内内容（text无嵌套引号，简单匹配）
    text_pattern = r'\\"text\\"\s*:\s*\\"(.*?)\\"'
    text_match = re.search(text_pattern, json_str, re.DOTALL)
    if text_match:
        result["text"] = text_match.group(1)
    
    # 3. 提取thinking：修复正则，匹配转义引号和内容
    thinking_pattern = r'\\"thinking\\"\s*:\s*\\"(.*?)\\"'
    thinking_match = re.search(thinking_pattern, json_str, re.DOTALL)
    if thinking_match:
        result["thinking"] = thinking_match.group(1)
    
    thinking_pattern = r'\\"answer\\"\s*:\s*\\"(.*?)\\"'
    thinking_match = re.search(thinking_pattern, json_str, re.DOTALL)
    if thinking_match:
        result["answer"] = thinking_match.group(1)

    # 4. 提取is_correct：匹配布尔值（true/false，不区分大小写）
    correct_pattern = r'\\"correctness\\"\s*:\s*(true|false)'
    correct_match = re.search(correct_pattern, json_str, re.IGNORECASE)
    if correct_match:
        result["correctness"] = correct_match.group(1).lower() == "true"
    
    # print("提取结果：", result)
    return result if result else None

def parse_response_to_objects(response_text): # 解析模型批改文件，提取结果列表
    response_text = response_text.strip()
    objects = []
    stack = []
    start_index = -1
    
    for i, char in enumerate(response_text):
        if char == '{':
            if not stack:
                start_index = i
            stack.append(char)
        elif char == '}':
            if stack:
                # print(stack)
                stack.pop()
                if not stack and start_index != -1:
                    obj_str = response_text[start_index:i+1]
                    try:
                        obj = json.loads(obj_str)
                        objects.append(obj)
                    except json.JSONDecodeError:
                        # 尝试修复常见问题
                        obj_str = fix_common_json_issues(obj_str)
                        try:
                            obj = json.loads(obj_str)
                            objects.append(obj)
                        except json.JSONDecodeError:
                            # 如果仍然失败，尝试提取主要字段
                            obj = extract_key_fields(obj_str)
                            if obj:
                                objects.append(obj)
                    start_index = -1
    
    # print(objects)
    # 如果找到了对象，返回对象列表
    if objects:
        return objects

def process_question_group(qid, boxes, item, subject):
    """处理单个题目组并返回结果"""
    # print(item)
    result_content = []
    result_content.append(f"\n作答编号: {qid}")
    result_content.append(f"包含 {len(boxes)} 个作答框")
    
    true_correct = boxes[0]['correctness']
    ground_truth = boxes[0]['ground_truth']
    model_correct = item['correctness']
    result_content.append(f"GT label: {'right' if true_correct else 'wrong'}")
    result_content.append(f"Model: {'right' if model_correct else 'wrong'}")
    result_content.append(f"验证结果: 匹配")
    result_content.append(f"参考答案：{boxes[0]['ground_truth']}")

    # print(item)
    for key in ["text", "thinking", "answer"]:
        if key in item:
            result_content.append(f"{key}: {item[key]}")

    return result_content, True, {
        "true_label": true_correct,
        "predicted_label": model_correct,
        "ground_truth": ground_truth,
        "answer": item["answer"],  # 新增：将模型输出的answer加入返回结果
        "dim1": boxes[0]['dim1'],
        "dim2": boxes[0]['dim2'],
        "dim3": boxes[0]['dim3'],
        "subject": subject
    }

def clean_blank(text):
    if not isinstance(text, str):  # 处理非字符串输入（如None、数字）
        return ""
    # 方法1：清除常见空白字符（空格、\t、\n、\r）
    cleaned = text.replace(" ", "").replace("\t", "").replace("\n", "").replace("\r", "")
    # 方法2：用正则匹配所有空白字符（更全面，包括全角空格等）
    # import re
    # cleaned = re.sub(r"\s+", "", text)  # \s 匹配所有空白字符，+ 表示1个及以上
    return cleaned
    
def calculate_OCR(hand_label, ocr_results) -> bool:
    # 执行清洗
    hand_label = clean_blank(hand_label)
    ocr_results = clean_blank(ocr_results)

    # 如果OCR结果为空，直接返回False
    if len(ocr_results) == 0:
        return False

    dis = Levenshtein.distance(hand_label, ocr_results)

    if hand_label in ocr_results:
        return dis / len(ocr_results)

    return dis / len(hand_label)