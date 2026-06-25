# utils.py
import os
import base64
import re
import ast
import json
from PIL import Image

def encode_image(image_path):
    """将图片编码为base64"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def save_to_textfile(filename, content, output_dir):
    """将内容保存到文本文件"""
    path = os.path.join(output_dir, filename)
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content)

def save_to_jsonfile(filename, content, output_dir):
    # 1. 尝试把 content 当成 JSON 字符串反序列化
    try:
        data = json.loads(content) if isinstance(content, str) else content
    except (TypeError, ValueError):
        # 不是合法 JSON 字符串，就原样用 content
        data = content
    # 2. 如果反序列化后得到 dict，则检查是否需要提取
    if isinstance(data, dict):
        for k in ["result", "results", "answer", "answers"]:
            if k in data:
                data = data[k]
                break
        if isinstance(data, dict):
            data = [data]

    # 3. 写入文件
    path = os.path.join(output_dir, filename)
    os.makedirs(output_dir, exist_ok=True)          # 确保目录存在
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def calculate_correct_ratio(question_groups):

    total = 0
    correct_count = 0
    
    for group in question_groups.values():
        for item in group:
            total += 1
            if item.get('correctness', False):
                correct_count += 1
    
    if total == 0:
        return 0.0 
    return correct_count / total

def categorize_files_by_subject(raw_files):
    """按科目对文件进行分类"""
    subject_files = {
        "math": [],
        "chinese": [],
        "science": [],
        "liberal_arts": [],
        "english": []
    }
    
    for filename in raw_files:
        if filename.startswith("math_"):
            subject_files["math"].append(filename)
        elif filename.startswith("chinese_"):
            subject_files["chinese"].append(filename)
        elif filename.startswith("science_"):
            subject_files["science"].append(filename)
        elif filename.startswith("english_"):
            subject_files["english"].append(filename)
        elif filename.startswith("liberal_arts_"):
            subject_files["liberal_arts"].append(filename)
    
    return subject_files
