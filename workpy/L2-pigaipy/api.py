import requests
import time
import math

from common.utils import encode_image
from PIL import Image


def build_common_prompt_parts(desc, fmt, need_answer, answer_str, question_str):

    if need_answer and answer_str != "":
        answer_str = f"""参考答案（核心依据）
{answer_str}
> 说明：非必要无需自主推导题目答案，直接以本参考答案为判断基准；
"""
        thinking_rule = "- thinking字段需：简要说明 “学生答案与参考答案的匹配逻辑”；"
    else:
        answer_str = ""
        thinking_rule = "thinking字段需按 “题目分析→步骤推导→结论” 结构书写，需包含关键依据（数学 / 理综说明公式定理，语文 / 英语说明语法 / 语境，文综关联知识点）。得到的题目答案输出到answer字段"

    prompt = f'''
你是一名专业的试卷批改助手，这是一张完整的学生试卷，请你仅批改{desc}{fmt}{question_str}处的题目，忽略其他题目的内容，严格按以下要求完成任务：

一、作答区域定位（输出到bbox_2d字段）
- 检测所有学生手写作答区域：填空题横线/括号、简答题文字区、计算题步骤区、选择题勾选区；
- 用边界框标记每个答案区域（一道题的多个子问题分别标记），坐标格式：{desc}

二、作答文本识别（输出到text字段）
- 完整提取区域内学生手写作答内容，特殊题型按以下格式整理：
  - 连线题：提取为“左侧内容—右侧内容”或“上方内容—下方内容”（如“苹果—水果”）；
  - 勾画题：提取勾画的完整文本（如勾画“大”则填“大”）；
  - 改错题：提取为“将‘xx’改成‘xx’”或“将‘xx’删掉”。

三、解题推导与答案（输出到thinking、answer字段）
{answer_str}
{thinking_rule}

四、正确性判断（输出到correctness字段）
- 仅输出true或false；
- 批改时需按“语义等价”原则判断，而无需逐字匹配，对于半开放题，学生答案核心信息、逻辑、结论与正确答案一致即判定为正确；
- 若学生出现关键性错别字、或拼音代替汉字，也算错误。

输出json列表格式:
```json
[
    {{
        "bbox_2d": 作答区域边界框{fmt},
        "text": 学生手写作答文本,    
        "thinking": 按前文规则生成的内容,
        "answer": 题目正确答案, 
        "correctness": 学生作答正确性(true|false)
    }}
]
```
'''
    return prompt


def build_prompt_L2(image_path, grouped_boxes, context_boxes, model_name, need_answer):
    """根据模型名返回对应坐标格式的prompt，区分坐标描述和格式"""
    coord_map = {
        "Qwen2.5-VL":         ("绝对像素坐标", "[xmin,ymin,xmax,ymax]"),
        "Qwen3":         ("0-1000 相对坐标", "[xmin,ymin,xmax,ymax]"),
        "claude-opus-4.6":        ("0-1 相对坐标", "[xmin,ymin,xmax,ymax]"),
        "claude":        ("绝对像素坐标", "[xmin,ymin,xmax,ymax]"),
        "gemini":        ("0-1000 相对坐标", "[ymin,xmin,ymax,xmax]"),
        "gpt":            ("0-1 归一化中心坐标", "[x_center,y_center,width,height]"),
        "doubao-seed-2.0":   ("0-1000 相对坐标", "[xmin,ymin,xmax,ymax]"),
        "doubao-seed":   ("0-1 相对坐标", "[xmin,ymin,xmax,ymax]"),
        "deepseek-vl2":       ("0-1 相对坐标", "[xmin,ymin,xmax,ymax]"),
        "doubao-1.5-vision-pro": ("0-1000 相对坐标", "[xmin,ymin,xmax,ymax]"),
        "InternVL3":          ("0-1000 相对坐标", "[xmin,ymin,xmax,ymax]"),
        "kimi":            ("0-1 相对坐标", "[xmin,ymin,xmax,ymax]"),
        "gemma-3":            ("0-1000 相对坐标", "[xmin,ymin,xmax,ymax]"),
        "glm":            ("0-1000 相对坐标", "[xmin,ymin,xmax,ymax]"),
    }

    # 初始化变量，避免未定义的情况
    original_width = None
    original_height = None
    resized_height = None
    resized_width = None
    
    # 将模型名转为小写后再判断前缀
    model_name_lower = model_name.lower()
    if model_name_lower.startswith("qwen2.5"):
        original_image = Image.open(image_path)
        original_width, original_height = original_image.size   # 先拆成 w, h
        resized_height, resized_width = smart_resize(original_height, original_width)

    answer_str = ""
    if need_answer:
        prompt_parts = []
        for qid, boxes in grouped_boxes.items():
            # 取当前组任意一个 box 的 ground_truth
            gt = next((b.get("ground_truth") for b in boxes if b.get("ground_truth")), None)
            if gt:
                part = f"{gt}; "
            else:
                part = ""  # 避免 gt 为 None 时出现未定义的情况
            
            prompt_parts.append(part)

        answer_str = "".join(prompt_parts)
    
    # 找到第一个匹配的前缀（转换为小写后匹配）
    for prefix, (desc, fmt) in coord_map.items():
        prefix_lower = prefix.lower()  # 将前缀转为小写
        if model_name_lower.startswith(prefix_lower):
            if desc == "0-1000 相对坐标" and fmt =="[xmin,ymin,xmax,ymax]":
                question_str = build_prompt_relative_1000(image_path, context_boxes)
            elif desc == "0-1 相对坐标":
                question_str = build_prompt_relative_1(image_path, context_boxes)
            elif desc == "0-1 归一化中心坐标":
                question_str =  build_prompt_yolo(image_path, context_boxes)
            elif desc == "0-1000 相对坐标":
                question_str =  build_prompt_gemini(image_path, context_boxes)
            elif model_name_lower.startswith("qwen2.5"):  # 小写匹配
                question_str = build_prompt_absolute(image_path, context_boxes, original_width, original_height, resized_height, resized_width)
            elif desc == "绝对像素坐标":
                question_str = build_prompt_absolute(image_path, context_boxes)
            else:
                question_str =  build_prompt_gemini(image_path, context_boxes)
            return build_common_prompt_parts(desc, fmt, need_answer, answer_str, question_str)

    question_str = build_prompt_relative_1000(image_path, context_boxes)
    return build_common_prompt_parts("0-1000 相对坐标", "[xmin,ymin,xmax,ymax]", need_answer, answer_str, question_str)




def smart_resize(
    height: int,
    width: int,
    factor: int = 28,
    min_pixels: int = 224 * 224,
    max_pixels: int = 2200 * 28 * 28,
) -> tuple[int, int]:
    # 2. 先把原分辨率缩放到最接近 factor 的整数倍
    h_bar = round(height / factor) * factor
    w_bar = round(width / factor) * factor

    # 3. 如果此时像素数 > max_pixels，按比例缩小
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((height * width) / max_pixels)   # 需要缩放的比例
        h_bar = math.floor(height / beta / factor) * factor
        w_bar = math.floor(width / beta / factor) * factor

    # 4. 如果此时像素数 < min_pixels，按比例放大
    if h_bar * w_bar < min_pixels:
        beta = math.sqrt((height * width) / min_pixels)   # 需要缩放的比例
        h_bar = max(math.ceil(height / beta / factor) * factor, factor)
        w_bar = max(math.ceil(width / beta / factor) * factor, factor)

    return h_bar, w_bar




def build_prompt_absolute(image_path, context_boxes, original_width=None, original_height=None, resized_height=None, resized_width=None):
    """构建绝对像素坐标提示（prompt）"""
    scale_x = 1
    scale_y = 1
    if original_width:
        scale_x = resized_width / original_width
        scale_y = resized_height / original_height

    first_value = next(iter(context_boxes.values()))
    bboxes = first_value['bbox']
    # 计算每个边界框的绝对坐标并转换为字符串
    abs_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        abs_bbox = [
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            int(round(x2 * scale_x)),
            int(round(y2 * scale_y))
        ]
        abs_bboxes.append(abs_bbox)

    # 将所有边界框的字符串组合成一个字符串
    abs_bboxes_str = ", ".join([f"[{xmin}, {ymin}, {xmax}, {ymax}]" for xmin, ymin, xmax, ymax in abs_bboxes])

    # 构建最终的字符串
    question_str = f"[{abs_bboxes_str}]"

    return question_str

def build_prompt_relative_1000(image_path, context_boxes):
    """构建0-1000相对坐标提示（prompt）"""
    img = Image.open(image_path)
    width, height = img.size
    scale = 1000

    first_value = next(iter(context_boxes.values()))
    bboxes = first_value['bbox']

    # 计算每个边界框的缩放坐标并转换为字符串
    scaled_bboxes = []
    for bbox in bboxes:
        context_xmin = int(bbox[0] / width * scale)
        context_ymin = int(bbox[1] / height * scale)
        context_xmax = int(bbox[2] / width * scale)
        context_ymax = int(bbox[3] / height * scale)
        scaled_bboxes.append(f"[{context_xmin}, {context_ymin}, {context_xmax}, {context_ymax}]")

    # 将所有边界框的字符串组合成一个字符串
    context_bboxes_str = ", ".join(scaled_bboxes)

    question_str =  f"[{context_bboxes_str}]"
    
    return question_str

def build_prompt_relative_1(image_path, context_boxes):
    """构建0-1相对坐标提示（prompt）"""
    img = Image.open(image_path)
    width, height = img.size

    first_value = next(iter(context_boxes.values()))
    bboxes = first_value['bbox']

    # 计算每个边界框的归一化坐标并转换为字符串
    normalized_bboxes = []
    for bbox in bboxes:
        context_xmin = round(bbox[0] / width, 2)
        context_ymin = round(bbox[1] / height, 2)
        context_xmax = round(bbox[2] / width, 2)
        context_ymax = round(bbox[3] / height, 2)
        normalized_bboxes.append(f"[{context_xmin:.2f}, {context_ymin:.2f}, {context_xmax:.2f}, {context_ymax:.2f}]")

    # 将所有边界框的字符串组合成一个字符串
    context_bboxes_str = ", ".join(normalized_bboxes)

    # 构建最终的字符串
    question_str = f"[{context_bboxes_str}]"

    return question_str

def build_prompt_gemini(image_path, context_boxes):
    """构建[ymin, xmin, ymax, xmax]归一化坐标提示（prompt）"""
    img = Image.open(image_path)
    width, height = img.size
    scale = 1000
    
    first_value = next(iter(context_boxes.values()))
    # 获取边界框列表
    bboxes = first_value['bbox']

    # 计算每个边界框的归一化坐标并转换为字符串
    normalized_bboxes = []
    for bbox in bboxes:
        context_ymin = int(bbox[1] / height * scale)
        context_xmin = int(bbox[0] / width * scale)
        context_ymax = int(bbox[3] / height * scale)
        context_xmax = int(bbox[2] / width * scale)
        normalized_bboxes.append(f"[{context_ymin}, {context_xmin}, {context_ymax}, {context_xmax}]")

    # 将所有边界框的字符串组合成一个字符串
    context_bboxes_str = ", ".join(normalized_bboxes)

    # 构建最终的字符串
    question_str = f"[{context_bboxes_str}]"

    return question_str

def build_prompt_yolo(image_path, context_boxes):
    """构建[cx, cy, w, h]归一化坐标提示（prompt）"""
    img = Image.open(image_path)
    width, height = img.size

    first_value = next(iter(context_boxes.values()))
    bboxes = first_value['bbox']

    # 计算每个边界框的归一化中心点坐标和宽度、高度并转换为字符串
    normalized_bboxes = []
    for bbox in bboxes:
        # 计算中心点坐标
        context_cx = (bbox[0] + bbox[2]) / 2
        context_cy = (bbox[1] + bbox[3]) / 2
        # 计算宽度和高度
        context_w = bbox[2] - bbox[0]
        context_h = bbox[3] - bbox[1]
        # 归一化坐标
        rel_cx = round(context_cx / width, 3)
        rel_cy = round(context_cy / height, 3)
        rel_w = round(context_w / width, 3)
        rel_h = round(context_h / height, 3)
        normalized_bboxes.append(f"[{rel_cx:.3f}, {rel_cy:.3f}, {rel_w:.3f}, {rel_h:.3f}]")

    # 将所有边界框的字符串组合成一个字符串
    context_bboxes_str = ", ".join(normalized_bboxes)

    # 构建最终的字符串
    question_str = f"[{context_bboxes_str}]"

    return question_str