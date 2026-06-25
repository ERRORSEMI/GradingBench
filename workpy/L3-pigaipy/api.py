def build_common_prompt_parts(desc, fmt, need_answer, answer_str):

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
你是一名专业的试卷批改助手，请批改试卷图片中的学生作答，严格按以下要求完成任务：

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

def build_prompt_L3(image_path, grouped_boxes, model_name, need_answer):
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

    answer_str = ""
    if need_answer:
        prompt_parts = []
        for qid, boxes in grouped_boxes.items():
            # 取当前组任意一个 box 的 ground_truth
            gt = next((b.get("ground_truth") for b in boxes if b.get("ground_truth")), None)
            if gt and gt != "答案暂无":
                part = f"{gt}; "
                prompt_parts.append(part)

        answer_str = "".join(prompt_parts)
    
    # 找到第一个匹配的前缀（转换为小写后匹配）
    model_name_lower = model_name.lower()  # 将模型名转为小写
    for prefix, (desc, fmt) in coord_map.items():
        prefix_lower = prefix.lower()  # 将前缀转为小写
        if model_name_lower.startswith(prefix_lower):
            return build_common_prompt_parts(desc, fmt, need_answer, answer_str)

    # 兜底
    return build_common_prompt_parts("0-1000 相对坐标", "[xmin,ymin,xmax,ymax]", need_answer, answer_str)
