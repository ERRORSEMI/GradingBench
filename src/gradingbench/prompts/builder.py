"""L1/L2/L3 批改 Prompt 构建。"""
from __future__ import annotations

from gradingbench.coords.specs import DEFAULT_COORD_SPEC, format_question_regions, resolve_coord_spec
from gradingbench.config.level_specs import Level, get_level_spec


def _build_answer_str(grouped_boxes: dict, need_answer: bool, level: Level) -> str:
    if not need_answer:
        return ""

    spec = get_level_spec(level)
    parts = []
    for _qid, boxes in grouped_boxes.items():
        gt = next((b.get("ground_truth") for b in boxes if b.get("ground_truth")), None)
        if not gt:
            continue
        if spec.skip_gt_placeholder and gt == "答案暂无":
            continue
        parts.append(f"{gt}; ")
    return "".join(parts)


def _build_prompt_body(
    desc: str,
    fmt: str,
    need_answer: bool,
    answer_str: str,
    *,
    level: Level,
    question_str: str = "",
) -> str:
    if need_answer and answer_str:
        answer_str = f"""参考答案（核心依据）
{answer_str}
> 说明：非必要无需自主推导题目答案，直接以本参考答案为判断基准；
"""
        thinking_rule = "- thinking字段需：简要说明 “学生答案与参考答案的匹配逻辑”；"
    else:
        answer_str = ""
        thinking_rule = (
            "thinking字段需按 “题目分析→步骤推导→结论” 结构书写，需包含关键依据"
            "（数学 / 理综说明公式定理，语文 / 英语说明语法 / 语境，文综关联知识点）。"
            "得到的题目答案输出到answer字段"
        )

    if level == "L2":
        intro = (
            f"你是一名专业的试卷批改助手，这是一张完整的学生试卷，"
            f"请你仅批改{desc}{fmt}{question_str}处的题目，忽略其他题目的内容，严格按以下要求完成任务："
        )
    else:
        intro = "你是一名专业的试卷批改助手，请批改试卷图片中的学生作答，严格按以下要求完成任务："

    return f"""
{intro}

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
"""


def build_prompt(
    level: Level,
    image_path: str,
    grouped_boxes: dict,
    model_name: str,
    need_answer: bool,
    context_boxes: dict | None = None,
) -> str:
    """按 level 与模型坐标规格构建完整 prompt。"""
    level_spec = get_level_spec(level)
    coord = resolve_coord_spec(model_name, default=DEFAULT_COORD_SPEC)
    answer_str = _build_answer_str(grouped_boxes, need_answer, level)

    question_str = ""
    if level_spec.needs_context:
        if context_boxes is None:
            raise ValueError("L2 需要提供 context_boxes")
        question_str = format_question_regions(image_path, context_boxes, model_name)

    return _build_prompt_body(
        coord.desc,
        coord.fmt,
        need_answer,
        answer_str,
        level=level,
        question_str=question_str,
    )


# 向后兼容别名
def build_prompt_L1(image_path, grouped_boxes, model_name, need_answer):
    return build_prompt("L1", image_path, grouped_boxes, model_name, need_answer)


def build_prompt_L2(image_path, grouped_boxes, context_boxes, model_name, need_answer):
    return build_prompt("L2", image_path, grouped_boxes, model_name, need_answer, context_boxes)


def build_prompt_L3(image_path, grouped_boxes, model_name, need_answer):
    return build_prompt("L3", image_path, grouped_boxes, model_name, need_answer)
