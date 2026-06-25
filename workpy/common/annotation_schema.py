"""Annotation field constants and parsers used by inference / evaluation."""
from __future__ import annotations

import re

CN_ONLY = re.compile(r"^[\u4e00-\u9fff]+$")

# markLabel values
MARK_RIGHT = "right"
MARK_WRONG = "wrong"
MARK_QUESTION = "question"

MARK_LABEL_FROM_CN = {
    "正确": MARK_RIGHT,
    "错误": MARK_WRONG,
    "整题": MARK_QUESTION,
}

ANNO_RESULT_FROM_CN = {"正确": MARK_RIGHT, "错误": MARK_WRONG}

# Paper-aligned canonical values (spaces -> _, / -> -)
OTHER = "Other"
CONTENT_FORMAT_VALUES = {"Text-only", "Chart-based", OTHER}
QUESTION_TYPE_VALUES = {
    "Multiple-choice",
    "Fill-in-the-blank",
    "True-False",
    "Problem-Solving",
    OTHER,
}
STAGE_VALUES = {"Elementary", "Junior_High", "Senior_High", OTHER}

# Chinese attribute keys -> English
ATTR_KEY_FROM_CN = {
    "拉框结果": "anno_result",
    "纬度1": "dimension1",
    "纬度2": "dimension2",
    "题目id": "question_id",
}

# Chinese metadata values -> English
DIM1_FROM_CN = {
    "文本": "Text-only",
    "图表": "Chart-based",
    "作图": "Chart-based",
    "其他": OTHER,
}
DIM2_FROM_CN = {
    "填空": "Fill-in-the-blank",
    "选择": "Multiple-choice",
    "判断": "True-False",
    "解答": "Problem-Solving",
    "开放": "Problem-Solving",
    "简单口算": "Fill-in-the-blank",
    "口算": "Fill-in-the-blank",
    "其他": OTHER,
}
STAGE_FROM_CN = {
    "小学": "Elementary",
    "初中": "Junior_High",
    "高中": "Senior_High",
    "其他": OTHER,
}

# Catch-all for rare / legacy Chinese metadata labels
METADATA_CATCHALL_CN = {
    "半对": OTHER,
    "未作答": OTHER,
    "未知": OTHER,
}

DIM1_FROM_LEGACY = {"Text": "Text-only", "Chart": "Chart-based"}
DIM2_FROM_LEGACY = {
    "Fill-in-blank": "Fill-in-the-blank",
    "True-false": "True-False",
    "True/False": "True-False",
    "Solution": "Problem-Solving",
    "Multiple-choice": "Multiple-choice",
}
STAGE_FROM_LEGACY = {
    "Primary": "Elementary",
    "Middle": "Junior_High",
    "Junior High": "Junior_High",
    "High": "Senior_High",
    "Senior High": "Senior_High",
}


def _fallback_cn_metadata(value: str) -> str:
    if not value:
        return value
    if value in METADATA_CATCHALL_CN:
        return METADATA_CATCHALL_CN[value]
    if CN_ONLY.match(value):
        return OTHER
    return value


def _canonical(field: str, value: str) -> str:
    if not value:
        return value
    if field == "dimension1":
        if value in CONTENT_FORMAT_VALUES:
            return value
        v = DIM1_FROM_CN.get(value) or DIM1_FROM_LEGACY.get(value, value)
        return _fallback_cn_metadata(v)
    if field == "dimension2":
        if value in QUESTION_TYPE_VALUES:
            return value
        v = DIM2_FROM_CN.get(value) or DIM2_FROM_LEGACY.get(value, value)
        return _fallback_cn_metadata(v)
    if field == "educational_stage":
        if value in STAGE_VALUES:
            return value
        v = STAGE_FROM_CN.get(value) or STAGE_FROM_LEGACY.get(value, value)
        return _fallback_cn_metadata(v)
    if field == "anno_result":
        return ANNO_RESULT_FROM_CN.get(value, value)
    return value


def _normalize_attributes(attrs: dict) -> dict:
    out: dict = {}
    for key, val in attrs.items():
        en_key = ATTR_KEY_FROM_CN.get(key, key)
        if en_key in ("dimension1", "dimension2"):
            out[en_key] = _canonical(en_key, val if isinstance(val, str) else val)
        elif en_key == "anno_result":
            out[en_key] = _canonical("anno_result", val if isinstance(val, str) else val)
        else:
            out[en_key] = val
    return out


def normalize_mark(mark: dict) -> dict:
    attrs = _normalize_attributes(dict(mark.get("attributes") or {}))

    label = MARK_LABEL_FROM_CN.get(mark.get("markLabel", ""), mark.get("markLabel", ""))
    if label and CN_ONLY.match(label):
        label = OTHER

    stage_raw = mark.get("educational_stage") or mark.get("学段", "")
    stage = _canonical("educational_stage", stage_raw)

    out = {"bbox_2d": mark["bbox_2d"], "markLabel": label, "attributes": attrs}
    if stage:
        out["educational_stage"] = stage
    return out


def normalize_marks(marks: list) -> list:
    return [normalize_mark(m) for m in marks]


def get_mark_label(mark: dict) -> str:
    return MARK_LABEL_FROM_CN.get(mark.get("markLabel", ""), mark.get("markLabel", ""))


def is_question_mark(mark: dict) -> bool:
    return get_mark_label(mark) == MARK_QUESTION


def is_right_mark(mark: dict) -> bool:
    return get_mark_label(mark) == MARK_RIGHT


def get_educational_stage(mark: dict) -> str:
    raw = mark.get("educational_stage") or mark.get("学段", "")
    return _canonical("educational_stage", raw)


def get_attr(mark: dict, en_key: str, cn_key: str, default=None):
    attrs = mark.get("attributes") or {}
    return attrs.get(en_key, attrs.get(cn_key, default))


def get_dimension1(mark: dict) -> str:
    return _canonical("dimension1", get_attr(mark, "dimension1", "纬度1", ""))


def get_dimension2(mark: dict) -> str:
    return _canonical("dimension2", get_attr(mark, "dimension2", "纬度2", ""))


def get_question_id_attr(mark: dict) -> str:
    return get_attr(mark, "question_id", "题目id", "")


DIM1_ORDER = ["Text-only", "Chart-based"]
DIM2_ORDER = ["Fill-in-the-blank", "Multiple-choice", "True-False", "Problem-Solving"]
STAGE_ORDER = ["Elementary", "Junior_High", "Senior_High"]

FIXED_TOTALS = {
    "subject-science": 684,
    "subject-math": 640,
    "subject-chinese": 693,
    "subject-english": 650,
    "subject-liberal_arts": 657,
    "dimension1-Text-only": 2225,
    "dimension1-Chart-based": 1081,
    "dimension2-Fill-in-the-blank": 1960,
    "dimension2-Multiple-choice": 761,
    "dimension2-True-False": 75,
    "dimension2-Problem-Solving": 214,
    "educational_stage-Elementary": 1460,
    "educational_stage-Junior_High": 1288,
    "educational_stage-Senior_High": 576,
}
