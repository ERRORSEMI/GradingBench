"""各模型 bbox 坐标格式：Prompt 说明与输出反变换的唯一配置源。"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Sequence, Tuple

from PIL import Image


def _smart_resize(height: int, width: int) -> Tuple[int, int]:
    from gradingbench.coords.bbox import smart_resize

    return smart_resize(height, width)


class CoordKind(Enum):
    """模型 bbox_2d 的坐标语义。"""

    ABS_XYXY = "abs_xyxy"
    ABS_XYXY_RESIZED = "abs_xyxy_resized"  # Qwen2.5：smart_resize 输入空间内的绝对像素
    SCALE_1000_XYXY = "scale_1000_xyxy"
    SCALE_1000_YXYX = "scale_1000_yxyx"  # Gemini：[ymin, xmin, ymax, xmax]
    NORM_XYXY = "norm_xyxy"
    NORM_CENTER = "norm_center"  # GPT：[cx, cy, w, h] 归一化


@dataclass(frozen=True)
class CoordSpec:
    kind: CoordKind
    desc: str
    fmt: str


DEFAULT_COORD_SPEC = CoordSpec(
    CoordKind.SCALE_1000_XYXY,
    "0-1000 相对坐标",
    "[xmin,ymin,xmax,ymax]",
)

# 前缀按长度降序排列，保证 claude-opus-4.6 优先于 claude 等
_PREFIX_SPECS: Sequence[Tuple[str, CoordSpec]] = (
    ("claude-opus-4.6", CoordSpec(CoordKind.NORM_XYXY, "0-1 相对坐标", "[xmin,ymin,xmax,ymax]")),
    ("doubao-seed-2.0", CoordSpec(CoordKind.SCALE_1000_XYXY, "0-1000 相对坐标", "[xmin,ymin,xmax,ymax]")),
    ("doubao-1.5-vision-pro", CoordSpec(CoordKind.SCALE_1000_XYXY, "0-1000 相对坐标", "[xmin,ymin,xmax,ymax]")),
    ("doubao-1.5", CoordSpec(CoordKind.SCALE_1000_XYXY, "0-1000 相对坐标", "[xmin,ymin,xmax,ymax]")),
    ("Qwen2.5-VL", CoordSpec(CoordKind.ABS_XYXY_RESIZED, "绝对像素坐标", "[xmin,ymin,xmax,ymax]")),
    ("Qwen2.5", CoordSpec(CoordKind.ABS_XYXY_RESIZED, "绝对像素坐标", "[xmin,ymin,xmax,ymax]")),
    ("deepseek-vl2", CoordSpec(CoordKind.NORM_XYXY, "0-1 相对坐标", "[xmin,ymin,xmax,ymax]")),
    ("deepseek", CoordSpec(CoordKind.NORM_XYXY, "0-1 相对坐标", "[xmin,ymin,xmax,ymax]")),
    ("doubao-seed", CoordSpec(CoordKind.NORM_XYXY, "0-1 相对坐标", "[xmin,ymin,xmax,ymax]")),
    ("InternVL3", CoordSpec(CoordKind.SCALE_1000_XYXY, "0-1000 相对坐标", "[xmin,ymin,xmax,ymax]")),
    ("InternVL", CoordSpec(CoordKind.SCALE_1000_XYXY, "0-1000 相对坐标", "[xmin,ymin,xmax,ymax]")),
    ("gemma-3", CoordSpec(CoordKind.SCALE_1000_XYXY, "0-1000 相对坐标", "[xmin,ymin,xmax,ymax]")),
    ("gemma", CoordSpec(CoordKind.SCALE_1000_XYXY, "0-1000 相对坐标", "[xmin,ymin,xmax,ymax]")),
    ("Qwen3", CoordSpec(CoordKind.SCALE_1000_XYXY, "0-1000 相对坐标", "[xmin,ymin,xmax,ymax]")),
    ("claude", CoordSpec(CoordKind.ABS_XYXY, "绝对像素坐标", "[xmin,ymin,xmax,ymax]")),
    ("gemini", CoordSpec(CoordKind.SCALE_1000_YXYX, "0-1000 相对坐标", "[ymin,xmin,ymax,xmax]")),
    ("gpt", CoordSpec(CoordKind.NORM_CENTER, "0-1 归一化中心坐标", "[x_center,y_center,width,height]")),
    ("kimi", CoordSpec(CoordKind.NORM_XYXY, "0-1 相对坐标", "[xmin,ymin,xmax,ymax]")),
    ("glm", CoordSpec(CoordKind.SCALE_1000_XYXY, "0-1000 相对坐标", "[xmin,ymin,xmax,ymax]")),
)

_PREFIX_TABLE = sorted(_PREFIX_SPECS, key=lambda item: len(item[0]), reverse=True)


def resolve_coord_spec(model_name: str, *, default: Optional[CoordSpec] = None) -> CoordSpec:
    """按模型名前缀匹配坐标规格。default=None 且未匹配时抛出 ValueError。"""
    name = model_name.lower()
    for prefix, spec in _PREFIX_TABLE:
        if name.startswith(prefix.lower()):
            return spec
    if default is not None:
        return default
    raise ValueError(f"未知的模型名: {model_name}")


def _qwen25_work_size(orig_w: int, orig_h: int) -> Tuple[int, int, float, float]:
    """Qwen2.5 推理输入尺寸及映射回原图的比例。"""
    work_w, work_h = _smart_resize(orig_h, orig_w)
    return work_w, work_h, orig_w / work_w, orig_h / work_h


def format_bbox_for_prompt(
    bbox: Sequence[float],
    spec: CoordSpec,
    img_w: int,
    img_h: int,
    *,
    scale_x: float = 1.0,
    scale_y: float = 1.0,
) -> str:
    """将原图 xyxy 框转为 Prompt 中使用的坐标字符串。"""
    x1, y1, x2, y2 = bbox

    if spec.kind in (CoordKind.ABS_XYXY, CoordKind.ABS_XYXY_RESIZED):
        coords = (
            int(round(x1 * scale_x)),
            int(round(y1 * scale_y)),
            int(round(x2 * scale_x)),
            int(round(y2 * scale_y)),
        )
        return f"[{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}]"

    if spec.kind == CoordKind.SCALE_1000_XYXY:
        coords = (
            int(x1 / img_w * 1000),
            int(y1 / img_h * 1000),
            int(x2 / img_w * 1000),
            int(y2 / img_h * 1000),
        )
        return f"[{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}]"

    if spec.kind == CoordKind.SCALE_1000_YXYX:
        coords = (
            int(y1 / img_h * 1000),
            int(x1 / img_w * 1000),
            int(y2 / img_h * 1000),
            int(x2 / img_w * 1000),
        )
        return f"[{coords[0]}, {coords[1]}, {coords[2]}, {coords[3]}]"

    if spec.kind == CoordKind.NORM_XYXY:
        return (
            f"[{round(x1 / img_w, 2):.2f}, {round(y1 / img_h, 2):.2f}, "
            f"{round(x2 / img_w, 2):.2f}, {round(y2 / img_h, 2):.2f}]"
        )

    if spec.kind == CoordKind.NORM_CENTER:
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        w = x2 - x1
        h = y2 - y1
        return (
            f"[{round(cx / img_w, 3):.3f}, {round(cy / img_h, 3):.3f}, "
            f"{round(w / img_w, 3):.3f}, {round(h / img_h, 3):.3f}]"
        )

    raise ValueError(f"不支持的坐标格式: {spec.kind}")


def format_question_regions(image_path: str, context_boxes: dict, model_name: str) -> str:
    """L2：将待批改题目区域坐标格式化为与模型输出一致的字符串。"""
    spec = resolve_coord_spec(model_name, default=DEFAULT_COORD_SPEC)
    with Image.open(image_path) as img:
        img_w, img_h = img.size

    scale_x, scale_y = 1.0, 1.0
    if spec.kind == CoordKind.ABS_XYXY_RESIZED:
        work_w, work_h = _smart_resize(img_h, img_w)
        scale_x = work_w / img_w
        scale_y = work_h / img_h

    first_value = next(iter(context_boxes.values()))
    bboxes = first_value["bbox"]
    parts = [
        format_bbox_for_prompt(bbox, spec, img_w, img_h, scale_x=scale_x, scale_y=scale_y)
        for bbox in bboxes
    ]
    return f"[{', '.join(parts)}]"


def translate_bbox_to_abs_xyxy(
    bbox: Sequence[float],
    spec: CoordSpec,
    orig_w: int,
    orig_h: int,
) -> List[int]:
    """将模型输出的单个 bbox 转为原图绝对像素 [xmin, ymin, xmax, ymax]。"""
    if spec.kind == CoordKind.ABS_XYXY_RESIZED:
        _, _, w_ratio, h_ratio = _qwen25_work_size(orig_w, orig_h)
        x1, y1, x2, y2 = bbox
        return [
            int(x1 * w_ratio),
            int(y1 * h_ratio),
            int(x2 * w_ratio),
            int(y2 * h_ratio),
        ]

    if spec.kind == CoordKind.ABS_XYXY:
        x1, y1, x2, y2 = bbox
        return [int(x1), int(y1), int(x2), int(y2)]

    if spec.kind == CoordKind.SCALE_1000_YXYX:
        ymin, xmin, ymax, xmax = bbox
        return [
            int(xmin / 1000 * orig_w),
            int(ymin / 1000 * orig_h),
            int(xmax / 1000 * orig_w),
            int(ymax / 1000 * orig_h),
        ]

    if spec.kind == CoordKind.SCALE_1000_XYXY:
        x1, y1, x2, y2 = bbox
        # 兼容个别模型误输出 0-1 范围
        if x2 < 1 and y2 < 1:
            return [
                int(x1 * orig_w),
                int(y1 * orig_h),
                int(x2 * orig_w),
                int(y2 * orig_h),
            ]
        return [
            int(x1 / 1000 * orig_w),
            int(y1 / 1000 * orig_h),
            int(x2 / 1000 * orig_w),
            int(y2 / 1000 * orig_h),
        ]

    if spec.kind == CoordKind.NORM_XYXY:
        x1, y1, x2, y2 = bbox
        return [
            int(x1 * orig_w),
            int(y1 * orig_h),
            int(x2 * orig_w),
            int(y2 * orig_h),
        ]

    if spec.kind == CoordKind.NORM_CENTER:
        cx, cy, w, h = bbox
        x1 = int((cx - w / 2) * orig_w)
        y1 = int((cy - h / 2) * orig_h)
        x2 = int((cx + w / 2) * orig_w)
        y2 = int((cy + h / 2) * orig_h)
        return [x1, y1, x2, y2]

    raise ValueError(f"不支持的坐标格式: {spec.kind}")
