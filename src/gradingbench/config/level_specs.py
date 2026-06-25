"""L1 / L2 / L3 任务层级差异配置（单一来源）。"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

Level = Literal["L1", "L2", "L3"]


@dataclass(frozen=True)
class LevelSpec:
    level: Level
    # 样本迭代：image_dir 按目录列 jpg；sample_ids 用 iter_sample_ids（L2 一页多题）
    iter_mode: Literal["image_dir", "sample_ids"]
    # L3 一页多题，question_id 需 parent_id-question_id 复合键
    multi_question: bool
    # L2 需在 prompt 中嵌入题目区域坐标
    needs_context: bool
    # answer-based 模式下过滤 ground_truth == "答案暂无"
    skip_gt_placeholder: bool
    api_max_retries: int
    api_retry_sleep_sec: int
    # L1 vLLM 推理时对 Kimi 输出做 think-marker  stripping
    strip_kimi_think_vllm: bool


LEVEL_SPECS: dict[Level, LevelSpec] = {
    "L1": LevelSpec(
        level="L1",
        iter_mode="image_dir",
        multi_question=False,
        needs_context=False,
        skip_gt_placeholder=True,
        api_max_retries=5,
        api_retry_sleep_sec=8,
        strip_kimi_think_vllm=True,
    ),
    "L2": LevelSpec(
        level="L2",
        iter_mode="sample_ids",
        multi_question=False,
        needs_context=True,
        skip_gt_placeholder=False,
        api_max_retries=5,
        api_retry_sleep_sec=5,
        strip_kimi_think_vllm=False,
    ),
    "L3": LevelSpec(
        level="L3",
        iter_mode="image_dir",
        multi_question=True,
        needs_context=False,
        skip_gt_placeholder=True,
        api_max_retries=2,
        api_retry_sleep_sec=8,
        strip_kimi_think_vllm=False,
    ),
}


def get_level_spec(level: str) -> LevelSpec:
    if level not in LEVEL_SPECS:
        raise ValueError(f"未知 level: {level}，可选: {list(LEVEL_SPECS)}")
    return LEVEL_SPECS[level]  # type: ignore[index]
