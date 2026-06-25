import json

from common.annotation_schema import (
    get_dimension1,
    get_dimension2,
    get_educational_stage,
    get_question_id_attr,
    is_question_mark,
    is_right_mark,
)


def parse_label_file(label_path):
    with open(label_path, encoding='utf-8') as f:
        data = json.load(f)
    return parse_label_data(data)


def parse_label_data(data):
    context_boxes = {}
    answer_boxes = []

    for mark in data.get("marks", []):
        points = mark["bbox_2d"]
        xs = [p[0] for p in points]
        ys = [p[1] for p in points]
        bbox = [min(xs), min(ys), max(xs), max(ys)]

        attrs = mark.get("attributes", {})

        if is_question_mark(mark):
            qid = get_question_id_attr(mark)
            if qid not in context_boxes:
                context_boxes[qid] = {"bbox": []}
            context_boxes[qid]["bbox"].append(bbox)
        else:
            full_id = attrs.get("id", "")
            parent_id, _, sub_id = full_id.partition("-")
            if not parent_id:
                parent_id, sub_id = full_id, ""

            answer_boxes.append({
                "bbox": bbox,
                "correctness": is_right_mark(mark),
                "question_id": sub_id,
                "parent_id": parent_id,
                "dim1": get_dimension1(mark),
                "dim2": get_dimension2(mark),
                "dim3": get_educational_stage(mark),
                "ground_truth": attrs.get("std"),
                "hand_written": attrs.get("hand"),
            })

    return context_boxes, answer_boxes


def group_boxes_by_question(answer_boxes, multi_question=False):
    grouped_boxes = {}
    for box in answer_boxes:
        if multi_question:
            qid = f"{box['parent_id']}-{box['question_id']}"
        else:
            qid = box['question_id']
        if qid not in grouped_boxes:
            grouped_boxes[qid] = []
        grouped_boxes[qid].append(box)
    return grouped_boxes


def group_boxes_by_parent(answer_boxes):
    grouped_boxes = {}
    for box in answer_boxes:
        parent_id = box['parent_id']
        question_id = box['question_id']
        if parent_id not in grouped_boxes:
            grouped_boxes[parent_id] = {}
        if question_id not in grouped_boxes[parent_id]:
            grouped_boxes[parent_id][question_id] = []
        grouped_boxes[parent_id][question_id].append(box)
    return grouped_boxes
