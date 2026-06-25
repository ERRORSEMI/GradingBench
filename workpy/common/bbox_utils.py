import math
import re
import json
from PIL import Image
from typing import List, Tuple

def calculate_iou(bboxes1, bboxes2):
    """
    计算两组边界框整体并集区域的交并比（IoU）
    
    参数:
        bboxes1: 第一组边界框列表，每个边界框为 [x1, y1, x2, y2]
        bboxes2: 第二组边界框列表，每个边界框为 [x1, y1, x2, y2]
        
    返回:
        两个区域整体的交并比（IoU）值
    """
    expanded_bboxes1 = []
    for box in bboxes1:
        x1, y1, x2, y2 = box
        # 扩大边界框：左移2，上移2，右移2，下移2
        expanded_x1 = x1 - 2
        expanded_y1 = y1 - 2
        expanded_x2 = x2 + 2
        expanded_y2 = y2 + 2
        expanded_bboxes1.append([expanded_x1, expanded_y1, expanded_x2, expanded_y2])
    
    bboxes1 = expanded_bboxes1
    
    area1 = calculate_union_area(bboxes1)
    # 计算第二组边界框的并集面积
    area2 = calculate_union_area(bboxes2)
    
    # 计算两个并集区域的交集面积
    inter_area = calculate_intersection_area(bboxes1, bboxes2)
    
    # 计算并集面积
    union_area = area1 + area2 - inter_area
    
    # 计算IoU（避免除以0）
    iou = inter_area / union_area if union_area > 0 else 0.0
    return iou

def calculate_union_area(bboxes):
    """
    使用扫描线算法计算多个边界框的并集面积
    """
    if not bboxes:
        return 0.0
    # 创建事件列表：每个边界框的开始和结束事件
    events = []
    for x1, y1, x2, y2 in bboxes:
        events.append((y1, x1, x2, 1))   # 进入事件
        events.append((y2, x1, x2, -1))  # 离开事件
    
    # 按y坐标排序事件
    events.sort(key=lambda e: (e[0], e[3]))
    
    # 初始化扫描线状态
    active_intervals = []
    last_y = events[0][0]
    total_area = 0.0
    
    for y, x1, x2, event_type in events:
        # 计算当前y区间的高度
        height = y - last_y
        
        # 计算当前活动区间的宽度总和
        if active_intervals:
            width = 0
            current_start = active_intervals[0][0]
            current_end = active_intervals[0][1]
            
            for interval in active_intervals[1:]:
                if interval[0] > current_end:
                    width += current_end - current_start
                    current_start, current_end = interval
                else:
                    current_end = max(current_end, interval[1])
            
            width += current_end - current_start
            total_area += width * height
        
        # 更新活动区间
        if event_type == 1:  # 进入事件
            active_intervals.append((x1, x2))
            active_intervals.sort(key=lambda i: i[0])
        else:  # 离开事件
            active_intervals.remove((x1, x2))
        
        last_y = y
    
    return total_area

def calculate_intersection_area(bboxes1, bboxes2):
    """
    计算两组边界框并集区域的交集面积
    """
    # 创建一个包含所有边界框的列表
    all_bboxes = []
    for bbox in bboxes1:
        all_bboxes.append((bbox, 1))  # 标记为第一组
    for bbox in bboxes2:
        all_bboxes.append((bbox, 2))  # 标记为第二组
    
    # 创建事件列表：每个边界框的开始和结束事件
    events = []
    for (x1, y1, x2, y2), group in all_bboxes:
        events.append((y1, x1, x2, group, 1))   # 进入事件
        events.append((y2, x1, x2, group, -1))  # 离开事件
    
    # 按y坐标排序事件
    events.sort(key=lambda e: e[0])
    
    # 初始化扫描线状态
    active_intervals1 = []  # 第一组活动区间
    active_intervals2 = []  # 第二组活动区间
    last_y = events[0][0] if events else 0
    total_area = 0.0
    
    for y, x1, x2, group, event_type in events:
        # 计算当前y区间的高度
        height = y - last_y
        
        # 计算当前交集区间的宽度
        if active_intervals1 and active_intervals2:
            # 合并第一组活动区间
            merged1 = merge_intervals(active_intervals1)
            # 合并第二组活动区间
            merged2 = merge_intervals(active_intervals2)
            
            # 计算两组活动区间的交集
            i = j = 0
            width = 0
            while i < len(merged1) and j < len(merged2):
                start = max(merged1[i][0], merged2[j][0])
                end = min(merged1[i][1], merged2[j][1])
                
                if start < end:
                    width += end - start
                
                # 移动到下一个区间
                if merged1[i][1] < merged2[j][1]:
                    i += 1
                else:
                    j += 1
            
            total_area += width * height
        
        # 更新活动区间
        if group == 1:
            if event_type == 1:  # 进入事件
                active_intervals1.append((x1, x2))
                active_intervals1.sort(key=lambda i: i[0])
            else:  # 离开事件
                active_intervals1.remove((x1, x2))
        else:  # group == 2
            if event_type == 1:  # 进入事件
                active_intervals2.append((x1, x2))
                active_intervals2.sort(key=lambda i: i[0])
            else:  # 离开事件
                active_intervals2.remove((x1, x2))
        
        last_y = y
    
    return total_area

def merge_intervals(intervals):
    """
    合并重叠的区间
    """
    if not intervals:
        return []
    
    intervals.sort(key=lambda i: i[0])
    merged = []
    current_start, current_end = intervals[0]
    
    for interval in intervals[1:]:
        if interval[0] <= current_end:
            current_end = max(current_end, interval[1])
        else:
            merged.append((current_start, current_end))
            current_start, current_end = interval
    
    merged.append((current_start, current_end))
    return merged

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

    return w_bar, h_bar

def parse_bbox(bbox):
    """将边界框解析为四个数值"""
    if isinstance(bbox, str):
        # 去掉首尾的方括号
        bbox = bbox.strip('[]')
        # 使用正则表达式分割字符串，同时处理逗号和空格
        parts = re.split(r'[,\s]+', bbox)
        if len(parts) != 4:
            return None
        try:
            x1, y1, x2, y2 = map(float, parts)
        except:
            return None
    elif isinstance(bbox, list):
        try:
            x1, y1, x2, y2 = bbox
        except:
            return None
    else:
        return None
    # print(x1, y1, x2, y2)
    return x1, y1, x2, y2

def infer_image_size_from_label(label_path):
    """从标注文件 marks 的最大坐标推断图片宽高（无需读图）。"""
    with open(label_path, encoding="utf-8") as f:
        data = json.load(f)
    max_x = 0
    max_y = 0
    for mark in data.get("marks", []):
        for p in mark.get("bbox_2d", []):
            max_x = max(max_x, p[0])
            max_y = max(max_y, p[1])
    return max(max_x + 1, 1), max(max_y + 1, 1)


def bbox_translate_with_size(original_width, original_height, model_results, model_name):
    """
    根据模型名对模型结果中的边界框进行坐标转换。
    参数：
        original_width / original_height: 原图像素尺寸。
        model_results: 模型预测结果，每一项包含 bbox_2d。
        model_name: 模型名称。
    """
    # 定义坐标转换逻辑
    coord_map = {
        "Qwen2.5":      "绝对像素坐标 [x1,y1,x2,y2]",
        "Qwen3":        "0-1000 相对坐标 [x1,y1,x2,y2]",
        "claude-opus-4.6":       "0-1 相对坐标 [x1,y1,x2,y2]",
        "claude":       "绝对像素坐标 [x1,y1,x2,y2]",
        "gemini":       "0-1000 相对坐标 [ymin,xmin,ymax,xmax]",
        "gpt":          "0-1 归一化中心坐标 [x_center,y_center,width,height]",
        "doubao-seed-2.0":  "0-1000 相对坐标 [x1,y1,x2,y2]",
        "doubao-seed":  "0-1 相对坐标 [x1,y1,x2,y2]",
        "deepseek":     "0-1 相对坐标 [x1,y1,x2,y2]",
        "doubao-1.5":   "0-1000 相对坐标 [x1,y1,x2,y2]",
        "InternVL":     "0-1000 相对坐标 [x1,y1,x2,y2]",
        "kimi":         "0-1 相对坐标 [x1,y1,x2,y2]",
        "gemma":        "0-1000 相对坐标 [x1,y1,x2,y2]",
        "glm":        "0-1000 相对坐标 [x1,y1,x2,y2]",
    }

    # 检查模型名是否匹配任何前缀
    matched_prefix = None
    for prefix in coord_map.keys():
        if model_name.startswith(prefix) or model_name.startswith(prefix.lower()):
            matched_prefix = prefix
            break

    if matched_prefix is None:
        raise ValueError(f"未知的模型名: {model_name}")

    # 特殊处理 Qwen 模型
    if matched_prefix == "Qwen2.5":
        new_width, new_height = smart_resize(original_height, original_width)
        width_ratio = original_width / new_width
        height_ratio = original_height / new_height
    else:
        new_width, new_height = original_width, original_height
        width_ratio = 1.0
        height_ratio = 1.0

    # print(original_width, original_height, new_width, new_height)
    # print(model_results)
    # 遍历每个模型结果，进行坐标转换
    seen_bboxes = set()
    translated_results = []
    for result in model_results:
        # print(result)
        translated_bboxes = []
        if 'bbox_2d' not in result:
            continue
        
        if isinstance(result['bbox_2d'], list) and not isinstance(result['bbox_2d'][0], list) and len(result['bbox_2d'])!=4:
            continue
        if isinstance(result['bbox_2d'], list) and isinstance(result['bbox_2d'][0], list):
            bboxes = result['bbox_2d']
        else:
            bboxes = [parse_bbox(result['bbox_2d']) ]
            
        for bbox in bboxes:
            try:
                if coord_map[matched_prefix] == "绝对像素坐标 [x1,y1,x2,y2]":
                    # 如果已经是绝对像素坐标，直接使用
                    x1, y1, x2, y2 = bbox
                    translated_bbox = [
                        int(x1 * width_ratio),
                        int(y1 * height_ratio),
                        int(x2 * width_ratio),
                        int(y2 * height_ratio)
                    ]

                    # print("before",bbox,"after", translated_bbox)
                elif coord_map[matched_prefix] == "0-1000 相对坐标 [ymin,xmin,ymax,xmax]":
                    # 从 0-1000 相对坐标转换为绝对像素坐标
                    ymin, xmin, ymax, xmax = bbox
                    translated_bbox = [
                        int(xmin / 1000 * new_width),
                        int(ymin / 1000 * new_height),
                        int(xmax / 1000 * new_width),
                        int(ymax / 1000 * new_height)
                    ]
                elif coord_map[matched_prefix] == "0-1000 相对坐标 [x1,y1,x2,y2]":
                    # 从 0-1 相对坐标转换为绝对像素坐标
                    x1, y1, x2, y2 = bbox
                    if x2<1 and y2<1:
                        translated_bbox = [
                            int(x1 * new_width),
                            int(y1 * new_height),
                            int(x2 * new_width),
                            int(y2 * new_height)
                        ]
                    else:
                        translated_bbox = [
                            int(x1 / 1000 * new_width),
                            int(y1 / 1000 * new_height),
                            int(x2 / 1000 * new_width),
                            int(y2 / 1000 * new_height)
                        ]
                elif coord_map[matched_prefix] == "0-1 相对坐标 [x1,y1,x2,y2]":
                    # 从 0-1 相对坐标转换为绝对像素坐标
                    x1, y1, x2, y2 = bbox
                    translated_bbox = [
                        int(x1 * new_width),
                        int(y1 * new_height),
                        int(x2 * new_width),
                        int(y2 * new_height)
                    ]
                elif coord_map[matched_prefix] == "0-1 相对坐标 [x1,y1,x2,y2]":
                    # 从 0-1 相对坐标转换为绝对像素坐标
                    x1, y1, x2, y2 = bbox
                    translated_bbox = [
                        int(x1 * new_width),
                        int(y1 * new_height),
                        int(x2 * new_width),
                        int(y2 * new_height)
                    ]
                elif coord_map[matched_prefix] == "0-1 归一化中心坐标 [x_center,y_center,width,height]":
                    # 从 0-1 归一化中心坐标转换为绝对像素坐标
                    x_center, y_center, width, height = bbox
                    x1 = int((x_center - width / 2) * new_width)
                    y1 = int((y_center - height / 2) * new_height)
                    x2 = int((x_center + width / 2) * new_width)
                    y2 = int((y_center + height / 2) * new_height)
                    translated_bbox = [x1, y1, x2, y2]
                else:
                    raise ValueError(f"不支持的坐标格式: {coord_map[matched_prefix]}")
            except (ValueError, TypeError):
                # 如果 bbox 格式不正确，跳过当前 bbox
                print(f"Invalid bbox format: {bbox}. Skipping this bbox.")
                continue
            x1,y1,x2,y2 = translated_bbox
            
            x1=max(0,x1-2)
            x2=min(original_width,x2+2)
            y1=max(0,y1-2)
            y2=min(original_height,y2+2)
            if x1>=x2:
                continue
            if y1>=y2:
                continue
            translated_bboxes.append([x1,y1,x2,y2])
            

        unique_bboxes = []
        for bbox in translated_bboxes:
            # 将列表转换为元组以便哈希
            # print(bbox)
            bbox_tuple = tuple(bbox)
            if bbox_tuple in seen_bboxes:
                # 重复的边界框，替换为[0,0,1,1]
                unique_bboxes.append([0, 0, 1, 1])
            else:
                # 新的边界框，添加到集合和结果列表
                seen_bboxes.add(bbox_tuple)
                unique_bboxes.append(bbox)
        
        # 更新为去重后的边界框列表
        translated_bboxes = unique_bboxes
        
        if len(translated_bboxes) == 0:
            continue
        if 'correctness' in result:
            translated_results.append({
                'bbox_2d': translated_bboxes,
                'text': result.get('text', ''),
                'thinking': result.get('thinking', ''),
                'answer': result.get('answer', ''),
                'correctness': result['correctness']
            })
    # print(translated_results)
    return translated_results


def bbox_translate(image_path, model_results, model_name):
    """从图片路径读取尺寸后做坐标转换。"""
    with Image.open(image_path) as original_image:
        original_width, original_height = original_image.size
    return bbox_translate_with_size(
        original_width, original_height, model_results, model_name
    )