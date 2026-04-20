import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import json

def parse_time_str(t_str):
    """
    将时分秒字符串 (如 '0:3:23' 或 '00:03:23') 转换为总秒数 (float)。
    """
    parts = t_str.split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])

def load_poses_from_json(json_path):
    if not json_path or not os.path.exists(json_path):
        print(f"✗ 错误: Mask JSON文件未找到: {json_path}")
        return None, {}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        poses = {}
        time_ranges = {}

        type_mapping = {
            '台标': '1', 'logo': '1',
            '字幕': '2', 'subtitle': '2',
            '剧名': '3', 'title': '3'
        }

        regions = data.get('regions', data) if isinstance(data, dict) else {}
        print(f"✓ 开始解析Mask JSON，找到区域: {list(regions.keys())}")

        for region_key, region_data in regions.items():
            type_name = region_data.get('typeName', region_key)
            pose_key = None

            if type_name in type_mapping:
                pose_key = type_mapping[type_name]
            elif region_key in type_mapping:
                pose_key = type_mapping[region_key]
            elif '其他' in type_name or 'other' in region_key:
                pose_key = region_key

            if pose_key:
                large_list = []
                small_list = []

                def process_boxes(box_list, box_type, target_list):
                    for box in box_list:
                        try:
                            if 'polygon' in box and box['polygon']:
                                target_list.append(box['polygon'])
                            elif all(k in box for k in ('x', 'y', 'width', 'height')):
                                x, y = int(float(box['x'])), int(float(box['y']))
                                w, h = int(float(box['width'])), int(float(box['height']))
                                target_list.append([x, y, x + w, y + h])
                        except Exception as e:
                            print(f"  ⚠ 解析 {type_name} {box_type} 出错: {e}")

                process_boxes(region_data.get('largeBoxes', []), 'largeBox', large_list)
                process_boxes(region_data.get('smallBoxes', []), 'smallBox', small_list)

                # ====== 【核心优化】完美解析人类直观时间格式 ======
                tr = str(region_data.get('timeRange', '')).strip()
                if tr and '-' in tr:
                    try:
                        start_str, end_str = tr.split('-')
                        start_sec = parse_time_str(start_str.strip())
                        end_sec = parse_time_str(end_str.strip())
                        time_ranges[pose_key] = [start_sec, end_sec]
                        print(f"  ⏱ 识别到时效限制 -> {type_name}: 第 {start_sec}秒 至 第 {end_sec}秒")
                    except Exception as e:
                        print(f"  ⚠ 时间解析失败: {e}，将默认全时段生效")

                # 防涂白机制：除了字幕，其他全用精准区域
                if pose_key in ['1', '3'] or 'other' in pose_key:
                    coords_list = small_list if small_list else large_list
                else:
                    coords_list = large_list + small_list

                if coords_list:
                    poses[pose_key] = coords_list
                    print(f"  ✓ 加载 {type_name} (key={pose_key}): {len(coords_list)} 个有效形状")
            else:
                print(f"  ℹ 跳过未知区域类型: {region_key} / {type_name}")

        return poses, time_ranges

    except Exception as e:
        print(f"✗ 加载Mask JSON失败: {e}")
        return None, {}