import sys
import os
import json

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


def parse_time_str(t_str):
    parts = str(t_str).split(':')
    if len(parts) == 3:
        return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    elif len(parts) == 2:
        return int(parts[0]) * 60 + float(parts[1])
    return float(parts[0])


def load_poses_from_json(json_path):
    if not json_path or not os.path.exists(json_path):
        return None, {}, None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        poses = {}
        reference_times = {}
        subtitle_pos = None  # 【核心保留】单独提取字幕坐标，禁止计入修复遮罩

        type_mapping = {'台标': '1', 'logo': '1', '字幕': '2', 'subtitle': '2', '剧名': '3', 'title': '3'}
        regions = data.get('regions', data) if isinstance(data, dict) else {}

        for region_key, region_data in regions.items():
            type_name = region_data.get('typeName', region_key)

            # 【截获字幕定位框】
            if region_key == 'sub_pos' or type_name == '蒙文字幕投放区':
                large_boxes = region_data.get('largeBoxes', [])
                if large_boxes:
                    box = large_boxes[0]
                    if 'x' in box:
                        subtitle_pos = [
                            int(float(box['x'])), int(float(box['y'])),
                            int(float(box['x']) + float(box['width'])), int(float(box['y']) + float(box['height']))
                        ]
                continue  # 【跳出】绝不把它混入遮罩数组！

            pose_key = None
            if type_name in type_mapping:
                pose_key = type_mapping[type_name]
            elif region_key in type_mapping:
                pose_key = type_mapping[region_key]
            elif '其他' in type_name or 'other' in region_key:
                pose_key = region_key

            if pose_key:
                large_list, small_list = [], []
                ref_t = region_data.get('reference_time', '')
                if ref_t: reference_times[pose_key] = parse_time_str(ref_t)

                for box in region_data.get('largeBoxes', []):
                    if 'polygon' in box and box['polygon']:
                        large_list.append(box['polygon'])
                    elif 'x' in box:
                        large_list.append([int(box['x']), int(box['y']), int(box['x']) + int(box['width']),
                                           int(box['y']) + int(box['height'])])

                for box in region_data.get('smallBoxes', []):
                    if 'polygon' in box and box['polygon']:
                        small_list.append(box['polygon'])
                    elif 'x' in box:
                        small_list.append([int(box['x']), int(box['y']), int(box['x']) + int(box['width']),
                                           int(box['y']) + int(box['height'])])

                coords_list = small_list if small_list else large_list
                if coords_list: poses[pose_key] = coords_list

        # 返回三个参数：遮罩矩阵、追踪时间、字幕挂载区
        return poses, reference_times, subtitle_pos
    except Exception as e:
        print(f"✗ 加载Mask JSON失败: {e}")
        return None, {}, None