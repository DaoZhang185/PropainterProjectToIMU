import sys
import os

# =================================================================
# 【路径修复】确保脚本能引用上级目录（根目录）的模块
# =================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import json

def load_poses_from_json(json_path):
    """
    解析 JSON，返回坐标信息与时效范围映射字典
    返回格式: poses, time_ranges
    """
    if not json_path or not os.path.exists(json_path):
        print(f"✗ 错误: Mask JSON文件未找到: {json_path}")
        return None, {}

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        poses = {}
        time_ranges = {}  # 存放有时效性的区域

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
                pose_key = region_key  # 动态 key，如 other_1, other_2

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

                # ====== 提取生效时间 ======
                tr = str(region_data.get('timeRange', '')).strip()
                if tr and '-' in tr:
                    try:
                        sf, ef = map(int, tr.replace(' ', '').split('-'))
                        time_ranges[pose_key] = [sf, ef]
                        print(f"  ⏱ 识别到时效限制 -> {type_name}: 帧 {sf} 至 {ef}")
                    except Exception:
                        pass

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