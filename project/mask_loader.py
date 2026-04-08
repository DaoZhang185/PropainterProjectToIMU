import sys
import os

# =================================================================
# 【路径修复】确保脚本能引用上级目录（根目录）的模块 (core, model, utils)
# =================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../ProPainter/project
root_dir = os.path.dirname(current_dir)                   # .../ProPainter
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# =================================================================

import json

def load_poses_from_json(json_path):
    """
    从前端生成的JSON文件中读取Mask坐标并转换为poses格式
    """
    if not json_path or not os.path.exists(json_path):
        print(f"✗ 错误: Mask JSON文件未找到: {json_path}")
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        poses = {}

        # 映射关系: typeName -> key
        type_mapping = {
            '台标': '1',
            '字幕': '2',
            '剧名': '3',
            'logo': '1',  # 兼容前端 key
            'subtitle': '2',  # 兼容前端 key
            'title': '3'  # 兼容前端 key
        }

        # 获取区域数据
        regions = {}
        if isinstance(data, dict):
            if 'regions' in data:
                regions = data['regions']
            else:
                regions = data

        print(f"✓ 开始解析Mask JSON，找到区域: {list(regions.keys())}")

        for region_key, region_data in regions.items():
            type_name = region_data.get('typeName', region_key)

            pose_key = None
            if type_name in type_mapping:
                pose_key = type_mapping[type_name]
            elif region_key in type_mapping:
                pose_key = type_mapping[region_key]

            if pose_key:
                large_list = []
                small_list = []

                def process_boxes(box_list, box_type, target_list):
                    for box in box_list:
                        try:
                            if 'polygon' in box and box['polygon']:
                                target_list.append(box['polygon'])
                            elif all(k in box for k in ('x', 'y', 'width', 'height')):
                                x = int(float(box['x']))
                                y = int(float(box['y']))
                                w = int(float(box['width']))
                                h = int(float(box['height']))
                                target_list.append([x, y, x + w, y + h])
                            else:
                                print(f"  ⚠ 忽略未知的 {type_name} {box_type} 数据格式")
                        except Exception as e:
                            print(f"  ⚠ 解析 {type_name} {box_type} 出错: {e}")

                process_boxes(region_data.get('largeBoxes', []), 'largeBox', large_list)
                process_boxes(region_data.get('smallBoxes', []), 'smallBox', small_list)

                # =========================================================
                # 【核心修复】智能防覆盖机制，防止大框把魔法棒涂成白板
                # =========================================================
                if pose_key in ['1', '3']:  # 静态区域（台标、剧名）
                    # 如果画了魔法棒/小框，就只画魔法棒，直接抛弃红色大框
                    coords_list = small_list if small_list else large_list
                else:  # 动态区域（字幕）
                    # 字幕必须要红色大框来确定阈值扫描范围，合并保留
                    coords_list = large_list + small_list

                if coords_list:
                    poses[pose_key] = coords_list
                    print(f"  ✓ 加载 {type_name} (key={pose_key}): {len(coords_list)} 个有效形状")
            else:
                print(f"  ℹ 跳过未知区域类型: {region_key} / {type_name}")

        return poses

    except Exception as e:
        print(f"✗ 加载Mask JSON失败: {e}")
        import traceback
        traceback.print_exc()
        return None