import json
import os


def load_poses_from_json(json_path):
    """
    从前端生成的JSON文件中读取Mask坐标并转换为poses格式

    JSON结构:
    {
        "regions": {
            "logo": { "typeName": "台标", "largeBoxes": [...], "smallBoxes": [...] },
            "subtitle": { ... },
            "title": { ... }
        }
    }

    转换目标 poses 格式:
    {
        '1': [[x1, y1, x2, y2], ...],  # 台标 (logo)
        '2': [[x1, y1, x2, y2], ...],  # 字幕 (subtitle)
        '3': [[x1, y1, x2, y2], ...]   # 剧名 (title)
    }
    """
    if not json_path or not os.path.exists(json_path):
        print(f"✗ 错误: Mask JSON文件未找到: {json_path}")
        return None

    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        poses = {}

        # 映射关系: typeName -> key
        # 务必确保前端 typeName 和这里匹配
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
                # 尝试直接解析根字典（如果是旧格式）
                regions = data

        print(f"✓ 开始解析Mask JSON，找到区域: {list(regions.keys())}")

        for region_key, region_data in regions.items():
            # 获取类型名称，如果没找到typeName，尝试用key映射
            type_name = region_data.get('typeName', region_key)

            # 确定 pose_key (1, 2, or 3)
            pose_key = None
            if type_name in type_mapping:
                pose_key = type_mapping[type_name]
            elif region_key in type_mapping:
                pose_key = type_mapping[region_key]

            if pose_key:
                coords_list = []

                # 1. 处理 largeBoxes
                large_boxes = region_data.get('largeBoxes', [])
                for box in large_boxes:
                    try:
                        # 确保转换为浮点数再取整
                        x = int(float(box['x']))
                        y = int(float(box['y']))
                        w = int(float(box['width']))
                        h = int(float(box['height']))
                        coords_list.append([x, y, x + w, y + h])
                    except (ValueError, KeyError) as e:
                        print(f"  ⚠ 解析 {type_name} largeBox 出错: {e}")

                # 2. 处理 smallBoxes
                small_boxes = region_data.get('smallBoxes', [])
                for box in small_boxes:
                    try:
                        x = int(float(box['x']))
                        y = int(float(box['y']))
                        w = int(float(box['width']))
                        h = int(float(box['height']))
                        coords_list.append([x, y, x + w, y + h])
                    except (ValueError, KeyError) as e:
                        print(f"  ⚠ 解析 {type_name} smallBox 出错: {e}")

                if coords_list:
                    poses[pose_key] = coords_list
                    print(f"  ✓ 加载 {type_name} (key={pose_key}): {len(coords_list)} 个框")
            else:
                print(f"  ℹ 跳过未知区域类型: {region_key} / {type_name}")

        return poses

    except Exception as e:
        print(f"✗ 加载Mask JSON失败: {e}")
        import traceback
        traceback.print_exc()
        return None