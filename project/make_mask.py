import sys
import os

# =================================================================
# 【路径修复】确保脚本能引用上级目录（根目录）的模块 (core, model, utils)
# =================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../ProPainter/project
root_dir = os.path.dirname(current_dir)  # .../ProPainter
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# =================================================================

import argparse
import json
import cv2
import numpy as np
from concurrent.futures import ThreadPoolExecutor
from functools import partial

try:
    from mask_loader import load_poses_from_json
except ImportError:
    def load_poses_from_json(path):
        print("错误: 缺少 mask_loader.py")
        return {}


def process_single_frame_mask(img_path, local_poses, output_dir, static_keys, threshold=135):
    """
    处理单张图片的 Mask 生成任务
    """
    fname = os.path.basename(img_path)
    save_path = os.path.join(output_dir, fname)

    # 如果文件已存在，跳过（支持断点续传）
    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        return

    # 读取原图用于动态计算 (如字幕)
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    # 创建全黑 Mask
    mask = np.zeros_like(img)
    has_content = False
    h_m, w_m = mask.shape

    for key, coords in local_poses.items():
        is_static = (key in static_keys)

        for item in coords:
            # 判断数据类型是多边形(魔法棒)还是矩形(普通画框)
            is_poly = isinstance(item, list) and isinstance(item[0], (list, tuple))
            is_rect = len(item) == 4 and isinstance(item[0], (int, float))

            if not (is_poly or is_rect):
                continue

            # ==========================================================
            # 【核心业务逻辑分类处理】
            # ==========================================================
            if is_static:
                # 1. 静态区域 (台标、剧名): 直接画成纯白色的实心块
                if is_poly:
                    pts = np.array(item, dtype=np.int32)
                    cv2.fillPoly(mask, [pts], 255)
                    has_content = True
                else:
                    x1, y1, x2, y2 = map(int, item)
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w_m, x2), min(h_m, y2)
                    if x2 > x1 and y2 > y1:
                        cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                        has_content = True
            else:
                # 2. 动态区域 (字幕): 【强制】使用原有的亮度阈值处理
                # 如果用户不小心对字幕用了魔法棒(多边形)，这里会自动提取其外接矩形，确保阈值处理照常运行！
                if is_poly:
                    pts = np.array(item, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(pts)
                    x1, y1, x2, y2 = x, y, x + w, y + h
                else:
                    x1, y1, x2, y2 = map(int, item)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_m, x2), min(h_m, y2)

                if x2 > x1 and y2 > y1:
                    # 沿用之前的阈值二值化提取亮度文字逻辑
                    roi = img[y1:y2, x1:x2]
                    _, bin_roi = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)
                    mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], bin_roi)
                    has_content = True

    # ==========================================================
    # 强力膨胀：向外扩充，消除半透明边缘鬼影
    # ==========================================================
    if has_content:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)

    cv2.imwrite(save_path, mask)


def generate_local_masks(frame_dir, output_dir, original_poses, crop_coords, static_keys=['1', '3']):
    """
    主控函数：生成适应裁切区域的 Mask。
    """
    os.makedirs(output_dir, exist_ok=True)

    frames = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if not frames:
        print("[MakeMask] 错误: 帧文件夹为空")
        return

    cx1, cy1, cx2, cy2 = crop_coords
    crop_w = cx2 - cx1
    crop_h = cy2 - cy1

    print(f"[MakeMask] 计算局部坐标 (裁切偏移: -{cx1}, -{cy1})")

    # 1. 坐标变换：Global -> Local
    local_poses = {}
    for key, coords_list in original_poses.items():
        local_list = []
        for item in coords_list:
            # 兼容旧矩形框
            if len(item) == 4 and isinstance(item[0], (int, float)):
                gx1, gy1, gx2, gy2 = item
                lx1, ly1 = max(0, gx1 - cx1), max(0, gy1 - cy1)
                lx2, ly2 = min(crop_w, gx2 - cx1), min(crop_h, gy2 - cy1)
                if lx2 > lx1 and ly2 > ly1:
                    local_list.append([lx1, ly1, lx2, ly2])

            # 兼容多边形
            elif isinstance(item, list) and isinstance(item[0], (list, tuple)):
                poly_local = []
                for pt in item:
                    lx = max(0, min(crop_w, pt[0] - cx1))
                    ly = max(0, min(crop_h, pt[1] - cy1))
                    poly_local.append([lx, ly])
                if len(poly_local) >= 3:
                    local_list.append(poly_local)

        if local_list:
            local_poses[key] = local_list

    # 2. 生成静态基准 Mask (static_mask.png)
    h, w = crop_h, crop_w
    if h <= 0 or w <= 0:
        sample = cv2.imread(os.path.join(frame_dir, frames[0]), cv2.IMREAD_GRAYSCALE)
        h, w = sample.shape

    static_mask = np.zeros((h, w), dtype=np.uint8)
    has_static_content = False

    # 注意：这里有 key in static_keys 拦截，所以动态字幕(key=2)绝对不会生成静态 Mask
    for key, coords in local_poses.items():
        if key in static_keys:
            for item in coords:
                if len(item) == 4 and isinstance(item[0], (int, float)):
                    x1, y1, x2, y2 = map(int, item)
                    cv2.rectangle(static_mask, (x1, y1), (x2, y2), 255, -1)
                    has_static_content = True
                elif isinstance(item, list) and isinstance(item[0], (list, tuple)):
                    pts = np.array(item, dtype=np.int32)
                    cv2.fillPoly(static_mask, [pts], 255)
                    has_static_content = True

    if has_static_content:
        kernel = np.ones((5, 5), np.uint8)
        static_mask = cv2.dilate(static_mask, kernel, iterations=3)

    cv2.imwrite(os.path.join(output_dir, "static_mask.png"), static_mask)
    print("[MakeMask] 已生成静态基准 Mask: static_mask.png")

    # 3. 检查是否需要生成动态序列 Mask
    has_dynamic = False
    for k in local_poses:
        if k not in static_keys:
            has_dynamic = True
            break

    if not has_dynamic:
        print("[MakeMask] 仅包含静态区域，跳过序列生成。")
        return

    # 4. 并行生成动态序列
    print(f"[MakeMask] 检测到动态区域，开始生成 {len(frames)} 张 Mask...")
    process_func = partial(
        process_single_frame_mask,
        local_poses=local_poses,
        output_dir=output_dir,
        static_keys=static_keys
    )

    frame_paths = [os.path.join(frame_dir, f) for f in frames]

    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(process_func, frame_paths))

    print("[MakeMask] 动态 Mask 序列生成完成。")


if __name__ == '__main__':
    # 命令行入口，用于调试
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', required=True, help='图片文件夹')
    parser.add_argument('--output', required=True, help='Mask输出文件夹')
    parser.add_argument('--json', required=True, help='原始Mask JSON')
    parser.add_argument('--crop', required=True, help='裁切坐标 x1,y1,x2,y2 (逗号分隔)')

    args = parser.parse_args()

    crop_coords = list(map(int, args.crop.split(',')))
    poses = load_poses_from_json(args.json)

    generate_local_masks(args.frames, args.output, poses, crop_coords)