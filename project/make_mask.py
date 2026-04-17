import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

import argparse
import cv2
import numpy as np
import re
from concurrent.futures import ThreadPoolExecutor
from functools import partial

try:
    from mask_loader import load_poses_from_json
except ImportError:
    def load_poses_from_json(path):
        print("错误: 缺少 mask_loader.py")
        return {}, {}


def process_single_frame_mask(img_path, local_poses, output_dir, time_ranges, threshold=135):
    fname = os.path.basename(img_path)
    save_path = os.path.join(output_dir, fname)

    if os.path.exists(save_path) and os.path.getsize(save_path) > 0:
        return

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    mask = np.zeros_like(img)
    has_content = False
    h_m, w_m = mask.shape

    # 提取当前帧编号，用于时间匹配
    m_idx = re.search(r'\d+', fname)
    frame_idx = int(m_idx.group()) if m_idx else -1

    for key, coords in local_poses.items():
        # 【核心过滤】如果该区域有时间限制，且当前帧不在范围内，直接跳过！
        if key in time_ranges:
            start_f, end_f = time_ranges[key]
            if not (start_f <= frame_idx <= end_f):
                continue

        is_solid = (key != '2')  # 除了字幕以外全都是实心块填充

        for item in coords:
            is_poly = isinstance(item, list) and isinstance(item[0], (list, tuple))
            is_rect = len(item) == 4 and isinstance(item[0], (int, float))

            if not (is_poly or is_rect):
                continue

            if is_solid:
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
                if is_poly:
                    pts = np.array(item, dtype=np.int32)
                    x, y, w, h = cv2.boundingRect(pts)
                    x1, y1, x2, y2 = x, y, x + w, y + h
                else:
                    x1, y1, x2, y2 = map(int, item)

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(w_m, x2), min(h_m, y2)

                if x2 > x1 and y2 > y1:
                    roi = img[y1:y2, x1:x2]
                    _, bin_roi = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)
                    mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], bin_roi)
                    has_content = True

    if has_content:
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=3)

    cv2.imwrite(save_path, mask)


def generate_local_masks(frame_dir, output_dir, original_poses, crop_coords, time_ranges=None):
    if time_ranges is None:
        time_ranges = {}

    os.makedirs(output_dir, exist_ok=True)
    frames = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith(('.png', '.jpg'))])
    if not frames:
        return

    cx1, cy1, cx2, cy2 = crop_coords
    crop_w, crop_h = cx2 - cx1, cy2 - cy1

    local_poses = {}
    for key, coords_list in original_poses.items():
        local_list = []
        for item in coords_list:
            if len(item) == 4 and isinstance(item[0], (int, float)):
                lx1, ly1 = max(0, item[0] - cx1), max(0, item[1] - cy1)
                lx2, ly2 = min(crop_w, item[2] - cx1), min(crop_h, item[3] - cy1)
                if lx2 > lx1 and ly2 > ly1:
                    local_list.append([lx1, ly1, lx2, ly2])
            elif isinstance(item, list) and isinstance(item[0], (list, tuple)):
                poly_local = [[max(0, min(crop_w, pt[0] - cx1)), max(0, min(crop_h, pt[1] - cy1))] for pt in item]
                if len(poly_local) >= 3:
                    local_list.append(poly_local)
        if local_list:
            local_poses[key] = local_list

    # ======= 制作绝对静态 Mask (无时间限制) =======
    h, w = crop_h, crop_w
    static_mask = np.zeros((h, w), dtype=np.uint8)
    has_static_content = False

    for key, coords in local_poses.items():
        # 【排除法则】如果它是字幕(2) 或 有时间限制区域，绝对不能画进静态底图
        if key != '2' and key not in time_ranges:
            for item in coords:
                if len(item) == 4 and isinstance(item[0], (int, float)):
                    cv2.rectangle(static_mask, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), 255, -1)
                    has_static_content = True
                elif isinstance(item, list) and isinstance(item[0], (list, tuple)):
                    pts = np.array(item, dtype=np.int32)
                    cv2.fillPoly(static_mask, [pts], 255)
                    has_static_content = True

    if has_static_content:
        kernel = np.ones((5, 5), np.uint8)
        static_mask = cv2.dilate(static_mask, kernel, iterations=3)
    cv2.imwrite(os.path.join(output_dir, "static_mask.png"), static_mask)

    # ======= 判断是否需要生成动态序列 =======
    has_dynamic = any(k == '2' or k in time_ranges for k in local_poses)

    if not has_dynamic:
        return

    process_func = partial(
        process_single_frame_mask,
        local_poses=local_poses,
        output_dir=output_dir,
        time_ranges=time_ranges
    )

    frame_paths = [os.path.join(frame_dir, f) for f in frames]
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(process_func, frame_paths))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--frames', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--json', required=True)
    parser.add_argument('--crop', required=True)
    args = parser.parse_args()

    crop_coords = list(map(int, args.crop.split(',')))
    poses, trs = load_poses_from_json(args.json)
    generate_local_masks(args.frames, args.output, poses, crop_coords, time_ranges=trs)