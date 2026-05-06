import sys
import os
import argparse
import cv2
import numpy as np
import re
import math  # 【核心新增】用于向上/向下取整
from concurrent.futures import ThreadPoolExecutor
from functools import partial

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

try:
    from mask_loader import load_poses_from_json
except ImportError:
    def load_poses_from_json(path):
        return {}, {}


def process_single_frame_mask(img_path, local_poses, output_dir, time_ranges, fps=25.0, threshold=135):
    fname = os.path.basename(img_path)
    save_path = os.path.join(output_dir, fname)

    if os.path.exists(save_path) and os.path.getsize(save_path) > 0: return

    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None: return

    mask = np.zeros_like(img)
    has_content = False
    h_m, w_m = mask.shape

    m_idx = re.search(r'\d+', fname)
    frame_idx = int(m_idx.group()) if m_idx else -1

    for key, coords in local_poses.items():
        # 【完美支持多段分散时间段判定】
        if key in time_ranges and time_ranges[key]:
            ranges = time_ranges[key]
            is_in_range = False
            for r in ranges:
                # r 是一个子列表: [start_sec, end_sec]
                if math.floor(r[0] * fps) <= frame_idx <= math.ceil(r[1] * fps):
                    is_in_range = True
                    break
            if not is_in_range:
                continue

        is_solid = (key != '2')

        for item in coords:
            is_poly = isinstance(item, list) and isinstance(item[0], (list, tuple))
            is_rect = len(item) == 4 and isinstance(item[0], (int, float))
            if not (is_poly or is_rect): continue

            if is_solid:
                if is_poly:
                    cv2.fillPoly(mask, [np.array(item, dtype=np.int32)], 255)
                    has_content = True
                else:
                    x1, y1, x2, y2 = map(int, item)
                    cv2.rectangle(mask, (max(0, x1), max(0, y1)), (min(w_m, x2), min(h_m, y2)), 255, -1)
                    has_content = True
            else:
                if is_poly:
                    x, y, w, h = cv2.boundingRect(np.array(item, dtype=np.int32))
                    x1, y1, x2, y2 = x, y, x + w, y + h
                else:
                    x1, y1, x2, y2 = map(int, item)
                x1, y1, x2, y2 = max(0, x1), max(0, y1), min(w_m, x2), min(h_m, y2)
                if x2 > x1 and y2 > y1:
                    roi = img[y1:y2, x1:x2]
                    _, bin_roi = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)
                    mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], bin_roi)
                    has_content = True

    if has_content:
        mask = cv2.dilate(mask, np.ones((5, 5), np.uint8), iterations=3)

    cv2.imwrite(save_path, mask)


def generate_local_masks(frame_dir, output_dir, original_poses, crop_coords, time_ranges=None, fps=25.0):
    if time_ranges is None: time_ranges = {}

    os.makedirs(output_dir, exist_ok=True)
    frames = sorted([f for f in os.listdir(frame_dir) if f.lower().endswith(('.png', '.jpg'))])
    if not frames: return

    cx1, cy1, cx2, cy2 = crop_coords
    crop_w, crop_h = cx2 - cx1, cy2 - cy1

    local_poses = {}
    for key, coords_list in original_poses.items():
        local_list = []
        for item in coords_list:
            if len(item) == 4 and isinstance(item[0], (int, float)):
                lx1, ly1 = max(0, item[0] - cx1), max(0, item[1] - cy1)
                lx2, ly2 = min(crop_w, item[2] - cx1), min(crop_h, item[3] - cy1)
                if lx2 > lx1 and ly2 > ly1: local_list.append([lx1, ly1, lx2, ly2])
            elif isinstance(item, list) and isinstance(item[0], (list, tuple)):
                poly_local = [[max(0, min(crop_w, pt[0] - cx1)), max(0, min(crop_h, pt[1] - cy1))] for pt in item]
                if len(poly_local) >= 3: local_list.append(poly_local)
        if local_list: local_poses[key] = local_list

    h, w = crop_h, crop_w
    static_mask = np.zeros((h, w), dtype=np.uint8)
    has_static_content = False

    for key, coords in local_poses.items():
        if key != '2' and key not in time_ranges:
            for item in coords:
                if len(item) == 4 and isinstance(item[0], (int, float)):
                    cv2.rectangle(static_mask, (int(item[0]), int(item[1])), (int(item[2]), int(item[3])), 255, -1)
                    has_static_content = True
                elif isinstance(item, list) and isinstance(item[0], (list, tuple)):
                    cv2.fillPoly(static_mask, [np.array(item, dtype=np.int32)], 255)
                    has_static_content = True

    if has_static_content:
        static_mask = cv2.dilate(static_mask, np.ones((5, 5), np.uint8), iterations=3)
    cv2.imwrite(os.path.join(output_dir, "static_mask.png"), static_mask)

    has_dynamic = any(k == '2' or k in time_ranges for k in local_poses)
    if not has_dynamic: return

    process_func = partial(
        process_single_frame_mask,
        local_poses=local_poses,
        output_dir=output_dir,
        time_ranges=time_ranges,
        fps=fps
    )

    frame_paths = [os.path.join(frame_dir, f) for f in frames]
    with ThreadPoolExecutor(max_workers=8) as executor:
        list(executor.map(process_func, frame_paths))