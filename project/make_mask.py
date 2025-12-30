import argparse
import json
import cv2
import numpy as np
import os
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial

# 尝试导入 mask_loader，如果单独运行且没有该文件可能会报错
try:
    from mask_loader import load_poses_from_json
except ImportError:
    # 定义一个简单的 fallback 或者提示用户
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
    # 使用灰度读取即可
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return

    # 创建全黑 Mask
    mask = np.zeros_like(img)

    has_content = False

    for key, coords in local_poses.items():
        # 判断是否为静态区域 (如台标 '1', 标题 '3')
        is_static = (key in static_keys)

        for box in coords:
            x1, y1, x2, y2 = map(int, box)

            # 边界检查，防止崩溃
            h, w = mask.shape
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            if x2 <= x1 or y2 <= y1:
                continue

            if is_static:
                # 【静态区域】：直接画白色实心矩形
                cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
                has_content = True
            else:
                # 【动态区域】：例如字幕(key='2')，根据像素亮度生成Mask
                # 截取 ROI
                roi = img[y1:y2, x1:x2]

                # 二值化处理：提取高亮文字
                # 这里默认文字是白色的，阈值 135
                _, bin_roi = cv2.threshold(roi, threshold, 255, cv2.THRESH_BINARY)

                # 将二值化后的 ROI 放入 Mask 对应位置
                # 使用 maximum 确保不会覆盖掉已有的 mask 区域
                mask[y1:y2, x1:x2] = np.maximum(mask[y1:y2, x1:x2], bin_roi)
                has_content = True

    # 如果生成了 Mask 内容，进行膨胀操作以确保覆盖边缘
    if has_content:
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.dilate(mask, kernel, iterations=2)

    # 保存 Mask
    cv2.imwrite(save_path, mask)


def generate_local_masks(frame_dir, output_dir, original_poses, crop_coords, static_keys=['1', '3']):
    """
    主控函数：生成适应裁切区域的 Mask。

    Args:
        frame_dir: 裁切后的图片文件夹
        output_dir: Mask 输出文件夹
        original_poses: 原始的全局坐标 {'1': [[x1,y1,x2,y2]...]}
        crop_coords: 全局裁切框 [gx1, gy1, gx2, gy2]
        static_keys: 哪些 key 被视为静态区域 (默认 '1'台标, '3'标题)
    """
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有帧文件
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
        for box in coords_list:
            gx1, gy1, gx2, gy2 = box

            # 减去裁切偏移量
            lx1 = gx1 - cx1
            ly1 = gy1 - cy1
            lx2 = gx2 - cx1
            ly2 = gy2 - cy1

            # 裁剪坐标使其不超出小图边界 (Clip)
            lx1 = max(0, lx1)
            ly1 = max(0, ly1)
            lx2 = min(crop_w, lx2)
            ly2 = min(crop_h, ly2)

            # 只有有效的框才保留
            if lx2 > lx1 and ly2 > ly1:
                local_list.append([lx1, ly1, lx2, ly2])

        if local_list:
            local_poses[key] = local_list

    # 2. 生成静态基准 Mask (static_mask.png)
    # 无论是否有动态内容，生成一张只包含静态框的 Mask 是很有用的备份
    h, w = crop_h, crop_w
    # 如果 crop 尺寸有问题，尝试读取一张图获取
    if h <= 0 or w <= 0:
        sample = cv2.imread(os.path.join(frame_dir, frames[0]), cv2.IMREAD_GRAYSCALE)
        h, w = sample.shape

    static_mask = np.zeros((h, w), dtype=np.uint8)
    has_static_content = False

    for key, coords in local_poses.items():
        # 如果是静态key，或者是某些没有标记但在列表里的
        if key in static_keys:
            for box in coords:
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(static_mask, (x1, y1), (x2, y2), 255, -1)
                has_static_content = True

    if has_static_content:
        kernel = np.ones((3, 3), np.uint8)
        static_mask = cv2.dilate(static_mask, kernel, iterations=2)

    cv2.imwrite(os.path.join(output_dir, "static_mask.png"), static_mask)
    print("[MakeMask] 已生成静态基准 Mask: static_mask.png")

    # 3. 检查是否需要生成动态序列 Mask
    # 如果包含除 static_keys 以外的 key (比如 '2')，则需要逐帧处理
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

    # 使用 partial 固定参数
    process_func = partial(
        process_single_frame_mask,
        local_poses=local_poses,
        output_dir=output_dir,
        static_keys=static_keys
    )

    frame_paths = [os.path.join(frame_dir, f) for f in frames]

    # 线程池并行处理
    with ThreadPoolExecutor(max_workers=8) as executor:
        # 使用 list 触发执行
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