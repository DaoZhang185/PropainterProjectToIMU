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

import argparse
import json
import cv2
import numpy as np
import glob


def detect_scenes_from_folder(folder_path, threshold=10.0, min_scene_frames=5):
    """
    读取文件夹中的图片序列进行场景检测。
    不再依赖 ffmpeg，直接比较图片像素差异。

    Args:
        folder_path: 包含帧图片的文件夹路径
        threshold: 差异阈值 (默认10.0，根据灰度均值差异)
        min_scene_frames: 最小场景帧数，防止闪烁误判

    Returns:
        List[int]: 场景切换的帧索引列表 (例如 [0, 150, 300])
    """
    # 1. 获取并排序文件
    # 支持多种常见图片格式
    extensions = ['*.png', '*.jpg', '*.jpeg', '*.bmp']
    files = []
    for ext in extensions:
        files.extend(glob.glob(os.path.join(folder_path, ext)))
        # 兼容大小写
        files.extend(glob.glob(os.path.join(folder_path, ext.upper())))

    files = sorted(list(set(files)))  # 去重并排序

    if not files:
        print(f"[SceneDetector] 警告: 文件夹 {folder_path} 为空或无图片")
        return [0]

    print(f"[SceneDetector] 正在扫描 {len(files)} 张图片 (阈值={threshold})...")

    scene_indices = [0]  # 第0帧永远是起始点
    prev_frame = None

    # 降采样大小，加快计算速度 (128x128 足以检测场景变化)
    process_size = (128, 128)

    # 遍历所有图片
    for i, file_path in enumerate(files):
        # 以灰度模式读取，减少计算量
        frame = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)

        if frame is None:
            continue

        # 缩小图片
        frame_resized = cv2.resize(frame, process_size)

        if prev_frame is not None:
            # 计算两帧之间的绝对差异
            diff = cv2.absdiff(prev_frame, frame_resized)
            # 计算差异均值
            score = np.mean(diff)

            # 如果差异超过阈值，且距离上一次切换有一定间隔
            if score > threshold:
                if (i - scene_indices[-1]) > min_scene_frames:
                    scene_indices.append(i)
                    # 可选：打印检测到的点
                    # print(f"  -> Frame {i}: 场景切换 (Score: {score:.2f})")

        prev_frame = frame_resized

        # 简单的进度打印
        if i > 0 and i % 500 == 0:
            sys.stdout.write(f"\r  扫描进度: {i}/{len(files)}")
            sys.stdout.flush()

    print(f"\n[SceneDetector] 检测结束，共发现 {len(scene_indices)} 个场景片段")
    return scene_indices


if __name__ == "__main__":
    # 命令行入口，用于单独测试
    parser = argparse.ArgumentParser(description="基于图片文件夹的场景检测工具")
    parser.add_argument('--input', required=True, help='输入图片文件夹路径')
    parser.add_argument('--output', required=True, help='输出JSON文件路径')
    parser.add_argument('--threshold', type=float, default=10.0, help='场景切换检测阈值')

    args = parser.parse_args()

    indices = detect_scenes_from_folder(args.input, args.threshold)

    # 确保目录存在
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)

    with open(args.output, 'w') as f:
        json.dump(indices, f)

    print(f"结果已保存至: {args.output}")