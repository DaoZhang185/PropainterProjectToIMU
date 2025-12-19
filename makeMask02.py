import argparse
import json
import cv2
import numpy as np
import os
from concurrent.futures import ThreadPoolExecutor
import time
from functools import lru_cache
import sys


def process_image(image):
    """基于直方图峰值生成二值化图像"""
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
    max_value_index = np.argmax(histogram)
    left_index = max(0, max_value_index - 30)
    right_index = min(255, max_value_index + 30)
    processed_image = np.zeros_like(image)
    processed_image[(image >= left_index) & (image <= right_index)] = 255
    return processed_image


def make_mask(src_img, poses):
    """根据多个子区域生成掩码"""
    x1, y1, x2, y2 = poses[0]
    cropped_image = src_img[y1:y2, x1:x2]
    result = np.zeros_like(cropped_image)
    for i in range(1, len(poses)):
        x1_sub, y1_sub, x2_sub, y2_sub = poses[i]
        temp = src_img[y1_sub:y2_sub, x1_sub:x2_sub]
        min_val = temp.min()
        max_val = temp.max()
        result[(cropped_image >= min_val + 40) & (cropped_image <= max_val - 40)] = 255
    return result


# 添加图像缓存
@lru_cache(maxsize=100)
def load_image_cached(image_path):
    """带缓存的图像加载"""
    return cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)


def process_and_place_image_with_threshold(image_path, poses, threshold):
    """使用指定阈值生成掩码并放置到原图位置"""
    x1, y1, x2, y2 = poses[0]
    original_image = load_image_cached(image_path)
    if original_image is None:
        raise ValueError(f"无法加载图像: {image_path}")
    if x1 < 0 or y1 < 0 or x2 > original_image.shape[1] or y2 > original_image.shape[0]:
        raise ValueError(f"无效坐标: {x1},{y1},{x2},{y2}")

    cropped_image = original_image[y1:y2, x1:x2]
    _, binary_image = cv2.threshold(cropped_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image, (x1, y1, x2, y2)


def process_and_place_image(img_path, pos, j):
    # 读取原始图像
    x1, y1, x2, y2 = pos[0]
    original_image = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise ValueError("Image not found or unable to load")

    # 检查坐标是否有效
    if x1 < 0 or y1 < 0 or x2 > original_image.shape[1] or y2 > original_image.shape[0]:
        raise ValueError("Invalid coordinates")

    # 截取指定区域的图像
    if int(j) == 2:  # 需要变化的框 (字幕)
        cropped_image = original_image[y1:y2, x1:x2]

        # 二值化处理
        _, binary_image = cv2.threshold(cropped_image, 135, 255, cv2.THRESH_BINARY)
    else:
        binary_image = make_mask(original_image, pos)

    # 定义膨胀操作的核
    kernel = np.ones((3, 3), np.uint8)

    # 进行膨胀处理
    dilated_image = cv2.dilate(binary_image, kernel, iterations=2)

    # 创建一个与原始图像相同大小的全0图片
    result_image = np.zeros_like(original_image)

    # 将处理后的图像放入到结果图像的指定位置
    result_image[y1:y2, x1:x2] = dilated_image

    return dilated_image, (x1, y1, x2, y2)


def make_frame1(img_path, poses, threshold, color_tolerance=20, manual_threshold_region2=None):
    """生成第一帧的掩码作为基准"""
    result = None
    original_image = load_image_cached(img_path)
    if original_image is None:
        raise ValueError(f"无法加载第一帧图像: {img_path}")

    # 关键修改：明确遍历 1(台标), 2(字幕), 3(标题)
    # 只有当 poses 中存在该 key 时才处理
    target_keys = ['1', '2', '3']

    for pos_key in target_keys:
        if pos_key not in poses:
            continue

        pos = poses[pos_key]
        j = int(pos_key)

        # 根据区域类型选择处理方法
        if j == 1:
            bin_img, (x1, y1, x2, y2) = process_and_place_image_with_threshold(img_path, pos, threshold)
        elif j == 2:
            # 字幕区域
            bin_img, (x1, y1, x2, y2) = process_and_place_image(img_path, pos, j)
        elif j == 3:
            # 标题区域
            bin_img, (x1, y1, x2, y2) = process_and_place_image(img_path, pos, j)
        else:
            # 其他区域
            bin_img, (x1, y1, x2, y2) = process_and_place_image(img_path, pos, j)

        temp_result = np.zeros_like(original_image)
        temp_result[y1:y2, x1:x2] = bin_img

        if result is None:
            result = temp_result
        else:
            result = cv2.bitwise_or(result, temp_result)

    # 如果没有任何区域被选中，返回全黑图像
    if result is None:
        result = np.zeros_like(original_image)

    return result


def process_single_image(args):
    """处理单张图像的包装函数，用于并行处理"""
    i, img_path, poses, save_path, frame1_mask, static_boxes, threshold, kernel_size, iterations, color_tolerance, manual_threshold_region2 = args

    original_image = load_image_cached(img_path)
    if original_image is None:
        print(f"无法加载图像: {img_path}")
        return None

    result = np.zeros_like(original_image)
    kernel = np.ones((kernel_size, kernel_size), np.uint8)

    # 关键修改：明确遍历所有可能的 key
    target_keys = ['1', '2', '3']

    for pos_key in target_keys:
        if pos_key not in poses:
            continue

        pos = poses[pos_key]
        j = int(pos_key)
        x1, y1, x2, y2 = pos[0]

        if j in static_boxes:
            # 静态框直接复用第一帧
            temp_result = np.zeros_like(original_image)
            temp_result[y1:y2, x1:x2] = frame1_mask[y1:y2, x1:x2]
        else:
            # 动态框重新计算
            if j == 1:
                bin_img, _ = process_and_place_image_with_threshold(img_path, pos, threshold)
            elif j == 2:
                bin_img, _ = process_and_place_image(img_path, pos, j)
            elif j == 3:
                bin_img, (x1, y1, x2, y2) = process_and_place_image(img_path, pos, j)
            else:
                bin_img, _ = process_and_place_image(img_path, pos, j)

            dilated_img = cv2.dilate(bin_img, kernel, iterations=iterations)
            temp_result = np.zeros_like(original_image)
            temp_result[y1:y2, x1:x2] = dilated_img

        result = cv2.bitwise_or(result, temp_result)

    save_path_img = os.path.join(save_path, os.path.basename(img_path))
    cv2.imwrite(save_path_img, result)

    return i


def process_and_place_images_parallel(img_paths, poses, save_path, frame1_mask, static_boxes, threshold, kernel_size,
                                      iterations, color_tolerance=20, max_workers=4, manual_threshold_region2=None):
    """批量处理图像并保存结果 - 并行版本"""
    os.makedirs(save_path, exist_ok=True)
    print(f"开始并行批量处理 {len(img_paths)} 张图像...")

    tasks = []
    for i, img_path in enumerate(img_paths):
        task = (
            i, img_path, poses, save_path, frame1_mask, static_boxes,
            threshold, kernel_size, iterations, color_tolerance, manual_threshold_region2
        )
        tasks.append(task)

    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(process_single_image, task) for task in tasks]

        for future in futures:
            result = future.result()
            if result is not None:
                completed_count += 1
                if completed_count % 50 == 0 or completed_count == len(img_paths):
                    print(f"进度: {completed_count}/{len(img_paths)}")

    print(f"并行处理完成，成功处理 {completed_count}/{len(img_paths)} 张图像")


def process_and_place_images(img_paths, poses, save_path, frame1_mask, static_boxes, threshold, kernel_size,
                             iterations, color_tolerance=20, manual_threshold_region2=None):
    """批量处理入口"""
    process_and_place_images_parallel(
        img_paths, poses, save_path, frame1_mask, static_boxes,
        threshold, kernel_size, iterations, color_tolerance,
        manual_threshold_region2=manual_threshold_region2
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='生成掩码图片工具 - 严格JSON版')
    parser.add_argument('--input', type=str, required=True, help='输入图片文件夹路径')
    parser.add_argument('--output', type=str, required=True, help='输出掩码文件夹路径')
    parser.add_argument('--threshold', type=int, default=105, help='二值化阈值，默认105')
    parser.add_argument('--kernel_size', type=int, default=3, help='膨胀操作的核大小，默认3')
    parser.add_argument('--iterations', type=int, default=2, help='膨胀操作迭代次数，默认2')
    parser.add_argument('--static-boxes', type=str, default='1,3', help='使用第一帧掩码的框索引（逗号分隔），默认1,3')
    parser.add_argument('--color-tolerance', type=int, default=40, help='颜色容忍度，默认40')
    parser.add_argument('--max-workers', type=int, default=4, help='并行处理的最大线程数，默认4')
    parser.add_argument('--region2-threshold', type=int, default=None, help='区域2手动阈值')
    parser.add_argument('--mask_json', type=str, required=True, help='指定的Mask JSON文件路径')

    args = parser.parse_args()

    # 1. 严格加载 JSON
    poses = {}
    if args.mask_json and os.path.exists(args.mask_json):
        print(f"✓ 正在加载Mask JSON文件: {args.mask_json}")
        try:
            from mask_loader import load_poses_from_json

            poses = load_poses_from_json(args.mask_json)
        except ImportError:
            print("✗ 错误: 找不到 mask_loader.py 模块")
            sys.exit(1)
        except Exception as e:
            print(f"✗ 错误: JSON加载过程发生异常: {e}")
            sys.exit(1)

        if not poses:
            print("✗ 错误: JSON文件加载后为空，或者没有有效区域。程序退出。")
            sys.exit(1)
        else:
            print(f"✓ 成功加载区域: {list(poses.keys())}")
            for k, v in poses.items():
                print(f"  区域 {k}: {len(v)} 个框")
    else:
        print(f"✗ 错误: Mask JSON文件不存在或未指定: {args.mask_json}")
        sys.exit(1)

    # 2. 解析静态框索引
    static_boxes = set(map(int, args.static_boxes.split(',')))

    # 3. 获取输入图像
    img_paths = [os.path.join(args.input, f) for f in os.listdir(args.input)
                 if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
    img_paths.sort()

    if not img_paths:
        print(f"警告：输入文件夹 {args.input} 中没有找到图片文件")
    else:
        print(f"找到 {len(img_paths)} 张图像文件")
        start_time = time.time()

        # 4. 生成第一帧基准掩码
        print("生成第一帧掩码...")
        try:
            frame1_mask = make_frame1(
                img_paths[0], poses, args.threshold, args.color_tolerance,
                manual_threshold_region2=args.region2_threshold
            )
        except Exception as e:
            print(f"✗ 生成第一帧掩码失败: {e}")
            sys.exit(1)

        # 5. 批量处理
        process_and_place_images(
            img_paths, poses, args.output, frame1_mask,
            static_boxes, args.threshold, args.kernel_size, args.iterations,
            args.color_tolerance, manual_threshold_region2=args.region2_threshold
        )

        end_time = time.time()
        print(f"所有掩码生成完成，耗时: {end_time - start_time:.2f} 秒")
        print(f"保存在: {args.output}")