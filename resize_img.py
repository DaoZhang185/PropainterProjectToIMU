import argparse
import cv2
import os
import glob
from concurrent.futures import ThreadPoolExecutor
import time


def resize_single_image(args):
    """处理单张图像的包装函数"""
    file_path, output_folder, scale = args
    # 支持的图片格式
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    ext = os.path.splitext(file_path)[1].lower()
    if ext not in valid_exts:
        return None

    # 读取图片
    img = cv2.imread(file_path)
    if img is None:
        return None

    # 计算新尺寸
    new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))

    # 缩放图片
    resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

    # 构造输出路径
    filename = os.path.basename(file_path)
    output_path = os.path.join(output_folder, filename)

    # 保存图片
    if cv2.imwrite(output_path, resized):
        return filename
    return None


def resize_images_parallel(input_folder, output_folder, scale=0.25, max_workers=4):
    """
    并行缩放图片
    """
    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有图片文件
    file_paths = [f for f in glob.glob(os.path.join(input_folder, '*'))
                  if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}]

    if not file_paths:
        print("没有找到支持的图片文件")
        return

    print(f"开始并行缩放 {len(file_paths)} 张图片...")
    start_time = time.time()

    # 准备任务
    tasks = [(file_path, output_folder, scale) for file_path in file_paths]

    # 使用线程池并行处理
    completed_count = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(resize_single_image, task) for task in tasks]

        for future in futures:
            result = future.result()
            if result is not None:
                completed_count += 1
                if completed_count % 50 == 0:
                    print(f"进度: {completed_count}/{len(file_paths)}")

    end_time = time.time()
    print(f"图片缩放完成！成功处理 {completed_count}/{len(file_paths)} 张图片")
    print(f"总耗时: {end_time - start_time:.2f} 秒")


def resize_images(input_folder, output_folder, scale=0.25):
    """
    将输入文件夹中的图片缩放后保存到输出文件夹
    """
    # 根据文件数量决定使用并行还是串行
    file_paths = [f for f in glob.glob(os.path.join(input_folder, '*'))
                  if os.path.splitext(f)[1].lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}]

    if len(file_paths) > 20:
        # 文件数量多，使用并行处理
        resize_images_parallel(input_folder, output_folder, scale)
    else:
        # 文件数量少，使用原来的串行处理
        resize_images_original(input_folder, output_folder, scale)


def resize_images_original(input_folder, output_folder, scale=0.25):
    """
    原来的串行处理函数
    """
    # 支持的图片格式
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

    # 创建输出文件夹
    os.makedirs(output_folder, exist_ok=True)

    # 获取所有图片文件
    file_paths = [f for f in glob.glob(os.path.join(input_folder, '*'))
                  if os.path.splitext(f)[1].lower() in valid_exts]

    print(f"开始处理 {len(file_paths)} 张图片...")
    start_time = time.time()

    processed_count = 0
    for file_path in file_paths:
        # 读取图片
        img = cv2.imread(file_path)
        if img is None:
            continue

        # 计算新尺寸
        new_size = (int(img.shape[1] * scale), int(img.shape[0] * scale))

        # 缩放图片
        resized = cv2.resize(img, new_size, interpolation=cv2.INTER_AREA)

        # 构造输出路径
        filename = os.path.basename(file_path)
        output_path = os.path.join(output_folder, filename)

        # 保存图片
        if cv2.imwrite(output_path, resized):
            processed_count += 1
            if processed_count % 50 == 0:
                print(f"进度: {processed_count}/{len(file_paths)}")

    end_time = time.time()
    print(f"图片缩放完成！成功处理 {processed_count} 张图片")
    print(f"总耗时: {end_time - start_time:.2f} 秒")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='缩放图片并保存')
    parser.add_argument('--input_dir', type=str, required=True, help='输入文件夹路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出文件夹路径')
    parser.add_argument('--scale', type=float, default=0.25, help='缩放比例（默认0.25）')
    parser.add_argument('--max-workers', type=int, default=4, help='并行处理的最大线程数，默认4')
    args = parser.parse_args()

    # 使用解析的参数调用函数
    if hasattr(args, 'max_workers'):
        resize_images_parallel(args.input_dir, args.output_dir, args.scale, args.max_workers)
    else:
        resize_images(args.input_dir, args.output_dir, args.scale)