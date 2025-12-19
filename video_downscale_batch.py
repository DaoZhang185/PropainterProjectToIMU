import os
import glob
import cv2
import argparse
import shutil
import time
from concurrent.futures import ThreadPoolExecutor


def get_video_info(video_path):
    """获取视频的基本信息"""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return None

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    cap.release()
    return {'fps': fps, 'width': width, 'height': height, 'total_frames': total_frames}


def save_frame_worker(args):
    """
    单个帧保存任务，参考 resize_img.py 的缩放逻辑
    """
    frame, output_path, scale = args

    # 计算新尺寸
    new_width = int(frame.shape[1] * scale)
    new_height = int(frame.shape[0] * scale)
    new_size = (new_width, new_height)

    # 缩放图片 (参考 resize_img.py 使用 INTER_AREA 进行降采样，这是缩小图片最高质量的算法)
    resized_frame = cv2.resize(frame, new_size, interpolation=cv2.INTER_AREA)

    # 保存
    cv2.imwrite(output_path, resized_frame)
    return output_path


def extract_and_resize_frames(video_path, temp_dir, scale=0.5, max_workers=8):
    """
    读取视频 -> 缩放帧 -> 并行保存到临时目录
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"无法打开视频: {video_path}")
        return None, None

    # 获取视频信息
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"正在提取并缩放帧 (共 {total_frames} 帧)...")

    frame_idx = 0
    tasks = []

    # 使用线程池处理写入磁盘的操作，提高速度 (参考 movie2img.py 的并行处理)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # 构造输出路径，保证文件名顺序 frame_0001.jpg
            frame_name = f"frame_{frame_idx:06d}.jpg"
            output_path = os.path.join(temp_dir, frame_name)

            # 提交任务：将 resizing 和 saving 放入线程或者在主线程 resize 线程 save
            # 为了节省内存，我们在主线程 resize (快)，在子线程 save (IO 慢)
            # 或者直接把 numpy array 传给线程 (消耗内存较大但代码简单)

            # 优化方案：主线程读取，直接提交给线程池去 缩放+保存
            # 注意：如果视频分辨率很高，队列太长会爆内存，这里做一个简单的流控
            if len(tasks) > 200:
                # 等待一部分任务完成，防止内存溢出
                tasks = [t for t in tasks if not t.done()]

            task = executor.submit(save_frame_worker, (frame, output_path, scale))
            tasks.append(task)

            frame_idx += 1
            if frame_idx % 100 == 0:
                print(f"\r进度: {frame_idx}/{total_frames}", end="")

    cap.release()
    # 等待所有任务完成
    for task in tasks:
        task.result()

    print(f"\n帧处理完成，保存在: {temp_dir}")
    return fps, frame_idx


def images_to_video(temp_dir, output_path, fps):
    """
    将图片序列合成为视频
    """
    # 获取所有图片并排序
    images = sorted(glob.glob(os.path.join(temp_dir, "*.jpg")))
    if not images:
        print("没有找到图片用于合成")
        return

    # 读取第一张图获取尺寸
    first_frame = cv2.imread(images[0])
    height, width, layers = first_frame.shape
    size = (width, height)

    print(f"正在合成视频 (分辨率: {width}x{height}, FPS: {fps})...")

    # 初始化视频写入器
    # mp4v 兼容性较好，如果服务器有 h264 也可以改用 'avc1'
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, size)

    count = 0
    total = len(images)

    for img_path in images:
        img = cv2.imread(img_path)
        out.write(img)
        count += 1
        if count % 100 == 0:
            print(f"\r合成进度: {count}/{total}", end="")

    out.release()
    print(f"\n视频合成完毕: {output_path}")


def process_single_video(video_path, output_folder, scale=0.5, keep_frames=False):
    """处理单个视频的主逻辑"""
    filename = os.path.basename(video_path)
    basename, ext = os.path.splitext(filename)

    # 输出视频路径
    output_video_path = os.path.join(output_folder, f"{basename}_small.mp4")

    # 创建专属的临时目录存放帧
    temp_frames_dir = os.path.join(output_folder, f"temp_{basename}")
    if os.path.exists(temp_frames_dir):
        shutil.rmtree(temp_frames_dir)  # 清理旧数据
    os.makedirs(temp_frames_dir, exist_ok=True)

    try:
        start_time = time.time()

        # 1. 提取并缩放
        original_fps, total_frames = extract_and_resize_frames(video_path, temp_frames_dir, scale)

        if total_frames and total_frames > 0:
            # 2. 合成视频
            images_to_video(temp_frames_dir, output_video_path, original_fps)
            print(f"处理成功！总耗时: {time.time() - start_time:.2f}秒")
        else:
            print("帧提取失败或视频为空")

    finally:
        # 3. 清理临时文件 (除非指定保留)
        if not keep_frames and os.path.exists(temp_frames_dir):
            print(f"正在清理临时文件: {temp_frames_dir}")
            shutil.rmtree(temp_frames_dir)


def main():
    parser = argparse.ArgumentParser(description='批量视频降采样工具 (Video -> Frames -> Resize -> Video)')
    parser.add_argument('--input_dir', type=str, required=True, help='输入存放视频的文件夹路径')
    parser.add_argument('--output_dir', type=str, required=True, help='输出结果的文件夹路径')
    parser.add_argument('--scale', type=float, default=0.5, help='缩放比例，默认0.5 (即长宽各变为一半)')
    parser.add_argument('--workers', type=int, default=8, help='帧处理并行线程数')

    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 支持的视频格式
    video_exts = {'.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv'}

    # 扫描视频文件
    video_files = [f for f in glob.glob(os.path.join(args.input_dir, '*'))
                   if os.path.splitext(f)[1].lower() in video_exts]

    if not video_files:
        print(f"在 {args.input_dir} 未找到视频文件")
        return

    print(f"找到 {len(video_files)} 个视频文件，准备开始处理...")
    print(f"缩放比例: {args.scale}")
    print("-" * 50)

    for i, video_path in enumerate(video_files):
        print(f"\n[{i + 1}/{len(video_files)}] 正在处理: {os.path.basename(video_path)}")
        process_single_video(video_path, args.output_dir, args.scale)


if __name__ == "__main__":
    main()