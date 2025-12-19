import os
import sys
import time
import shutil
from moviepy.editor import VideoFileClip
from concurrent.futures import ThreadPoolExecutor
import cv2


def clear_directory_safe(folder_path):
    """安全清空目录，避免 NFS 设备繁忙错误"""
    if not os.path.exists(folder_path):
        return True

    max_retries = 3  # 减少重试次数
    for attempt in range(max_retries):
        try:
            # 只删除文件，不删除目录本身
            for filename in os.listdir(folder_path):
                file_path = os.path.join(folder_path, filename)

                # 跳过NFS临时文件
                if filename.startswith('.nfs'):
                    continue

                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            return True
        except OSError as e:
            if attempt < max_retries - 1:
                time.sleep(2)
            else:
                print(f"警告：无法完全清空目录 {folder_path}，跳过NFS锁定文件")
                return False
    return True


def save_frame_batch(args):
    """批量保存帧的包装函数"""
    clip, frame_indices, frame_interval, output_folder = args
    saved_count = 0

    for i in frame_indices:
        t = i * frame_interval
        if t >= clip.duration:
            break

        frame_filename = os.path.join(output_folder, f"frame_{i + 1:04d}.png")
        try:
            clip.save_frame(frame_filename, t=t)
            saved_count += 1
        except Exception as e:
            continue

    return saved_count


def video_to_frames_parallel(video_path, output_folder, target_fps=None, max_workers=4):
    """
    并行视频帧提取
    """
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    # 安全清空目录
    clear_directory_safe(output_folder)

    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件 {video_path} 不存在")
        return False

    # 加载视频文件
    try:
        clip = VideoFileClip(video_path)
    except Exception as e:
        print(f"错误：无法加载视频文件 {video_path}: {e}")
        return False

    # 获取视频的原始帧率
    original_fps = clip.fps
    print(f"原始视频帧率: {original_fps} FPS")

    # 确定使用的帧率
    if target_fps is None:
        used_fps = original_fps
        print(f"使用原始帧率: {used_fps} FPS")
    else:
        used_fps = float(target_fps)
        print(f"使用指定帧率: {used_fps} FPS (原始: {original_fps} FPS)")

    print(f"视频时长: {clip.duration} 秒")

    # 计算使用的帧间隔和总帧数
    frame_interval = 1 / used_fps
    total_frames = int(clip.duration * used_fps)
    print(f"预计提取帧数: {total_frames}")

    # 将帧索引分成多个批次
    frame_indices = list(range(total_frames))
    batch_size = max(1, total_frames // (max_workers * 2))
    batches = [frame_indices[i:i + batch_size] for i in range(0, total_frames, batch_size)]

    print(f"使用 {len(batches)} 个批次并行处理...")

    # 准备任务
    tasks = [(clip, batch, frame_interval, output_folder) for batch in batches]

    # 使用线程池并行处理
    total_saved = 0
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(save_frame_batch, task) for task in tasks]

        for i, future in enumerate(futures):
            saved_in_batch = future.result()
            total_saved += saved_in_batch
            print(f"批次 {i + 1}/{len(batches)} 完成，保存 {saved_in_batch} 帧，总计 {total_saved}/{total_frames}")

    # 关闭视频文件
    clip.close()

    print(f"帧提取完成！共保存 {total_saved} 帧到 {output_folder}")
    return total_saved > 0


def video_to_frames(video_path, output_folder, target_fps=None):
    """
    将视频转换为帧图片 - 自动选择并行或串行
    """
    # 先获取视频信息来决定处理方式
    try:
        clip = VideoFileClip(video_path)
        duration = clip.duration
        original_fps = clip.fps
        used_fps = float(target_fps) if target_fps else original_fps
        total_frames = int(duration * used_fps)
        clip.close()

        # 如果帧数很多，使用并行处理
        if total_frames > 100:
            return video_to_frames_parallel(video_path, output_folder, target_fps)
        else:
            return video_to_frames_original(video_path, output_folder, target_fps)

    except Exception as e:
        print(f"错误：无法获取视频信息: {e}")
        return video_to_frames_original(video_path, output_folder, target_fps)


def video_to_frames_original(video_path, output_folder, target_fps=None):
    """
    原来的串行处理函数
    """
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)

    # 安全清空目录
    clear_directory_safe(output_folder)

    # 检查视频文件是否存在
    if not os.path.exists(video_path):
        print(f"错误：视频文件 {video_path} 不存在")
        return False

    # 加载视频文件
    try:
        clip = VideoFileClip(video_path)
    except Exception as e:
        print(f"错误：无法加载视频文件 {video_path}: {e}")
        return False

    # 获取视频的原始帧率
    original_fps = clip.fps
    print(f"原始视频帧率: {original_fps} FPS")

    # 确定使用的帧率
    if target_fps is None:
        used_fps = original_fps
        print(f"使用原始帧率: {used_fps} FPS")
    else:
        used_fps = float(target_fps)
        print(f"使用指定帧率: {used_fps} FPS (原始: {original_fps} FPS)")

    print(f"视频时长: {clip.duration} 秒")

    # 计算使用的帧间隔
    frame_interval = 1 / used_fps

    saved_count = 0
    total_frames = int(clip.duration * used_fps)
    print(f"预计提取帧数: {total_frames}")

    # 生成帧并保存为 PNG 图片
    for i in range(total_frames):
        t = i * frame_interval
        if t >= clip.duration:
            break

        frame_filename = os.path.join(output_folder, f"frame_{i + 1:04d}.png")
        try:
            clip.save_frame(frame_filename, t=t)
            saved_count += 1
            if saved_count % 50 == 0:
                print(f"已保存 {saved_count}/{total_frames} 帧...")
        except Exception as e:
            continue

    # 关闭视频文件
    clip.close()
    print(f"帧提取完成！共保存 {saved_count} 帧到 {output_folder}")
    return saved_count > 0


# 示例用法
if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("用法: python movie2img.py <视频路径> <输出文件夹> [目标帧率] [最大线程数]")
        print("示例: python movie2img.py input.mp4 output_frames 24 4")
        sys.exit(1)

    video_path = sys.argv[1]
    output_folder = sys.argv[2]

    # 处理可选的帧率参数
    target_fps = None
    max_workers = 4

    if len(sys.argv) >= 4:
        try:
            target_fps = float(sys.argv[3])
            if target_fps <= 0:
                print("错误：帧率必须是正数")
                sys.exit(1)
        except ValueError:
            print("错误：帧率必须是数字")
            sys.exit(1)

    if len(sys.argv) >= 5:
        try:
            max_workers = int(sys.argv[4])
        except ValueError:
            print("错误：线程数必须是整数")

    print(f"输入视频: {video_path}")
    print(f"输出文件夹: {output_folder}")
    if target_fps:
        print(f"指定帧率: {target_fps} FPS")
    else:
        print("使用视频原始帧率")

    success = video_to_frames(video_path, output_folder, target_fps)
    if not success:
        print("帧提取失败！")
        sys.exit(1)