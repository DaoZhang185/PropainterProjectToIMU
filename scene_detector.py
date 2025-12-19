import argparse
import os
import subprocess
import json
import cv2
import numpy as np


def load_mask_regions(mask_json_path=None):
    """加载mask区域，仅返回台标(1)用于场景检测，忽略字幕和标题"""
    if mask_json_path and os.path.exists(mask_json_path):
        try:
            from mask_loader import load_poses_from_json
            all_poses = load_poses_from_json(mask_json_path)

            # 【调试信息】打印原始数据，确认文件更新
            print(f"DEBUG: 原始JSON包含区域: {list(all_poses.keys()) if all_poses else 'None'}")

            # 逻辑修改：只要包含 '1' (台标)，就只提取 '1' 返回。
            if all_poses and '1' in all_poses and len(all_poses['1']) > 0:
                print(f"✓ 场景检测加载区域1坐标: {len(all_poses['1'])} 个框")

                # 【验证点】如果日志没显示这句话，说明代码没更新！
                print("DEBUG: 已执行强制过滤，仅保留台标区域！")

                filtered_poses = {'1': all_poses['1']}
                return filtered_poses
            else:
                print("✗ JSON加载成功但未包含有效的'台标'(1)区域，将跳过场景检测。")
                return None
        except Exception as e:
            print(f"✗ 加载JSON失败: {e}，将跳过场景检测。")
            return None
    print("✗ 未指定JSON路径或路径无效，无法进行Mask区域检测，将跳过场景检测。")
    return None


def create_mask_region_image(frame_shape, poses):
    mask = np.zeros(frame_shape[:2], dtype=np.uint8)
    if not poses: return mask
    for region_key in poses:
        for region in poses[region_key]:
            x1, y1, x2, y2 = region
            padding = 1
            x1, y1 = max(0, x1 - padding), max(0, y1 - padding)
            x2, y2 = min(frame_shape[1], x2 + padding), min(frame_shape[0], y2 + padding)
            cv2.rectangle(mask, (x1, y1), (x2, y2), 255, -1)
    return mask


def detect_scenes_simple_masked(video_path, threshold=8.0, min_scene_duration=0.2, max_scene_duration=12.0,
                                mask_json_path=None):
    try:
        poses = load_mask_regions(mask_json_path)
        if poses is None: return None

        print(f"使用Mask区域场景检测，阈值: {threshold}")
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened(): return None

        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if fps <= 0 or total_frames <= 0: return None

        ret, first_frame = cap.read()
        if not ret: return None

        mask_image = create_mask_region_image(first_frame.shape, poses)
        if np.count_nonzero(mask_image) == 0:
            print("警告: Mask图像全黑，将跳过场景检测。")
            return None

        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        prev_frame = None
        scene_changes = [0]
        mask_pixel_count = np.count_nonzero(mask_image)
        sample_rate = max(1, int(fps / 8))

        for frame_idx in range(0, total_frames, sample_rate):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret: break

            masked_frame = cv2.bitwise_and(frame, frame, mask=mask_image)
            if prev_frame is not None:
                prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
                curr_gray = cv2.cvtColor(masked_frame, cv2.COLOR_BGR2GRAY)
                diff = cv2.absdiff(prev_gray, curr_gray)
                pixel_diff = np.sum(diff) / mask_pixel_count

                if pixel_diff > threshold:
                    scene_time = frame_idx / fps
                    if scene_time - scene_changes[-1] >= min_scene_duration:
                        scene_changes.append(scene_time)
            prev_frame = masked_frame
        cap.release()

        if total_frames > 0:
            final_time = total_frames / fps
            if final_time - scene_changes[-1] >= min_scene_duration:
                scene_changes.append(final_time)
            else:
                scene_changes[-1] = final_time

        print(f"检测完成: 扫描结束, 发现 {len(scene_changes) - 1} 个片段")
        if len(scene_changes) > 2: return scene_changes[1:-1]
        return []
    except Exception as e:
        print(f"场景检测异常: {e}")
        return None


def get_ffmpeg_path():
    try:
        result = subprocess.run(['which', 'ffmpeg'], capture_output=True, text=True)
        if result.returncode == 0: return result.stdout.strip()
    except:
        pass
    return 'ffmpeg'


def get_video_duration_exact(video_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-show_entries', 'format=duration', '-of',
               'default=noprint_wrappers=1:nokey=1', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        if result.returncode == 0 and result.stdout.strip(): return float(result.stdout.strip())
    except:
        pass
    return None


def get_video_fps(video_path):
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries',
               'stream=avg_frame_rate', '-of', 'default=noprint_wrappers=1:nokey=1', video_path]
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=5)
        if result.returncode == 0 and result.stdout.strip():
            num, den = map(int, result.stdout.strip().split('/'))
            if den > 0: return num / den
    except:
        pass
    return 25.0


# =========================================================================
# 终极修复：使用【帧数限制】而非【时间限制】
# =========================================================================
def create_video_segment_fixed(video_path, start, end, output_path):
    segment_duration = end - start
    ffmpeg_path = get_ffmpeg_path()

    # 1. 计算精确的帧数
    fps = get_video_fps(video_path)
    num_frames = int(segment_duration * fps)

    # 2. 打印调试信息，确认代码更新
    print(f"DEBUG: 切割区间 {start:.2f}-{end:.2f}, 时长 {segment_duration:.2f}s, 预计帧数: {num_frames}")

    # 3. 构建命令：使用 -frames:v 强制限制输出帧数
    # -ss 在 -i 之后（精确seek）
    # -frames:v 替代 -t，彻底规避时间戳问题
    cmd = [
        ffmpeg_path,
        '-i', video_path,
        '-ss', str(start),
        '-frames:v', str(num_frames),  # 强制只输出这么多帧
        '-c:v', 'mpeg4',
        '-q:v', '2',
        '-c:a', 'aac',
        '-strict', '-2',
        '-avoid_negative_ts', 'make_zero',
        '-y', output_path
    ]

    try:
        subprocess.run(cmd, capture_output=True, check=True)
        return os.path.exists(output_path) and os.path.getsize(output_path) > 1024
    except Exception as e:
        print(f"分割失败: {e}")
        return False


def segment_video_fixed(video_path, scene_times, output_dir, min_duration=0.5, max_scene_duration=15.0):
    duration = get_video_duration_exact(video_path)
    if not duration: return []

    all_times = sorted([0.0] + [t for t in scene_times if 0 < t < duration] + [duration])
    segments = []
    idx = 0

    for i in range(len(all_times) - 1):
        s, e = all_times[i], all_times[i + 1]
        if e - s < min_duration: continue

        path = os.path.join(output_dir, f"scene_{idx:04d}.mp4")
        print(f"正在分割场景 {idx}: {s:.2f}s - {e:.2f}s")
        if create_video_segment_fixed(video_path, s, e, path):
            segments.append({'index': idx, 'start': s, 'end': e, 'duration': e - s, 'path': path})
            idx += 1
    return segments


def test_ffmpeg_command():
    try:
        subprocess.run(['ffmpeg', '-version'], capture_output=True, timeout=10)
        return True
    except:
        return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--threshold', type=float, default=8.0)
    parser.add_argument('--min-duration', type=float, default=0.2)
    parser.add_argument('--max-duration', type=float, default=12.0)
    parser.add_argument('--method', default='simple')
    parser.add_argument('--mask-json', type=str, default=None)

    args = parser.parse_args()
    os.makedirs(args.output, exist_ok=True)

    if not test_ffmpeg_command(): exit(1)

    print(f"开始Mask区域场景检测: {args.input}")
    if args.mask_json: print(f"Mask JSON文件: {args.mask_json}")

    scene_times = detect_scenes_simple_masked(args.input, args.threshold, args.min_duration, args.max_duration,
                                              args.mask_json)

    if scene_times is None:
        print("提示：跳过场景检测。")
        scene_times = []

    print(f"检测完成: {len(scene_times)} 个场景切换点")

    if not scene_times:
        dur = get_video_duration_exact(args.input)
        if dur:
            p = os.path.join(args.output, "scene_0000.mp4")
            if create_video_segment_fixed(args.input, 0, dur, p):
                segs = [{'index': 0, 'start': 0, 'end': dur, 'duration': dur, 'path': p}]
            else:
                segs = []
        else:
            segs = []
    else:
        segs = segment_video_fixed(args.input, scene_times, args.output, args.min_duration, args.max_duration)

    if not segs: exit(1)

    info = {'total_segments': len(segs), 'segments': segs}
    with open(os.path.join(args.output, 'segment_info.json'), 'w') as f:
        json.dump(info, f, indent=2)