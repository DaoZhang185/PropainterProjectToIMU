import os
import argparse
import json
import shutil
import cv2
import numpy as np
import subprocess
import sys
import glob
import datetime
from concurrent.futures import ThreadPoolExecutor

# =================================================================
# 引入依赖
# =================================================================
try:
    from scene_detector import detect_scenes_from_folder
    from make_mask import generate_local_masks
    from mask_loader import load_poses_from_json
except ImportError as e:
    print(f"CRITICAL ERROR: 缺少依赖脚本 ({e})。请确保 scene_detector.py, make_mask.py, mask_loader.py 在同一目录下。")
    sys.exit(1)


# =================================================================
# 辅助函数
# =================================================================

def report_progress(stage, percent):
    percent = max(0, min(100, int(percent)))
    print(f"PROGRESS:{stage}:{percent}", flush=True)


def setup_dirs(base_path):
    if not os.path.exists(base_path):
        os.makedirs(base_path, exist_ok=True)


def extract_frames_ffmpeg(video_path, output_dir):
    """
    使用 FFmpeg 提取帧。
    每次强制重新提取，确保数据最新。
    """
    if not os.path.exists(video_path):
        print(f"[Error] 视频文件不存在: {video_path}")
        return False

    # 强制清理旧目录
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    setup_dirs(output_dir)

    print(f"[Extract] 正在使用 FFmpeg 提取全量帧: {video_path}")

    cmd = [
        'ffmpeg',
        '-i', video_path,
        '-start_number', '0',
        '-vsync', '0',
        '-q:v', '2',
        os.path.join(output_dir, 'frame_%04d.png')
    ]

    try:
        subprocess.run(cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        count = len(os.listdir(output_dir))
        print(f"[Extract] 成功提取 {count} 帧")
        return count > 0
    except subprocess.CalledProcessError as e:
        print(f"[Error] FFmpeg 提帧失败: {e}")
        return False


def get_recursive_segments(start_idx, end_idx, max_frames=300):
    length = end_idx - start_idx
    if length <= max_frames:
        return [(start_idx, end_idx)]
    half = length // 2
    remainder = length % 2
    len1 = half + remainder
    mid_idx = start_idx + len1
    return get_recursive_segments(start_idx, mid_idx, max_frames) + \
        get_recursive_segments(mid_idx, end_idx, max_frames)


def resolve_overlaps(all_poses, padding=150, img_w=1920, img_h=1080):
    """
    计算裁切区域。
    Padding 默认 150，且强制 16 倍数对齐。
    """
    boxes = []
    for key, coords_list in all_poses.items():
        if not coords_list: continue
        arr = np.array(coords_list)
        x1, y1 = np.min(arr[:, 0]), np.min(arr[:, 1])
        x2, y2 = np.max(arr[:, 2]), np.max(arr[:, 3])

        # 1. 基础 Padding
        nx1 = max(0, x1 - padding)
        ny1 = max(0, y1 - padding)
        nx2 = min(img_w, x2 + padding)
        ny2 = min(img_h, y2 + padding)

        # 2. 强制宽高为 16 的倍数
        w = nx2 - nx1
        h = ny2 - ny1
        pad_w = (16 - (w % 16)) % 16
        pad_h = (16 - (h % 16)) % 16

        if nx2 + pad_w <= img_w:
            nx2 += pad_w
        else:
            nx1 = max(0, nx1 - pad_w)

        if ny2 + pad_h <= img_h:
            ny2 += pad_h
        else:
            ny1 = max(0, ny1 - pad_h)

        boxes.append({
            "keys": [key],
            "crop_coords": [int(nx1), int(ny1), int(nx2), int(ny2)],
            "original_poses": {key: coords_list}
        })

    # 合并重叠
    merged = True
    while merged:
        merged = False
        new_boxes = []
        skip_indices = set()
        for i in range(len(boxes)):
            if i in skip_indices: continue
            curr = boxes[i]
            cx1, cy1, cx2, cy2 = curr["crop_coords"]
            for j in range(i + 1, len(boxes)):
                if j in skip_indices: continue
                other = boxes[j]
                ox1, oy1, ox2, oy2 = other["crop_coords"]
                if not (cx2 < ox1 or cx1 > ox2 or cy2 < oy1 or cy1 > oy2):
                    cx1 = min(cx1, ox1)
                    cy1 = min(cy1, oy1)
                    cx2 = max(cx2, ox2)
                    cy2 = max(cy2, oy2)
                    curr["keys"].extend(other["keys"])
                    curr["original_poses"].update(other["original_poses"])
                    skip_indices.add(j)
                    merged = True
            curr["crop_coords"] = [cx1, cy1, cx2, cy2]
            new_boxes.append(curr)
        boxes = new_boxes
    return boxes


def crop_worker(args):
    src_path, dest_path, coords = args
    # 强制覆盖
    img = cv2.imread(src_path)
    if img is None: return False
    x1, y1, x2, y2 = map(int, coords)
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(dest_path, crop)
    return True


def merge_worker(args):
    frame_idx, frame_filename, full_frames_dir, final_frames_dir, merge_plans = args
    save_path = os.path.join(final_frames_dir, frame_filename)

    original_img_path = os.path.join(full_frames_dir, frame_filename)
    base_img = cv2.imread(original_img_path)
    if base_img is None: return False

    for plan in merge_plans:
        relative_idx = frame_idx - plan['start_frame']

        restored_filename = f"{relative_idx:04d}.png"
        restored_path_v1 = os.path.join(plan['result_dir'], "frames", restored_filename)
        restored_path_v2 = os.path.join(plan['result_dir'], restored_filename)
        restored_path = restored_path_v1 if os.path.exists(restored_path_v1) else restored_path_v2

        if os.path.exists(restored_path):
            restored_crop = cv2.imread(restored_path)
            if restored_crop is not None:
                x1, y1, x2, y2 = map(int, plan['coords'])

                h_crop, w_crop = restored_crop.shape[:2]
                target_h, target_w = y2 - y1, x2 - x1

                if h_crop != target_h or w_crop != target_w:
                    restored_crop = cv2.resize(restored_crop, (target_w, target_h))

                base_img[y1:y2, x1:x2] = 0
                base_img[y1:y2, x1:x2] = restored_crop

    cv2.imwrite(save_path, base_img)
    return True


def render_video(frames_dir, source_video, output_path, fps=25.0):
    print(f"[Render] 正在合成视频: {output_path} (FPS: {fps})")

    ffmpeg_cmd = [
        'ffmpeg', '-y',
        '-f', 'image2',
        '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-i', source_video,
        '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        '-c:v', 'mpeg4',
        '-q:v', '2',
        '-map', '0:v:0',
        '-map', '1:a:0?',
        '-c:a', 'copy',
        '-shortest',
        output_path
    ]

    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        try:
            cmd_no_audio = [
                'ffmpeg', '-y',
                '-f', 'image2',
                '-framerate', str(fps),
                '-i', os.path.join(frames_dir, 'frame_%04d.png'),
                '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                '-c:v', 'mpeg4',
                '-q:v', '2',
                output_path
            ]
            subprocess.run(cmd_no_audio, check=True)
            return True
        except Exception as e2:
            print(f"[Render] 最终合成失败: {e2}")
            return False


# =================================================================
# 主流程
# =================================================================

def main():
    parser = argparse.ArgumentParser(description="ProPainter Python 全流程主控脚本")
    parser.add_argument('--video', required=True, help='原始视频路径')
    parser.add_argument('--mask_json', required=True, help='Mask坐标JSON文件路径')
    parser.add_argument('--workspace', required=True, help='工作目录')
    parser.add_argument('--model_path', default='inference_propainter.py', help='推理脚本路径')
    parser.add_argument('--output_video', default='final_output.mp4', help='最终输出视频文件名')
    parser.add_argument('--max_frames', type=int, default=300, help='单次推理最大帧数限制')
    # Padding 默认 150，保证背景参考充足
    parser.add_argument('--padding', type=int, default=150, help='裁切区域外扩像素')
    parser.add_argument('--max_workers', type=int, default=8, help='并行处理线程数')
    args = parser.parse_args()

    # 0. 初始化 & 【目录结构升级】
    report_progress("start", 0)

    # 视频名 + 时间戳 目录结构
    video_name = os.path.splitext(os.path.basename(args.video))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.workspace = os.path.join(args.workspace, video_name, f"{video_name}_{timestamp}")

    print(f"\n[Main] ===========================================")
    print(f"[Main] 任务启动: {video_name}")
    print(f"[Main] 独立工作目录: {args.workspace}")
    print(f"[Main] ===========================================\n")

    setup_dirs(args.workspace)

    # === Step 1: 视频转全量帧 ===
    print("\n=== Step 1: 全量帧提取 ===")
    frames_dir = os.path.join(args.workspace, "full_frames_original")

    success = extract_frames_ffmpeg(args.video, frames_dir)
    if not success: sys.exit(1)

    report_progress("extract_frames", 10)

    frame_files_map = {}
    for f in os.listdir(frames_dir):
        if f.endswith(('.png', '.jpg')) and 'frame_' in f:
            try:
                idx = int(f.split('_')[-1].split('.')[0])
                frame_files_map[idx] = f
            except:
                pass

    sorted_indices = sorted(frame_files_map.keys())
    if not sorted_indices:
        print("[Error] 无法解析帧文件索引")
        sys.exit(1)

    frame_files = [frame_files_map[i] for i in sorted_indices]
    print(f"[Main] 帧索引范围: {min(sorted_indices)} - {max(sorted_indices)}, 总计: {len(frame_files)}")

    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=avg_frame_rate', '-of',
               'default=noprint_wrappers=1:nokey=1', args.video]
        res = subprocess.run(cmd, capture_output=True, text=True)
        vals = res.stdout.strip().split('/')
        fps = float(vals[0]) / float(vals[1]) if len(vals) == 2 else float(vals[0])
    except:
        fps = 25.0
    print(f"[Main] 视频帧率: {fps:.2f} FPS")

    sample_img = cv2.imread(os.path.join(frames_dir, frame_files[0]))
    img_h, img_w = sample_img.shape[:2]

    # === Step 2: 区域解析与裁切 ===
    print("\n=== Step 2: 区域解析与裁切 ===")
    all_poses = load_poses_from_json(args.mask_json)
    if not all_poses: sys.exit(1)

    work_plans = resolve_overlaps(all_poses, padding=args.padding, img_w=img_w, img_h=img_h)
    total_plans = len(work_plans)

    all_merge_tasks = []

    for plan_idx, plan in enumerate(work_plans):
        region_id = "_".join(plan['keys'])
        region_name = f"region_{region_id}"

        region_dir = os.path.join(args.workspace, region_name)
        crop_frames_dir = os.path.join(region_dir, "frames")
        setup_dirs(crop_frames_dir)

        # 2.1 裁切
        print(f"\n>>> 处理区域 [{region_name}] - 裁切")
        tasks = []
        for fname in frame_files:
            src = os.path.join(frames_dir, fname)
            dst = os.path.join(crop_frames_dir, fname)
            tasks.append((src, dst, plan['crop_coords']))

        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            list(executor.map(crop_worker, tasks))

        report_progress("cropping", 10 + int((plan_idx + 0.5) / total_plans * 20))

        # 2.2 场景检测 (阈值 6.0)
        print(f"    检测场景...")
        scene_json_path = os.path.join(region_dir, f"{region_name}_scenes.json")
        scene_indices = detect_scenes_from_folder(crop_frames_dir, threshold=6.0)

        total_video_frames = len(frame_files)
        if 0 not in scene_indices: scene_indices.insert(0, 0)
        if total_video_frames not in scene_indices: scene_indices.append(total_video_frames)
        scene_indices = sorted(list(set(scene_indices)))

        with open(scene_json_path, 'w') as f:
            json.dump(scene_indices, f)

        # 2.3 生成 Mask
        print(f"    生成 Mask...")
        mask_output_dir = os.path.join(region_dir, "masks")
        generate_local_masks(crop_frames_dir, mask_output_dir, plan['original_poses'], plan['crop_coords'],
                             static_keys=['1', '3'])

        report_progress("mask_generation", 40)

        # 2.4 分段推理
        print(f"    开始推理...")
        results_dir = os.path.join(region_dir, "results")
        temp_inputs_base = os.path.join(region_dir, "temp_inputs")
        temp_masks_base = os.path.join(region_dir, "temp_masks")
        setup_dirs(results_dir)
        setup_dirs(temp_inputs_base)
        setup_dirs(temp_masks_base)

        total_scenes = len(scene_indices) - 1

        for i in range(total_scenes):
            scene_start = scene_indices[i]
            scene_end = scene_indices[i + 1]
            segments = get_recursive_segments(scene_start, scene_end, args.max_frames)

            for seg_start, seg_end in segments:
                seg_name = f"seg_{seg_start:06d}_{seg_end:06d}"
                seg_out_dir = os.path.join(results_dir, seg_name)

                all_merge_tasks.append({
                    'result_dir': seg_out_dir,
                    'coords': plan['crop_coords'],
                    'start_frame': seg_start,
                    'end_frame': seg_end
                })

                frames_out_dir = os.path.join(seg_out_dir, "frames")
                mp4_output = os.path.join(seg_out_dir, "inference_output.mp4")

                # 每次重新运行推理
                current_input_dir = os.path.join(temp_inputs_base, seg_name)
                current_mask_dir = os.path.join(temp_masks_base, seg_name)
                if os.path.exists(current_input_dir): shutil.rmtree(current_input_dir)
                if os.path.exists(current_mask_dir): shutil.rmtree(current_mask_dir)
                os.makedirs(current_input_dir)
                os.makedirs(current_mask_dir)

                valid_count = 0
                for f_idx in range(seg_start, seg_end):
                    if f_idx >= len(frame_files): break
                    fname = frame_files[f_idx]

                    src_f = os.path.abspath(os.path.join(crop_frames_dir, fname))
                    dst_f = os.path.join(current_input_dir, fname)
                    os.symlink(src_f, dst_f)

                    src_m_dynamic = os.path.abspath(os.path.join(mask_output_dir, fname))
                    src_m_static = os.path.abspath(os.path.join(mask_output_dir, "static_mask.png"))
                    dst_m = os.path.join(current_mask_dir, fname)

                    if os.path.exists(src_m_dynamic):
                        os.symlink(src_m_dynamic, dst_m)
                    elif os.path.exists(src_m_static):
                        os.symlink(src_m_static, dst_m)
                    valid_count += 1

                if valid_count == 0: continue

                # 【核心修复】：回归标准参数，解决黑色残留
                # mask_dilation: 4 (标准值，避免0导致的边缘生硬)
                # flow_mask_dilation: 4 (标准值，避免20对字幕的过度忽略)
                cmd = [
                    sys.executable, args.model_path,
                    "--video", current_input_dir,
                    "--mask", current_mask_dir,
                    "--output", seg_out_dir,
                    "--fp16",
                    "--mask_dilation", "4",
                    "--flow_mask_dilation", "4",
                    "--raft_iter", "20",
                    "--ref_stride", "10"
                ]

                try:
                    subprocess.run(cmd, check=True)
                    if os.path.exists(mp4_output):
                        os.makedirs(frames_out_dir, exist_ok=True)
                        subprocess.run(['ffmpeg', '-y', '-i', mp4_output, '-start_number', '0',
                                        os.path.join(frames_out_dir, '%04d.png')], check=True,
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                except subprocess.CalledProcessError:
                    print(f"推理失败: {seg_name}")

                try:
                    shutil.rmtree(current_input_dir)
                    shutil.rmtree(current_mask_dir)
                except:
                    pass

                base_percent = 40 + (plan_idx / total_plans * 50)
                current_add = (1.0 / total_plans) * 50 * ((i + 1) / total_scenes)
                report_progress("processing", base_percent + current_add)

    # === Step 6: 结果回贴 ===
    print("\n=== Step 6: 合并修复结果到原图 ===")
    report_progress("merging", 90)

    final_frames_dir = os.path.join(args.workspace, "full_frames_final")
    setup_dirs(final_frames_dir)

    print(f"[Main] 正在生成 {len(frame_files)} 帧的最终图像...")

    merge_worker_tasks = []
    for idx in sorted_indices:
        fname = frame_files_map[idx]
        relevant_plans = []
        for task in all_merge_tasks:
            if task['start_frame'] <= idx < task['end_frame']:
                relevant_plans.append(task)
        merge_worker_tasks.append((idx, fname, frames_dir, final_frames_dir, relevant_plans))

    with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        list(executor.map(merge_worker, merge_worker_tasks))

    print("[Main] 图片合并完成。")

    # === Step 7: 合成视频 ===
    print("\n=== Step 7: 渲染最终视频 ===")
    report_progress("rendering", 95)

    output_video_path = os.path.join(args.workspace, args.output_video)
    if render_video(final_frames_dir, args.video, output_video_path, fps=fps):
        print(f"[Success] 最终视频已生成: {output_video_path}")
    else:
        print("[Error] 视频合成失败")

    report_progress("completed", 100)


if __name__ == "__main__":
    main()