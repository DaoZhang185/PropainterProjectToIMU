import os
import argparse
import json
import shutil
import cv2
import numpy as np
import subprocess
import sys
import glob
from concurrent.futures import ThreadPoolExecutor

# =================================================================
# 引入依赖
# =================================================================
try:
    from movie2img import video_to_frames
    from scene_detector import detect_scenes_from_folder
    from make_mask import generate_local_masks
    from mask_loader import load_poses_from_json
except ImportError as e:
    print(f"CRITICAL ERROR: 缺少依赖脚本 ({e})。请确保所有 .py 文件都在同一目录下。")
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


def resolve_overlaps(all_poses, padding=50, img_w=1920, img_h=1080):
    boxes = []
    for key, coords_list in all_poses.items():
        if not coords_list: continue
        arr = np.array(coords_list)
        x1, y1 = np.min(arr[:, 0]), np.min(arr[:, 1])
        x2, y2 = np.max(arr[:, 2]), np.max(arr[:, 3])
        nx1 = max(0, x1 - padding)
        ny1 = max(0, y1 - padding)
        nx2 = min(img_w, x2 + padding)
        ny2 = min(img_h, y2 + padding)
        boxes.append({
            "keys": [key],
            "crop_coords": [nx1, ny1, nx2, ny2],
            "original_poses": {key: coords_list}
        })

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
    if os.path.exists(dest_path) and os.path.getsize(dest_path) > 0: return True
    img = cv2.imread(src_path)
    if img is None: return False
    x1, y1, x2, y2 = map(int, coords)
    crop = img[y1:y2, x1:x2]
    cv2.imwrite(dest_path, crop)
    return True


# --- 核心修改：实现“挖孔 -> 填补”逻辑 ---
def merge_worker(args):
    """
    frame_idx: 当前处理的全局帧号
    frame_filename: 原图文件名
    full_frames_dir: 原图目录
    final_frames_dir: 结果保存目录
    merge_plans: 包含该帧需要合并的所有区域任务
    """
    frame_idx, frame_filename, full_frames_dir, final_frames_dir, merge_plans = args

    save_path = os.path.join(final_frames_dir, frame_filename)

    # 1. 读取原始高清大图
    original_img_path = os.path.join(full_frames_dir, frame_filename)
    base_img = cv2.imread(original_img_path)
    if base_img is None: return False

    # 遍历所有需要在此帧上修复的区域
    for plan in merge_plans:
        # 计算在片段中的相对索引
        relative_idx = frame_idx - plan['start_frame']

        # 寻找对应的修复结果图
        # 优先查找 frames 子文件夹，找不到则找根目录 (兼容不同版本的 ProPainter 输出)
        restored_filename = f"{relative_idx:04d}.png"
        restored_path_v1 = os.path.join(plan['result_dir'], "frames", restored_filename)
        restored_path_v2 = os.path.join(plan['result_dir'], restored_filename)

        restored_path = restored_path_v1 if os.path.exists(restored_path_v1) else restored_path_v2

        if os.path.exists(restored_path):
            restored_crop = cv2.imread(restored_path)

            if restored_crop is not None:
                x1, y1, x2, y2 = map(int, plan['coords'])

                # 双重检查尺寸，防止溢出
                h_crop, w_crop = restored_crop.shape[:2]
                target_h, target_w = y2 - y1, x2 - x1

                # 如果模型输出尺寸有微小偏差，resize一下以完美匹配孔位
                if h_crop != target_h or w_crop != target_w:
                    restored_crop = cv2.resize(restored_crop, (target_w, target_h))

                # =========================================================
                # 【逻辑修改】实现“先清空，后填入”
                # =========================================================

                # 步骤 A: 物理清空 (Set to Black/Zero)
                # 将原图上该区域的像素值全部置为 0
                # 这确保了原始带水印的像素被彻底移除
                base_img[y1:y2, x1:x2] = 0

                # 步骤 B: 填入修复后的像素
                # 将修复好的局部图填入刚刚挖出的“黑洞”中
                # 此时我们是在纯黑背景上写入，保证了像素的纯净性
                base_img[y1:y2, x1:x2] = restored_crop

        else:
            # 如果某帧推理失败或缺失，为了不让画面该区域闪烁或黑屏，
            # 我们选择保持原图不动（不挖孔），或者你可以选择报错。
            # 这里保持原图不动是比较稳妥的降级方案。
            pass

    cv2.imwrite(save_path, base_img)
    return True


def render_video(frames_dir, source_video, output_path, fps=25.0):
    print(f"[Render] 正在合成视频: {output_path} (FPS: {fps})")
    has_audio = False
    ffmpeg_cmd = ['ffmpeg', '-y']
    ffmpeg_cmd.extend(['-f', 'image2', '-framerate', str(fps)])
    ffmpeg_cmd.extend(['-i', os.path.join(frames_dir, 'frame_%04d.png')])
    ffmpeg_cmd.extend(['-i', source_video])
    ffmpeg_cmd.extend(['-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18'])
    ffmpeg_cmd.extend(['-map', '0:v:0', '-map', '1:a:0?', '-c:a', 'copy'])
    ffmpeg_cmd.append('-shortest')
    ffmpeg_cmd.append(output_path)

    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError as e:
        print(f"[Render] ffmpeg 合成失败: {e}")
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
    parser.add_argument('--padding', type=int, default=50, help='裁切区域外扩像素')
    parser.add_argument('--max_workers', type=int, default=8, help='并行处理线程数')
    args = parser.parse_args()

    # 0. 初始化
    report_progress("start", 0)
    setup_dirs(args.workspace)

    # === Step 1: 视频转全量帧 ===
    print("\n=== Step 1: 全量帧提取 ===")
    frames_dir = os.path.join(args.workspace, "full_frames_original")

    has_frames = False
    if os.path.exists(frames_dir):
        existing_frames = [f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))]
        if len(existing_frames) > 10: has_frames = True

    if not has_frames:
        print("[Main] 正在转换视频为帧序列...")
        success = video_to_frames(args.video, frames_dir)
        if not success: sys.exit(1)

    report_progress("extract_frames", 10)

    # 获取基础信息
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith(('.png', '.jpg'))])
    if not frame_files: sys.exit(1)

    # 自动探测 FPS
    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=avg_frame_rate', '-of',
               'default=noprint_wrappers=1:nokey=1', args.video]
        res = subprocess.run(cmd, capture_output=True, text=True)
        num, den = map(int, res.stdout.strip().split('/'))
        fps = num / den
    except:
        fps = 25.0
    print(f"[Main] 探测到帧率: {fps}")

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

        current_percent = 10 + int((plan_idx + 0.5) / total_plans * 20)
        report_progress("cropping", current_percent)

        # 2.2 场景检测
        print(f"    检测场景...")
        scene_json_path = os.path.join(region_dir, f"{region_name}_scenes.json")
        scene_indices = detect_scenes_from_folder(crop_frames_dir, threshold=10.0)

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

                # 记录任务
                all_merge_tasks.append({
                    'result_dir': seg_out_dir,
                    'coords': plan['crop_coords'],
                    'start_frame': seg_start,
                    'end_frame': seg_end
                })

                if os.path.exists(os.path.join(seg_out_dir, "success_flag")):
                    continue

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

                cmd = [sys.executable, args.model_path, "--video", current_input_dir, "--mask", current_mask_dir,
                       "--output", seg_out_dir, "--fp16"]
                try:
                    subprocess.run(cmd, check=True)
                    with open(os.path.join(seg_out_dir, "success_flag"), 'w') as f:
                        f.write("ok")
                except subprocess.CalledProcessError:
                    print(f"推理失败: {seg_name}")

                base_percent = 40 + (plan_idx / total_plans * 50)
                current_add = (1.0 / total_plans) * 50 * ((i + 1) / total_scenes)
                report_progress("processing", base_percent + current_add)

                try:
                    shutil.rmtree(current_input_dir)
                    shutil.rmtree(current_mask_dir)
                except:
                    pass

    # === Step 6: 结果回贴 (Merge Back) ===
    print("\n=== Step 6: 合并修复结果到原图 ===")
    report_progress("merging", 90)

    final_frames_dir = os.path.join(args.workspace, "full_frames_final")
    setup_dirs(final_frames_dir)

    print(f"[Main] 正在生成 {len(frame_files)} 帧的最终图像...")

    merge_worker_tasks = []

    for idx, fname in enumerate(frame_files):
        # 筛选出覆盖当前帧 idx 的所有 plan
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

    # === 结束 ===
    report_progress("completed", 100)


if __name__ == "__main__":
    main()