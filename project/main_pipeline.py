import sys
import os

# =================================================================
# 【路径修复】确保脚本能引用上级目录（根目录）的模块
# =================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../ProPainter/project
root_dir = os.path.dirname(current_dir)  # .../ProPainter
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# =================================================================

import argparse
import json
import shutil
import cv2
import numpy as np
import subprocess
import glob
import datetime
import math
from concurrent.futures import ThreadPoolExecutor
import queue
import threading
import time


# =================================================================
# 0. 日志记录器
# =================================================================
class DualLogger(object):
    """
    同时将输出打印到控制台（供Web端捕获）和写入日志文件（供排查）
    """

    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding='utf-8', buffering=1)

    def write(self, message):
        try:
            self.terminal.write(message)
            self.log.write(message)
        except Exception:
            pass

    def flush(self):
        try:
            self.terminal.flush()
            self.log.flush()
        except Exception:
            pass


# =================================================================
# 引入依赖
# =================================================================
try:
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


def extract_frames_ffmpeg(video_path, output_dir):
    if not os.path.exists(video_path):
        print(f"[Error] 视频文件不存在: {video_path}")
        return False

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
    boxes = []
    for key, coords_list in all_poses.items():
        if not coords_list: continue

        all_x, all_y = [], []
        for item in coords_list:
            if len(item) == 4 and isinstance(item[0], (int, float)):
                all_x.extend([item[0], item[2]])
                all_y.extend([item[1], item[3]])
            elif isinstance(item, list) and isinstance(item[0], (list, tuple)):
                pts = np.array(item)
                all_x.extend(pts[:, 0])
                all_y.extend(pts[:, 1])

        if not all_x or not all_y:
            continue

        x1, y1 = min(all_x), min(all_y)
        x2, y2 = max(all_x), max(all_y)

        nx1 = max(0, x1 - padding)
        ny1 = max(0, y1 - padding)
        nx2 = min(img_w, x2 + padding)
        ny2 = min(img_h, y2 + padding)

        boxes.append({
            "keys": [key],
            "crop_coords": [int(nx1), int(ny1), int(nx2), int(ny2)],
            "original_poses": {key: coords_list}
        })

    adjusted = True
    iterations = 0
    while adjusted and iterations < 10:
        adjusted = False
        iterations += 1
        for i in range(len(boxes)):
            cx1, cy1, cx2, cy2 = boxes[i]["crop_coords"]
            c_center_x = (cx1 + cx2) / 2.0
            c_center_y = (cy1 + cy2) / 2.0

            for j in range(i + 1, len(boxes)):
                ox1, oy1, ox2, oy2 = boxes[j]["crop_coords"]
                if not (cx2 <= ox1 or cx1 >= ox2 or cy2 <= oy1 or cy1 >= oy2):
                    o_center_x = (ox1 + ox2) / 2.0
                    o_center_y = (oy1 + oy2) / 2.0

                    dx = c_center_x - o_center_x
                    dy = c_center_y - o_center_y

                    if abs(dx) > abs(dy):
                        if cx1 < ox1:
                            mid = (ox1 + cx2) // 2
                            cx2 = mid
                            ox1 = mid
                        else:
                            mid = (cx1 + ox2) // 2
                            cx1 = mid
                            ox2 = mid
                    else:
                        if cy1 < oy1:
                            mid = (oy1 + cy2) // 2
                            cy2 = mid
                            oy1 = mid
                        else:
                            mid = (cy1 + oy2) // 2
                            cy1 = mid
                            oy2 = mid

                    boxes[i]["crop_coords"] = [cx1, cy1, cx2, cy2]
                    boxes[j]["crop_coords"] = [ox1, oy1, ox2, oy2]
                    adjusted = True

    for i in range(len(boxes)):
        nx1, ny1, nx2, ny2 = boxes[i]["crop_coords"]
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

        boxes[i]["crop_coords"] = [int(nx1), int(ny1), int(nx2), int(ny2)]

    return boxes


def crop_worker(args):
    src_path, dest_path, coords = args
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
        'ffmpeg', '-y', '-f', 'image2', '-framerate', str(fps),
        '-i', os.path.join(frames_dir, 'frame_%04d.png'),
        '-i', source_video,
        '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2",
        '-c:v', 'mpeg4', '-q:v', '2',
        '-map', '0:v:0', '-map', '1:a:0?', '-c:a', 'copy', '-shortest',
        output_path
    ]
    try:
        subprocess.run(ffmpeg_cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        return True
    except subprocess.CalledProcessError:
        try:
            cmd_no_audio = [
                'ffmpeg', '-y', '-f', 'image2', '-framerate', str(fps),
                '-i', os.path.join(frames_dir, 'frame_%04d.png'),
                '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                '-c:v', 'mpeg4', '-q:v', '2',
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
    parser.add_argument('--padding', type=int, default=150, help='裁切区域外扩像素')
    parser.add_argument('--max_workers', type=int, default=8, help='最大并行处理线程数')
    args = parser.parse_args()

    if not os.path.isabs(args.model_path):
        args.model_path = os.path.join(current_dir, args.model_path)

    report_progress("start", 0)

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    args.workspace = os.path.join(args.workspace, video_name, f"{video_name}_{timestamp}")
    setup_dirs(args.workspace)

    log_file_path = os.path.join(args.workspace, "pipeline_run.log")
    sys.stdout = DualLogger(log_file_path)
    sys.stderr = sys.stdout

    print(f"\n[Main] ===========================================")
    print(f"[Main] 任务启动: {video_name}")
    print(f"[Main] 工作目录: {args.workspace}")
    print(f"[Main] 允许最高并发: {args.max_workers}")
    print(f"[Main] ===========================================\n")

    # === Step 1: 视频转全量帧 ===
    print("\n=== Step 1: 全量帧提取 ===")
    frames_dir = os.path.join(args.workspace, "full_frames_original")
    if not extract_frames_ffmpeg(args.video, frames_dir): sys.exit(1)

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

    print(f"[Main] 总计 {total_plans} 个处理区域计划。")

    for plan_idx, plan in enumerate(work_plans):
        region_id = "_".join(plan['keys'])
        region_name = f"region_{region_id}"
        print(f"\n>>> [Step 2.{plan_idx + 1}] 处理区域: {region_name} ({plan_idx + 1}/{total_plans})")

        region_dir = os.path.join(args.workspace, region_name)
        crop_frames_dir = os.path.join(region_dir, "frames")
        setup_dirs(crop_frames_dir)

        pct_step = 80.0 / total_plans
        plan_start_pct = 10.0 + plan_idx * pct_step

        print(f"    裁切中...")
        tasks = []
        for fname in frame_files:
            src = os.path.join(frames_dir, fname)
            dst = os.path.join(crop_frames_dir, fname)
            tasks.append((src, dst, plan['crop_coords']))
        with ThreadPoolExecutor(max_workers=args.max_workers) as executor:
            list(executor.map(crop_worker, tasks))

        report_progress(f"cropping_{region_id}", plan_start_pct + (pct_step * 0.1))

        print(f"    检测场景...")
        scene_json_path = os.path.join(region_dir, f"{region_name}_scenes.json")
        scene_indices = detect_scenes_from_folder(crop_frames_dir, threshold=6.0)
        total_video_frames = len(frame_files)
        if 0 not in scene_indices: scene_indices.insert(0, 0)
        if total_video_frames not in scene_indices: scene_indices.append(total_video_frames)
        scene_indices = sorted(list(set(scene_indices)))
        with open(scene_json_path, 'w') as f:
            json.dump(scene_indices, f)

        print(f"    生成 Mask...")
        mask_output_dir = os.path.join(region_dir, "masks")
        generate_local_masks(crop_frames_dir, mask_output_dir, plan['original_poses'], plan['crop_coords'],
                             static_keys=['1', '3', '4'])

        report_progress(f"mask_gen_{region_id}", plan_start_pct + (pct_step * 0.2))

        # =======================================================================
        # 【核心重构：动态智能调度池 - AIMD算法】
        # =======================================================================
        print(f"    开始智能调度推理 (防OOM并发)...")
        results_dir = os.path.join(region_dir, "results")
        temp_inputs_base = os.path.join(region_dir, "temp_inputs")
        temp_masks_base = os.path.join(region_dir, "temp_masks")
        setup_dirs(results_dir)
        setup_dirs(temp_inputs_base)
        setup_dirs(temp_masks_base)

        inference_tasks = []
        total_scenes = len(scene_indices) - 1

        # 第一阶段：将所有的任务分片打包进列表
        for i in range(total_scenes):
            scene_start = scene_indices[i]
            scene_end = scene_indices[i + 1]
            segments = get_recursive_segments(scene_start, scene_end, args.max_frames)

            for seg_idx, (seg_start, seg_end) in enumerate(segments):
                seg_name = f"seg_{seg_start:06d}_{seg_end:06d}"
                seg_out_dir = os.path.join(results_dir, seg_name)

                all_merge_tasks.append({
                    'result_dir': seg_out_dir,
                    'coords': plan['crop_coords'],
                    'start_frame': seg_start,
                    'end_frame': seg_end
                })

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

                    src_m = os.path.abspath(os.path.join(mask_output_dir, fname))
                    src_m_static = os.path.abspath(os.path.join(mask_output_dir, "static_mask.png"))
                    dst_m = os.path.join(current_mask_dir, fname)
                    if os.path.exists(src_m):
                        os.symlink(src_m, dst_m)
                    elif os.path.exists(src_m_static):
                        os.symlink(src_m_static, dst_m)
                    valid_count += 1

                if valid_count > 0:
                    cmd = [
                        sys.executable, args.model_path,
                        "--video", current_input_dir,
                        "--mask", current_mask_dir,
                        "--output", seg_out_dir,
                        "--fp16",
                        "--mask_dilation", "4",
                        "--flow_mask_dilation", "20",
                        "--raft_iter", "20",
                        "--ref_stride", "10",
                        "--subvideo_length", "30"
                    ]
                    inference_tasks.append({
                        'seg_name': seg_name,
                        'cmd': cmd,
                        'input_dir': current_input_dir,
                        'mask_dir': current_mask_dir,
                        'out_dir': seg_out_dir
                    })

        # 第二阶段：动态分发执行 (AIMD算法)
        task_queue = queue.Queue()
        for t in inference_tasks:
            task_queue.put(t)

        total_inference_tasks = len(inference_tasks)
        completed_tasks = 0

        # 初始并发数设置为 1，安全起步
        concurrency_limit = 1
        limit_locked = False
        active_threads = []
        print_lock = threading.Lock()

        def inference_worker(task, result_dict):
            seg_name = task['seg_name']
            with print_lock:
                print(f"    [分配] -> 启动线程处理: {seg_name}")

            try:
                process = subprocess.Popen(
                    task['cmd'],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    cwd=current_dir,
                    universal_newlines=True,
                    encoding='utf-8',
                    errors='replace'
                )

                is_oom = False
                while True:
                    line = process.stdout.readline()
                    if line == '' and process.poll() is not None:
                        break
                    if line:
                        line_str = line.rstrip()
                        # 核心：实时嗅探 OOM 报错
                        if "CUDA out of memory" in line_str or "RuntimeError: CUDA" in line_str:
                            is_oom = True
                        # 过滤无用日志，加上片头标签，防止不同线程日志互相干扰
                        elif "Processing" in line_str or "Network" in line_str or "%|" in line_str:
                            with print_lock:
                                print(f"    | [{seg_name}] {line_str}")

                return_code = process.wait()
                if is_oom or return_code == 137:
                    result_dict['status'] = 'oom'
                elif return_code != 0:
                    result_dict['status'] = 'error'
                    result_dict['code'] = return_code
                else:
                    mp4_output = os.path.join(task['out_dir'], "inference_output.mp4")
                    if os.path.exists(mp4_output):
                        frames_out_dir = os.path.join(task['out_dir'], "frames")
                        os.makedirs(frames_out_dir, exist_ok=True)
                        subprocess.run(['ffmpeg', '-y', '-i', mp4_output, '-start_number', '0',
                                        os.path.join(frames_out_dir, '%04d.png')],
                                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    result_dict['status'] = 'success'
            except Exception as e:
                with print_lock:
                    print(f"    [异常] [{seg_name}] 线程执行出错: {e}")
                result_dict['status'] = 'error'

        # 主调度循环
        while not task_queue.empty() or active_threads:
            # 1. 如果当前活跃线程小于限制，且队列有任务，就补充线程
            while len(active_threads) < concurrency_limit and not task_queue.empty():
                task = task_queue.get()
                result_dict = {}
                t = threading.Thread(target=inference_worker, args=(task, result_dict))
                t.task = task
                t.result_dict = result_dict
                t.start()
                active_threads.append(t)

            # 2. 检查是否有线程处理完毕
            done_threads = [t for t in active_threads if not t.is_alive()]
            for t in done_threads:
                active_threads.remove(t)
                status = t.result_dict.get('status', 'error')

                if status == 'oom':
                    # 【核心机制】显存爆炸，撤销并回退！
                    with print_lock:
                        print(f"    ⚠️ [OOM 拦截] {t.task['seg_name']} 导致显存溢出！撤回排队，并锁定最高并发数。")
                    concurrency_limit = max(1, concurrency_limit - 1)  # 降低并发
                    limit_locked = True  # 永久锁定并发数
                    task_queue.put(t.task)  # 把被 OOM 杀掉的任务重新塞回队列开头
                elif status == 'success':
                    completed_tasks += 1
                    # 【核心机制】游刃有余，尝试新增线程！
                    if not limit_locked and concurrency_limit < args.max_workers:
                        concurrency_limit += 1
                        with print_lock:
                            print(f"    🚀 [动态升频] 显存充裕，正在提升系统并发线程数至: {concurrency_limit}")

                    # 进度条平滑增长
                    scene_progress = completed_tasks / total_inference_tasks
                    current_pct = plan_start_pct + (pct_step * 0.2) + (pct_step * 0.8 * scene_progress)
                    report_progress("processing", current_pct)

                    # 清理成功跑完的临时垃圾
                    try:
                        shutil.rmtree(t.task['input_dir'])
                        shutil.rmtree(t.task['mask_dir'])
                    except:
                        pass
                else:
                    code = t.result_dict.get('code', 'Unknown')
                    with print_lock:
                        print(f"    ❌ [错误] 任务 {t.task['seg_name']} 执行失败 (Code: {code})")
                    completed_tasks += 1  # 失败也算走完了进度，防止死循环

            time.sleep(0.5)  # 防止主线程空转占用CPU

    # === Step 6: 结果回贴 ===
    print("\n=== Step 6: 合并修复结果到原图 ===")
    report_progress("merging", 90)
    final_frames_dir = os.path.join(args.workspace, "full_frames_final")
    setup_dirs(final_frames_dir)

    merge_worker_tasks = []
    for idx in sorted_indices:
        fname = frame_files_map[idx]
        relevant_plans = [t for t in all_merge_tasks if t['start_frame'] <= idx < t['end_frame']]
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