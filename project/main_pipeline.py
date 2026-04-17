import sys
import os
import argparse
import json
import shutil
import cv2
import numpy as np
import subprocess
import glob
import datetime
import time
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)


class DualLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding='utf-8', buffering=1)

    def write(self, message):
        try:
            self.terminal.write(message); self.log.write(message)
        except Exception:
            pass

    def flush(self):
        try:
            self.terminal.flush(); self.log.flush()
        except Exception:
            pass


try:
    from scene_detector import detect_scenes_from_folder
    from make_mask import generate_local_masks
    from mask_loader import load_poses_from_json
except ImportError as e:
    print(f"CRITICAL ERROR: 缺少依赖脚本 ({e})。")
    sys.exit(1)


def report_progress(stage, percent):
    percent = max(0, min(100, int(percent)))
    print(f"PROGRESS:{stage}:{percent}", flush=True)


def setup_dirs(base_path):
    if not os.path.exists(base_path): os.makedirs(base_path, exist_ok=True)


def extract_frames_ffmpeg(video_path, output_dir):
    if not os.path.exists(video_path): return False
    if os.path.exists(output_dir): shutil.rmtree(output_dir)
    setup_dirs(output_dir)
    cmd = ['ffmpeg', '-i', video_path, '-start_number', '0', '-vsync', '0', '-q:v', '2',
           os.path.join(output_dir, 'frame_%04d.png')]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); return True
    except:
        return False


def get_recursive_segments(start_idx, end_idx, max_frames=300):
    length = end_idx - start_idx
    if length <= max_frames: return [(start_idx, end_idx)]
    half = length // 2
    return get_recursive_segments(start_idx, start_idx + half + (length % 2), max_frames) + \
        get_recursive_segments(start_idx + half + (length % 2), end_idx, max_frames)


def resolve_overlaps(all_poses, padding=150, img_w=1920, img_h=1080, time_ranges=None):
    if time_ranges is None: time_ranges = {}
    boxes = []
    for key, coords_list in all_poses.items():
        if not coords_list: continue
        all_x, all_y = [], []
        for item in coords_list:
            if len(item) == 4 and isinstance(item[0], (int, float)):
                all_x.extend([item[0], item[2]]);
                all_y.extend([item[1], item[3]])
            elif isinstance(item, list) and isinstance(item[0], (list, tuple)):
                pts = np.array(item)
                all_x.extend(pts[:, 0]);
                all_y.extend(pts[:, 1])
        if not all_x or not all_y: continue

        nx1, ny1 = max(0, min(all_x) - padding), max(0, min(all_y) - padding)
        nx2, ny2 = min(img_w, max(all_x) + padding), min(img_h, max(all_y) + padding)

        # 捆绑时间信息，方便合并不丢
        plan_tr = {key: time_ranges[key]} if key in time_ranges else {}
        boxes.append({"keys": [key], "crop_coords": [int(nx1), int(ny1), int(nx2), int(ny2)],
                      "original_poses": {key: coords_list}, "time_ranges": plan_tr})

    adjusted, iterations = True, 0
    while adjusted and iterations < 10:
        adjusted = False;
        iterations += 1
        for i in range(len(boxes)):
            cx1, cy1, cx2, cy2 = boxes[i]["crop_coords"]
            for j in range(i + 1, len(boxes)):
                ox1, oy1, ox2, oy2 = boxes[j]["crop_coords"]
                if not (cx2 <= ox1 or cx1 >= ox2 or cy2 <= oy1 or cy1 >= oy2):
                    if abs((cx1 + cx2) / 2 - (ox1 + ox2) / 2) > abs((cy1 + cy2) / 2 - (oy1 + oy2) / 2):
                        if cx1 < ox1:
                            mid = (ox1 + cx2) // 2; cx2 = mid; ox1 = mid
                        else:
                            mid = (cx1 + ox2) // 2; cx1 = mid; ox2 = mid
                    else:
                        if cy1 < oy1:
                            mid = (oy1 + cy2) // 2; cy2 = mid; oy1 = mid
                        else:
                            mid = (cy1 + oy2) // 2; cy1 = mid; oy2 = mid
                    boxes[i]["crop_coords"], boxes[j]["crop_coords"] = [cx1, cy1, cx2, cy2], [ox1, oy1, ox2, oy2]
                    boxes[i]["keys"].extend(boxes[j]["keys"])
                    boxes[i]["original_poses"].update(boxes[j]["original_poses"])
                    boxes[i]["time_ranges"].update(boxes[j]["time_ranges"])
                    adjusted = True

    for i in range(len(boxes)):
        nx1, ny1, nx2, ny2 = boxes[i]["crop_coords"]
        pad_w, pad_h = (16 - ((nx2 - nx1) % 16)) % 16, (16 - ((ny2 - ny1) % 16)) % 16
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
    src, dst, coords = args
    img = cv2.imread(src)
    if img is not None: cv2.imwrite(dst, img[coords[1]:coords[3], coords[0]:coords[2]])


def merge_worker(args):
    idx, fname, full_dir, final_dir, merge_plans = args
    base_img = cv2.imread(os.path.join(full_dir, fname))
    if base_img is None: return
    for plan in merge_plans:
        r_path = os.path.join(plan['result_dir'], "frames", f"{(idx - plan['start_frame']):04d}.png")
        if os.path.exists(r_path):
            crop = cv2.imread(r_path)
            if crop is not None:
                x1, y1, x2, y2 = map(int, plan['coords'])
                if crop.shape[:2] != (y2 - y1, x2 - x1): crop = cv2.resize(crop, (x2 - x1, y2 - y1))
                base_img[y1:y2, x1:x2] = crop
    cv2.imwrite(os.path.join(final_dir, fname), base_img)


def render_video(frames_dir, source_video, output_path, fps=25.0):
    cmd = ['ffmpeg', '-y', '-f', 'image2', '-framerate', str(fps), '-i', os.path.join(frames_dir, 'frame_%04d.png'),
           '-i', source_video, '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2", '-c:v', 'mpeg4', '-q:v', '2',
           '-map', '0:v:0', '-map', '1:a:0?', '-c:a', 'copy', '-shortest', output_path]
    try:
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL); return True
    except:
        return False


class GPUManager:
    def __init__(self, gpu_id, scale_delay_sec, cooldown_sec, max_absolute_workers=4):
        self.gpu_id = str(gpu_id)
        self.scale_delay_sec = scale_delay_sec
        self.cooldown_sec = cooldown_sec
        self.max_absolute_workers = max_absolute_workers
        self.concurrency_limit = 1
        self.active_tasks = 0
        self.last_scale_time = time.time()
        self.cooldown_until = 0

    def can_accept_task(self):
        return self.active_tasks < self.concurrency_limit

    def try_scale_up(self, print_lock):
        t = time.time()
        if t > self.cooldown_until and self.active_tasks == self.concurrency_limit:
            if t - self.last_scale_time > self.scale_delay_sec:
                if self.concurrency_limit < self.max_absolute_workers:
                    self.concurrency_limit += 1
                    self.last_scale_time = t
                    with print_lock: print(f"    🚀 [扩容] GPU {self.gpu_id} 并发上限提升至: {self.concurrency_limit}")

    def handle_oom(self, print_lock):
        self.concurrency_limit = max(1, self.active_tasks)
        self.cooldown_until = time.time() + self.cooldown_sec
        self.last_scale_time = time.time()
        with print_lock: print(
            f"    ⚠️ [OOM] GPU {self.gpu_id} 锁定并发数为 {self.concurrency_limit}，冷却 {self.cooldown_sec // 60} 分钟。")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--video', required=True)
    parser.add_argument('--mask_json', required=True)
    parser.add_argument('--workspace', required=True)
    parser.add_argument('--model_path', default='inference_propainter.py')
    parser.add_argument('--output_video', default='final_output.mp4')
    parser.add_argument('--max_frames', type=int, default=300)
    parser.add_argument('--padding', type=int, default=150)
    parser.add_argument('--gpus', default='0')
    parser.add_argument('--scale_delay', type=int, default=3)
    parser.add_argument('--cooldown', type=int, default=10)
    parser.add_argument('--max_workers_per_gpu', type=int, default=4)
    args = parser.parse_args()

    if not os.path.isabs(args.model_path): args.model_path = os.path.join(current_dir, args.model_path)

    video_name = os.path.splitext(os.path.basename(args.video))[0]
    args.workspace = os.path.join(args.workspace, video_name,
                                  f"{video_name}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}")
    setup_dirs(args.workspace)
    sys.stdout = DualLogger(os.path.join(args.workspace, "pipeline_run.log"))
    sys.stderr = sys.stdout

    gpu_list = [g.strip() for g in args.gpus.split(',')]
    print(f"\n[Main] ====== 高级 AIMD 时效水印调度启动 ======")

    frames_dir = os.path.join(args.workspace, "full_frames_original")
    extract_frames_ffmpeg(args.video, frames_dir)
    frame_files = sorted([f for f in os.listdir(frames_dir) if f.endswith('.png')])
    fps = 25.0
    img_h, img_w = cv2.imread(os.path.join(frames_dir, frame_files[0])).shape[:2]

    all_poses, time_ranges = load_poses_from_json(args.mask_json)
    if not all_poses: sys.exit(1)
    work_plans = resolve_overlaps(all_poses, padding=args.padding, img_w=img_w, img_h=img_h, time_ranges=time_ranges)

    global_task_queue = queue.Queue()
    all_merge_tasks = []

    print("\n[Step 2] 正在进行全量场景切分与预处理...")
    for plan in work_plans:
        region_id = "_".join(plan['keys'])
        region_dir = os.path.join(args.workspace, f"region_{region_id}")
        crop_frames_dir = os.path.join(region_dir, "frames")
        setup_dirs(crop_frames_dir)

        tasks = [(os.path.join(frames_dir, f), os.path.join(crop_frames_dir, f), plan['crop_coords']) for f in
                 frame_files]
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(crop_worker, tasks))

        scene_indices = detect_scenes_from_folder(crop_frames_dir, threshold=6.0)
        if 0 not in scene_indices: scene_indices.insert(0, 0)
        if len(frame_files) not in scene_indices: scene_indices.append(len(frame_files))
        scene_indices = sorted(list(set(scene_indices)))

        mask_output_dir = os.path.join(region_dir, "masks")
        # 传入 time_ranges 生成精准 Mask
        generate_local_masks(crop_frames_dir, mask_output_dir, plan['original_poses'], plan['crop_coords'],
                             time_ranges=plan['time_ranges'])

        results_dir = os.path.join(region_dir, "results")
        for i in range(len(scene_indices) - 1):
            for seg_start, seg_end in get_recursive_segments(scene_indices[i], scene_indices[i + 1], args.max_frames):
                seg_name = f"seg_{seg_start:06d}_{seg_end:06d}"
                seg_out_dir = os.path.join(results_dir, seg_name)

                in_dir = os.path.join(region_dir, "temp_inputs", seg_name)
                mk_dir = os.path.join(region_dir, "temp_masks", seg_name)
                setup_dirs(in_dir);
                setup_dirs(mk_dir)

                valid = 0
                has_white_mask = False  # 👈 核心检测位

                for f_idx in range(seg_start, seg_end):
                    if f_idx >= len(frame_files): break
                    f_name = frame_files[f_idx]
                    os.symlink(os.path.abspath(os.path.join(crop_frames_dir, f_name)), os.path.join(in_dir, f_name))
                    src_m = os.path.abspath(os.path.join(mask_output_dir, f_name))
                    src_s = os.path.abspath(os.path.join(mask_output_dir, "static_mask.png"))

                    if os.path.exists(src_m):
                        os.symlink(src_m, os.path.join(mk_dir, f_name))
                    elif os.path.exists(src_s):
                        os.symlink(src_s, os.path.join(mk_dir, f_name))

                    # 检查此切片中是否有需要处理的非全黑遮罩
                    if not has_white_mask:
                        check_path = src_m if os.path.exists(src_m) else (src_s if os.path.exists(src_s) else None)
                        if check_path:
                            img_chk = cv2.imread(check_path, cv2.IMREAD_GRAYSCALE)
                            if img_chk is not None and cv2.countNonZero(img_chk) > 0:
                                has_white_mask = True
                    valid += 1

                if valid > 0:
                    # ✅ 如果在这个短视频切片中，由于时间限制所有的帧 Mask 全黑，则直接跳过 GPU 处理！
                    if has_white_mask:
                        global_task_queue.put(
                            {'region': region_id, 'seg_name': seg_name, 'in_dir': in_dir, 'mk_dir': mk_dir,
                             'out_dir': seg_out_dir})
                    else:
                        print(f"    ℹ [{region_id} | {seg_name}] 无生效遮罩(全黑)，已自动跳过深度推理以节省算力。")

                    # 但无论如何必须保留合并队列，以保证视频最后渲染帧率不断
                    all_merge_tasks.append(
                        {'result_dir': seg_out_dir, 'coords': plan['crop_coords'], 'start_frame': seg_start,
                         'end_frame': seg_end})

    total_tasks = global_task_queue.qsize()
    print(f"[Main] 预处理完成！过滤后总计 {total_tasks} 个有效片段进入 GPU 队列。")

    print("\n[Step 3] 启动多 GPU AIMD 调度引擎...")
    gpu_managers = [GPUManager(g, args.scale_delay * 60, args.cooldown * 60, args.max_workers_per_gpu) for g in
                    gpu_list]
    active_threads = []
    print_lock = threading.Lock()
    completed_tasks = 0

    def inference_worker(task, gpu_manager, result_dict):
        cmd = [
            sys.executable, args.model_path,
            "--video", task['in_dir'], "--mask", task['mk_dir'], "--output", task['out_dir'],
            "--fp16", "--mask_dilation", "4", "--flow_mask_dilation", "20",
            "--raft_iter", "20", "--ref_stride", "10", "--subvideo_length", "30"
        ]

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu_manager.gpu_id
        tag = f"[GPU {gpu_manager.gpu_id} | {task['region']} | {task['seg_name']}]"

        with print_lock:
            print(f"    分配任务 -> {tag}")

        try:
            process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, cwd=current_dir,
                                       universal_newlines=True, errors='replace')
            is_oom = False
            while True:
                line = process.stdout.readline()
                if line == '' and process.poll() is not None: break
                if line:
                    l = line.strip()
                    if "CUDA out of memory" in l or "RuntimeError: CUDA" in l:
                        is_oom = True
                    elif "Processing" in l or "%|" in l:
                        with print_lock:
                            print(f"    {tag} {l}")

            if is_oom or process.returncode == 137:
                result_dict['status'] = 'oom'
            elif process.returncode != 0:
                result_dict['status'] = 'error'
            else:
                mp4 = os.path.join(task['out_dir'], "inference_output.mp4")
                if os.path.exists(mp4):
                    frm_dir = os.path.join(task['out_dir'], "frames")
                    os.makedirs(frm_dir, exist_ok=True)
                    subprocess.run(['ffmpeg', '-y', '-i', mp4, '-start_number', '0', os.path.join(frm_dir, '%04d.png')],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                result_dict['status'] = 'success'
        except Exception as e:
            with print_lock:
                print(f"    异常 {tag} -> {e}")
            result_dict['status'] = 'error'

    while not global_task_queue.empty() or active_threads:
        for gm in gpu_managers:
            gm.try_scale_up(print_lock)
            while gm.can_accept_task() and not global_task_queue.empty():
                task = global_task_queue.get()
                gm.active_tasks += 1
                res = {}
                t = threading.Thread(target=inference_worker, args=(task, gm, res))
                t.task = task;
                t.gm = gm;
                t.res = res
                t.start()
                active_threads.append(t)

        done = [t for t in active_threads if not t.is_alive()]
        for t in done:
            active_threads.remove(t)
            t.gm.active_tasks -= 1
            status = t.res.get('status', 'error')

            if status == 'oom':
                t.gm.handle_oom(print_lock)
                global_task_queue.put(t.task)
            else:
                completed_tasks += 1
                if total_tasks > 0:
                    report_progress("processing", 15 + (completed_tasks / total_tasks) * 75)
                try:
                    shutil.rmtree(t.task['in_dir']); shutil.rmtree(t.task['mk_dir'])
                except:
                    pass
        time.sleep(0.5)

    print("\n[Step 6] 图片合并...")
    report_progress("merging", 90)
    final_dir = os.path.join(args.workspace, "full_frames_final")
    setup_dirs(final_dir)
    m_tasks = []
    for i, f in enumerate(frame_files):
        p = [pl for pl in all_merge_tasks if pl['start_frame'] <= i < pl['end_frame']]
        m_tasks.append((i, f, frames_dir, final_dir, p))
    with ThreadPoolExecutor(max_workers=8) as ex:
        list(ex.map(merge_worker, m_tasks))

    print("\n[Step 7] 渲染视频...")
    report_progress("rendering", 95)
    render_video(final_dir, args.video, os.path.join(args.workspace, args.output_video), fps)
    report_progress("completed", 100)


if __name__ == "__main__":
    main()