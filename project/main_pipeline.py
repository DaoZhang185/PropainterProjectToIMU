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
import re
import hashlib
import urllib.parse
import requests
from concurrent.futures import ThreadPoolExecutor

current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
if root_dir not in sys.path: sys.path.insert(0, root_dir)


class DualLogger(object):
    def __init__(self, filepath):
        self.terminal = sys.stdout
        self.log = open(filepath, "a", encoding='utf-8', buffering=1)

    def write(self, message):
        try:
            self.terminal.write(message);
            self.log.write(message)
        except Exception:
            pass

    def flush(self):
        try:
            self.terminal.flush();
            self.log.flush()
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
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL);
        return True
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
                all_x.extend([item[0], item[2]])
                all_y.extend([item[1], item[3]])
            elif isinstance(item, list) and isinstance(item[0], (list, tuple)):
                pts = np.array(item)
                all_x.extend(pts[:, 0])
                all_y.extend(pts[:, 1])
        if not all_x or not all_y: continue

        # 确立不可侵犯的基础原框
        bx1, by1 = max(0, min(all_x)), max(0, min(all_y))
        bx2, by2 = min(img_w, max(all_x)), min(img_h, max(all_y))

        # 初始的外扩框（如果没有任何碰撞的情况）
        cx1, cy1 = max(0, bx1 - padding), max(0, by1 - padding)
        cx2, cy2 = min(img_w, bx2 + padding), min(img_h, by2 + padding)

        plan_tr = {key: time_ranges[key]} if key in time_ranges else {}
        boxes.append({
            "keys": [key],
            "base_coords": [bx1, by1, bx2, by2],
            "crop_coords": [cx1, cy1, cx2, cy2],
            "original_poses": {key: coords_list},
            "time_ranges": plan_tr
        })

    # =================================================================
    # 【核心逻辑】：不合并区域！仅在外扩空间发生重叠时，从间隙的绝对中点砌墙截断
    # =================================================================
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            b1 = boxes[i]["base_coords"]
            b2 = boxes[j]["base_coords"]

            # 判断两个原始基准框在坐标轴上是否有物理交集
            x_overlap = not (b1[2] <= b2[0] or b1[0] >= b2[2])
            y_overlap = not (b1[3] <= b2[1] or b1[1] >= b2[3])

            # 1. 仅在 X 轴上分开（左右排列）
            if not x_overlap:
                if b1[2] <= b2[0]:  # box_i 在左，box_j 在右
                    mid_x = (b1[2] + b2[0]) / 2.0
                    boxes[i]["crop_coords"][2] = min(boxes[i]["crop_coords"][2], mid_x)
                    boxes[j]["crop_coords"][0] = max(boxes[j]["crop_coords"][0], mid_x)
                elif b2[2] <= b1[0]:  # box_j 在左，box_i 在右
                    mid_x = (b2[2] + b1[0]) / 2.0
                    boxes[j]["crop_coords"][2] = min(boxes[j]["crop_coords"][2], mid_x)
                    boxes[i]["crop_coords"][0] = max(boxes[i]["crop_coords"][0], mid_x)

            # 2. 仅在 Y 轴上分开（上下排列）
            if not y_overlap:
                if b1[3] <= b2[1]:  # box_i 在上，box_j 在下
                    mid_y = (b1[3] + b2[1]) / 2.0
                    boxes[i]["crop_coords"][3] = min(boxes[i]["crop_coords"][3], mid_y)
                    boxes[j]["crop_coords"][1] = max(boxes[j]["crop_coords"][1], mid_y)
                elif b2[3] <= b1[1]:  # box_j 在上，box_i 在下
                    mid_y = (b2[3] + b1[1]) / 2.0
                    boxes[j]["crop_coords"][3] = min(boxes[j]["crop_coords"][3], mid_y)
                    boxes[i]["crop_coords"][1] = max(boxes[i]["crop_coords"][1], mid_y)

            # 3. 如果 x_overlap 和 y_overlap 同时为真，说明用户画的红框本身就已经重合了。
            # 此时绝不干涉裁剪，任由它们自然外扩并生成两个交叠的任务图层，确保原区域不缺失。

    # 收尾工序：保证截断后的坐标依然有效，并满足 ProPainter 16像素边界要求
    for i in range(len(boxes)):
        cx1, cy1, cx2, cy2 = boxes[i]["crop_coords"]
        bx1, by1, bx2, by2 = boxes[i]["base_coords"]

        cx1, cy1, cx2, cy2 = int(cx1), int(cy1), int(cx2), int(cy2)

        # 安全断言：无论如何截断，外扩区域都绝对不能切到最初画好的红框内部！
        cx1 = max(0, min(cx1, int(bx1)))
        cy1 = max(0, min(cy1, int(by1)))
        cx2 = min(img_w, max(cx2, int(bx2)))
        cy2 = min(img_h, max(cy2, int(by2)))

        pad_w = (16 - ((cx2 - cx1) % 16)) % 16
        pad_h = (16 - ((cy2 - cy1) % 16)) % 16

        if cx2 + pad_w <= img_w:
            cx2 += pad_w
        else:
            cx1 = max(0, cx1 - pad_w)

        if cy2 + pad_h <= img_h:
            cy2 += pad_h
        else:
            cy1 = max(0, cy1 - pad_h)

        boxes[i]["crop_coords"] = [int(cx1), int(cy1), int(cx2), int(cy2)]

    return boxes


def auto_track_regions(all_poses, reference_times, frames_dir, frame_files, fps):
    # =================================================================
    # 【临时修改】展示期间强制跳过追踪算法！直接返回空字典。
    # 这样既省下了 Canny 边缘扫描的算力时间，又因为时间库为空，
    # 导致下游掩码渲染时直接无视时间限制，变为全局渲染生效！
    return {}
    # =================================================================

    time_ranges = {}
    print("\n[AutoTracker] 正在全景扫描动态掩码生效时间段...")
    for key, coords in all_poses.items():
        if 'other' in str(key) and key in reference_times:
            ref_time = reference_times[key]
            ref_frame_idx = max(0, min(int(ref_time * fps), len(frame_files) - 1))
            ref_img = cv2.imread(os.path.join(frames_dir, frame_files[ref_frame_idx]), cv2.IMREAD_GRAYSCALE)
            if ref_img is None: continue

            all_x, all_y = [], []
            for item in coords:
                if len(item) == 4 and isinstance(item[0], (int, float)):
                    all_x.extend([item[0], item[2]]);
                    all_y.extend([item[1], item[3]])
                elif isinstance(item, list):
                    pts = np.array(item)
                    all_x.extend(pts[:, 0]);
                    all_y.extend(pts[:, 1])

            if not all_x: continue
            x1, y1 = max(0, int(min(all_x))), max(0, int(min(all_y)))
            x2, y2 = min(ref_img.shape[1], int(max(all_x))), min(ref_img.shape[0], int(max(all_y)))

            template = ref_img[y1:y2, x1:x2]
            if template.size == 0 or template.shape[0] < 5 or template.shape[1] < 5: continue
            template_edges = cv2.Canny(template, 50, 150)

            active_frames = []
            for i, f_name in enumerate(frame_files):
                img = cv2.imread(os.path.join(frames_dir, f_name), cv2.IMREAD_GRAYSCALE)
                if img is None: continue
                roi = img[y1:y2, x1:x2]
                roi_edges = cv2.Canny(roi, 50, 150)
                score = np.mean(cv2.absdiff(template_edges, roi_edges))
                if score < 20.0: active_frames.append(i)

            ranges = []
            if active_frames:
                start, prev = active_frames[0], active_frames[0]
                for idx in active_frames[1:]:
                    if idx - prev > int(fps * 1.5):
                        ranges.append([start / fps, prev / fps])
                        start = idx
                    prev = idx
                ranges.append([start / fps, prev / fps])
            time_ranges[key] = ranges
            print(
                f"    ✓ {key} 自动检出 {len(ranges)} 个生效时段: {[[round(r[0], 1), round(r[1], 1)] for r in ranges]}")
    return time_ranges


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
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL);
        return True
    except:
        return False


# =================================================================
# 【核心新增】音频提取、机器翻译与蒙文字幕渲染挂载链路
# =================================================================
def get_md5(text):
    import uuid  # 防止报错
    return hashlib.md5(text.encode('utf-8')).hexdigest().upper()


def translate_to_mongolian(text, pid="YOUR_PID", appKey="YOUR_APPKEY"):
    """
    带超时重试机制的翻译接口调用
    """
    if not text.strip(): return text
    url = "https://oy.nmgoyun.com/api/fy/v1"
    timestamp = str(int(time.time() * 1000))
    import uuid
    nonce = uuid.uuid4().hex

    params = {
        "inputStr": text, "nonce": nonce, "pid": pid,
        "timestamp": timestamp, "type": "5", "appKey": appKey
    }

    sorted_keys = sorted(params.keys())
    temp_list = [f"{k}={urllib.parse.quote_plus(str(params[k]))}" for k in sorted_keys]
    sign = get_md5("&".join(temp_list))

    payload = {
        "inputStr": text, "nonce": nonce, "pid": pid,
        "sign": sign, "timestamp": timestamp, "type": 5
    }

    # 【修复1：翻译防漏防超时】增加 3次重试，超时时长放宽至 30秒
    max_retries = 3
    for attempt in range(max_retries):
        try:
            resp = requests.post(url, json=payload, timeout=30)
            res_json = resp.json()
            if res_json.get("code") == "0000":
                return res_json.get("data", text)
            else:
                print(f"    ⚠ [翻译拦截] 接口返回失败: {res_json}")
                return text
        except Exception as e:
            print(f"    ⚠ [翻译异常/超时] 第 {attempt + 1} 次请求失败: {e}，正在重试...")
            time.sleep(2)

    print(f"    ❌ [翻译彻底失败] 超过 {max_retries} 次仍无法连接奥云服务器，使用原中文字幕。")
    return text


def translate_to_mongolian(text, pid="YOUR_PID", appKey="YOUR_APPKEY"):
    # 【修复 3：防御性编程】内容、pid 或 appkey 为空时直接跳过，绝不浪费网络请求
    if not text.strip() or not pid or not appKey:
        return text

    url = "http://oy.nmgoyun.com/api/fy/v1"
    timestamp = str(int(time.time() * 1000))
    nonce = uuid.uuid4().hex

    # 【修复 2：类型一致性】明确 type 为整数 5
    sign_params = {
        "appKey": appKey,
        "inputStr": text,
        "nonce": nonce,
        "pid": pid,
        "timestamp": timestamp,
        "type": 5
    }

    # 【修复 1：严格签名机制】按字典序排序，直接拼接原始字符串，绝对不要 url_encode
    sorted_keys = sorted(sign_params.keys())
    sign_str = "&".join([f"{k}={sign_params[k]}" for k in sorted_keys])
    sign = hashlib.md5(sign_str.encode('utf-8')).hexdigest().upper()

    # 实际发送的 payload 不包含 appKey
    payload = {
        "inputStr": text,
        "nonce": nonce,
        "pid": pid,
        "sign": sign,
        "timestamp": timestamp,
        "type": 5
    }

    # 【修复 4：网络代理穿透】主动抓取系统的 http/https 代理环境变量
    proxies = {
        "http": os.environ.get("http_proxy") or os.environ.get("HTTP_PROXY"),
        "https": os.environ.get("https_proxy") or os.environ.get("HTTPS_PROXY")
    }

    max_retries = 3
    for attempt in range(max_retries):
        try:
            # 引入 proxies 参数，采用 (3秒连接, 5秒读取) 的快连快断策略
            resp = requests.post(url, json=payload, timeout=(3, 5), proxies=proxies)
            res_json = resp.json()

            if res_json.get("code") == "0000":
                return res_json.get("data", text)
            else:
                print(f"    ⚠ [翻译接口报错] 代码: {res_json.get('code')}, 信息: {res_json.get('message')}")
                return text

        except requests.exceptions.Timeout:
            print(f"    ⚠ [翻译超时] 第 {attempt + 1} 次请求超时，正在重试...")
        except Exception as e:
            print(f"    ⚠ [翻译异常] 第 {attempt + 1} 次请求失败: {e}，正在重试...")

        time.sleep(1)  # 失败缓冲 1 秒

    print(f"    ❌ [翻译彻底失败] 超过 {max_retries} 次仍无法连接奥云服务器，保留原中文字幕。")
    return text


def process_audio_and_subtitles(original_video, video_source, is_image_sequence, final_output_video, workspace,
                                subtitle_pos, img_w, img_h, print_lock, fps):
    def fallback_encode():
        if is_image_sequence:
            # 【修复2：视频打不开】在回退渲染中也加入 -pix_fmt yuv420p
            cmd = ['ffmpeg', '-y', '-f', 'image2', '-framerate', str(fps),
                   '-i', os.path.join(video_source, 'frame_%04d.png'),
                   '-i', original_video, '-vf', "pad=ceil(iw/2)*2:ceil(ih/2)*2",
                   '-map', '0:v:0', '-map', '1:a:0?',
                   '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-c:a', 'copy', '-shortest',
                   final_output_video]
            subprocess.run(cmd)  # 【修复3：解锁日志】删掉 DEVNULL
        else:
            shutil.copy(original_video, final_output_video)

    if not subtitle_pos:
        with print_lock: print("\n[Step 7] 未检测到蒙文字幕定位框，直接渲染最终纯净视频...")
        fallback_encode()
        return True

    subtitle_dir = os.path.join(workspace, "subtitle")
    setup_dirs(subtitle_dir)
    video_basename = os.path.splitext(os.path.basename(original_video))[0]

    with print_lock:
        print("\n[Step 7] 激活语音提取、翻译与蒙文字幕【单次高速压制】系统...")

    audio_path = os.path.join(subtitle_dir, f"{video_basename}.wav")
    with print_lock:
        print("    >> 提取视频音轨...")

    # 【修复3：解锁日志】让 FFmpeg 提取音频的日志可以直接打到控制台
    subprocess.run(
        ['ffmpeg', '-y', '-i', original_video, '-vn', '-acodec', 'pcm_s16le', '-ar', '16000', '-ac', '1', audio_path]
    )

    if not os.path.exists(audio_path):
        with print_lock: print("    ⚠ 原视频无声轨，跳过字幕生成。")
        fallback_encode()
        return False

    with print_lock:
        print("    >> 唤醒 Whisper 提取文字...")
    # 【修复3：解锁日志】解锁 Whisper 提取日志
    subprocess.run(['whisper', audio_path, '--model', 'turbo', '--output_format', 'srt', '--output_dir', subtitle_dir])

    final_srt = os.path.join(subtitle_dir, f"{video_basename}.srt")
    if not os.path.exists(final_srt):
        with print_lock: print("    ⚠ Whisper 生成失败，将输出无字幕视频。")
        fallback_encode()
        return False

    with print_lock:
        print("    >> 触发奥云翻译流与高阶 ASS 特效阵列装配...")
    mn_ass_path = os.path.join(subtitle_dir, f"{video_basename}_mn.ass")
    config_path = os.path.join(root_dir, "project", "api_configure", "api_configuration.json")
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            api_config = json.load(f)
        pid = api_config.get("pid", "")
        appkey = api_config.get("appKey", "")
    except Exception as e:
        pid, appkey = "", ""

    convert_srt_to_ass_vertical(final_srt, mn_ass_path, subtitle_pos, img_w, img_h, pid, appkey)

    with print_lock:
        print("    >> 挂载图片序列与翻译字幕，正在进行终极直出压制...")
    fonts_dir_relative = "fronts"
    ass_relative = os.path.relpath(mn_ass_path, current_dir).replace('\\', '/')

    if is_image_sequence:
        cmd = [
            'ffmpeg', '-y', '-f', 'image2', '-framerate', str(fps),
            '-i', os.path.join(video_source, 'frame_%04d.png'),
            '-i', original_video,
            '-vf', f"pad=ceil(iw/2)*2:ceil(ih/2)*2,ass='{ass_relative}':fontsdir='{fonts_dir_relative}'",
            '-map', '0:v:0', '-map', '1:a:0?',
            '-c:a', 'copy',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18', '-shortest',  # 【修复2：视频打不开】加入 yuv420p
            final_output_video
        ]
    else:
        cmd = [
            'ffmpeg', '-y', '-i', video_source,
            '-vf', f"ass='{ass_relative}':fontsdir='{fonts_dir_relative}'",
            '-c:a', 'copy',
            '-c:v', 'libx264', '-pix_fmt', 'yuv420p', '-crf', '18',  # 【修复2：视频打不开】加入 yuv420p
            final_output_video
        ]

    # 【修复3：解锁日志】用普通 run 运行，让 FFmpeg 的进度和报错完全暴露在你的 log 文件里
    subprocess.run(cmd)

    if os.path.exists(final_output_video):
        with print_lock:
            print(f"    ✓ 定点旋转字幕烧录成功！成品路径: {final_output_video}")
        return True
    else:
        with print_lock:
            print(f"    ✗ 字幕烧录失败，启用安全回退方案...")
        fallback_encode()
        return False


class GPUManager:
    def __init__(self, gpu_id, scale_delay_sec, cooldown_sec, max_absolute_workers=4):
        self.gpu_id = str(gpu_id);
        self.scale_delay_sec = scale_delay_sec;
        self.cooldown_sec = cooldown_sec;
        self.max_absolute_workers = max_absolute_workers
        self.concurrency_limit = 1;
        self.active_tasks = 0;
        self.last_scale_time = time.time();
        self.cooldown_until = 0

    def can_accept_task(self):
        return self.active_tasks < self.concurrency_limit

    def try_scale_up(self, print_lock):
        t = time.time()
        if t > self.cooldown_until and self.active_tasks == self.concurrency_limit:
            if t - self.last_scale_time > self.scale_delay_sec:
                if self.concurrency_limit < self.max_absolute_workers:
                    self.concurrency_limit += 1;
                    self.last_scale_time = t
                    with print_lock: print(f"    🚀 [扩容] GPU {self.gpu_id} 并发上限提升至: {self.concurrency_limit}")

    def handle_oom(self, print_lock):
        self.concurrency_limit = max(1, self.active_tasks);
        self.cooldown_until = time.time() + self.cooldown_sec;
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

    try:
        cmd = ['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_entries', 'stream=avg_frame_rate', '-of',
               'default=noprint_wrappers=1:nokey=1', args.video]
        res = subprocess.run(cmd, capture_output=True, text=True)
        vals = res.stdout.strip().split('/')
        fps = float(vals[0]) / float(vals[1]) if len(vals) == 2 else float(vals[0])
    except:
        fps = 25.0
    print(f"[Main] 视频原始帧率检测为: {fps:.2f} FPS")

    img_h, img_w = cv2.imread(os.path.join(frames_dir, frame_files[0])).shape[:2]

    # 【获取蒙文字幕放置位置】
    all_poses, reference_times, subtitle_pos = load_poses_from_json(args.mask_json)

    if all_poses:
        time_ranges = auto_track_regions(all_poses, reference_times, frames_dir, frame_files, fps)
        work_plans = resolve_overlaps(all_poses, padding=args.padding, img_w=img_w, img_h=img_h,
                                      time_ranges=time_ranges)

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

            scene_indices = detect_scenes_from_folder(crop_frames_dir, threshold=20.0)
            if 0 not in scene_indices: scene_indices.insert(0, 0)
            if len(frame_files) not in scene_indices: scene_indices.append(len(frame_files))
            scene_indices = sorted(list(set(scene_indices)))

            mask_output_dir = os.path.join(region_dir, "masks")
            generate_local_masks(crop_frames_dir, mask_output_dir, plan['original_poses'], plan['crop_coords'],
                                 time_ranges=plan['time_ranges'], fps=fps)

            results_dir = os.path.join(region_dir, "results")
            for i in range(len(scene_indices) - 1):
                for seg_start, seg_end in get_recursive_segments(scene_indices[i], scene_indices[i + 1],
                                                                 args.max_frames):
                    seg_name = f"seg_{seg_start:06d}_{seg_end:06d}"
                    seg_out_dir = os.path.join(results_dir, seg_name)

                    in_dir = os.path.join(region_dir, "temp_inputs", seg_name)
                    mk_dir = os.path.join(region_dir, "temp_masks", seg_name)
                    setup_dirs(in_dir);
                    setup_dirs(mk_dir)

                    valid = 0;
                    has_white_mask = False
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

                        if not has_white_mask:
                            check_path = src_m if os.path.exists(src_m) else (src_s if os.path.exists(src_s) else None)
                            if check_path:
                                img_chk = cv2.imread(check_path, cv2.IMREAD_GRAYSCALE)
                                if img_chk is not None and cv2.countNonZero(img_chk) > 0: has_white_mask = True
                        valid += 1

                    if valid > 0:
                        if has_white_mask:
                            global_task_queue.put(
                                {'region': region_id, 'seg_name': seg_name, 'in_dir': in_dir, 'mk_dir': mk_dir,
                                 'out_dir': seg_out_dir})
                        all_merge_tasks.append(
                            {'result_dir': seg_out_dir, 'coords': plan['crop_coords'], 'start_frame': seg_start,
                             'end_frame': seg_end})

        total_tasks = global_task_queue.qsize()
        print(f"[Main] 预处理完成！算力豁免后总计 {total_tasks} 个待处理片段进入 GPU 队列。")

        print("\n[Step 3] 启动多 GPU AIMD 调度引擎...")
        gpu_managers = [GPUManager(g, args.scale_delay * 60, args.cooldown * 60, args.max_workers_per_gpu) for g in
                        gpu_list]
        active_threads = [];
        print_lock = threading.Lock();
        completed_tasks = 0

        def inference_worker(task, gpu_manager, result_dict):
            cmd = [sys.executable, args.model_path, "--video", task['in_dir'], "--mask", task['mk_dir'], "--output",
                   task['out_dir'], "--fp16", "--mask_dilation", "4", "--flow_mask_dilation", "20", "--raft_iter", "20",
                   "--ref_stride", "10", "--subvideo_length", "40"]
            env = os.environ.copy();
            env['CUDA_VISIBLE_DEVICES'] = gpu_manager.gpu_id
            tag = f"[GPU {gpu_manager.gpu_id} | {task['region']} | {task['seg_name']}]"
            try:
                process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           cwd=current_dir, universal_newlines=True, errors='replace')
                is_oom = False
                while True:
                    line = process.stdout.readline()
                    if line == '' and process.poll() is not None: break
                    if line:
                        l = line.strip()
                        if "CUDA out of memory" in l or "RuntimeError: CUDA" in l: is_oom = True
                if is_oom or process.returncode == 137:
                    result_dict['status'] = 'oom'
                elif process.returncode != 0:
                    result_dict['status'] = 'error'
                else:
                    mp4 = os.path.join(task['out_dir'], "inference_output.mp4")
                    if os.path.exists(mp4):
                        frm_dir = os.path.join(task['out_dir'], "frames")
                        os.makedirs(frm_dir, exist_ok=True)
                        subprocess.run(
                            ['ffmpeg', '-y', '-i', mp4, '-start_number', '0', os.path.join(frm_dir, '%04d.png')],
                            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
                    result_dict['status'] = 'success'
            except Exception as e:
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
                    t.gm.handle_oom(print_lock);
                    global_task_queue.put(t.task)
                else:
                    completed_tasks += 1
                    if total_tasks > 0: report_progress("video_processing", 15 + (completed_tasks / total_tasks) * 75)
                    try:
                        shutil.rmtree(t.task['in_dir']); shutil.rmtree(t.task['mk_dir'])
                    except:
                        pass
            time.sleep(0.5)

        print("\n[Step 6] 图片合并...")
        report_progress("finalizing", 90)
        final_dir = os.path.join(args.workspace, "full_frames_final")
        setup_dirs(final_dir)
        m_tasks = [
            (i, f, frames_dir, final_dir, [pl for pl in all_merge_tasks if pl['start_frame'] <= i < pl['end_frame']])
            for i, f in enumerate(frame_files)]
        with ThreadPoolExecutor(max_workers=8) as ex:
            list(ex.map(merge_worker, m_tasks))

        # 【核心优化】不生成 temp_clean_video.mp4，直接把图片文件夹交给下一步压制
        report_progress("subtitle_burning", 95)
        video_source = final_dir
        is_image_sequence = True
    else:
        # 如果根本没画水印消除区域
        print("\n[INFO] 未检测到去水印标注，将使用原视频直接进入字幕处理流程。")
        video_source = args.video
        is_image_sequence = False

        # 【终极工序】字幕与翻译系统 (接收图片流，单次压制，性能翻倍！)
    final_output = os.path.join(args.workspace, args.output_video)
    process_audio_and_subtitles(args.video, video_source, is_image_sequence, final_output, args.workspace, subtitle_pos,
                                img_w, img_h, threading.Lock(), fps)

    report_progress("completed", 100)


if __name__ == "__main__":
    main()