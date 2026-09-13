import os
import sys
import argparse
import json
import threading
import queue
import time
import shutil

# 复用主程序的组件，避免重复造轮子
from main_pipeline import GPUManager, merge_worker, setup_dirs, report_progress


def main():
    parser = argparse.ArgumentParser(description="智能视频清洗 - 断点续传调度器")
    parser.add_argument('--workspace', required=True, help="之前运行失败的 workspace 目录")
    parser.add_argument('--gpus', default='0', help="本次补救使用的 GPU")
    parser.add_argument('--model_path', default='inference_propainter.py')
    parser.add_argument('--max_workers_per_gpu', type=int, default=1, help="建议 OOM 补救时降低并发")
    parser.add_argument('--scale_delay', type=int, default=3)
    parser.add_argument('--cooldown', type=int, default=10)
    args = parser.parse_args()

    manifest_path = os.path.join(args.workspace, "task_manifest.json")
    if not os.path.exists(manifest_path):
        print(f"❌ 找不到清单文件: {manifest_path}。无法执行断点续传。")
        return

    # 读取并筛选出非 success 的任务
    with open(manifest_path, 'r', encoding='utf-8') as f:
        manifest_data = json.load(f)

    pending_queue = queue.Queue()
    for task_id, info in manifest_data.items():
        if info.get('status') != 'success':
            pending_queue.put(info['task_data'])

    total_tasks = pending_queue.qsize()
    if total_tasks == 0:
        print("✅ 检测到所有切片均已处理成功，无需重新推理！您可以直接执行图片合并步骤。")
        return

    print(f"🚀 [Resume] 发现 {total_tasks} 个失败/未完成的片段，准备重新加载推理引擎...")

    gpu_list = [g.strip() for g in args.gpus.split(',')]
    gpu_managers = [GPUManager(g, args.scale_delay * 60, args.cooldown * 60, args.max_workers_per_gpu) for g in
                    gpu_list]
    active_threads = []
    print_lock = threading.Lock()
    manifest_lock = threading.Lock()
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # 完整复刻带日志的 Worker
    def inference_worker_resume(task, gpu_manager, result_dict):
        cmd = [sys.executable, args.model_path, "--video", task['in_dir'], "--mask", task['mk_dir'], "--output",
               task['out_dir'], "--fp16", "--mask_dilation", "4", "--flow_mask_dilation", "20", "--raft_iter", "20",
               "--ref_stride", "10", "--subvideo_length", "60"]

        env = os.environ.copy()
        env['CUDA_VISIBLE_DEVICES'] = gpu_manager.gpu_id
        log_path = os.path.join(task['out_dir'], "inference_detailed_resume.log")  # 独立续传日志

        import subprocess
        import datetime
        with open(log_path, 'w', encoding='utf-8') as log_f:
            log_f.write(f"--- 续传启动 | GPU: {gpu_manager.gpu_id} | 时间: {datetime.datetime.now()} ---\n")
            try:
                process = subprocess.Popen(cmd, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           cwd=current_dir, universal_newlines=True, errors='replace')
                is_oom = False
                while True:
                    line = process.stdout.readline()
                    if line == '' and process.poll() is not None: break
                    if line:
                        log_f.write(line);
                        log_f.flush()
                        if "CUDA out of memory" in line or "RuntimeError: CUDA" in line: is_oom = True

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

        with manifest_lock:
            try:
                with open(manifest_path, 'r', encoding='utf-8') as mf:
                    data = json.load(mf)
                data[task['task_id']]['status'] = result_dict['status']
                with open(manifest_path, 'w', encoding='utf-8') as mf:
                    json.dump(data, mf, indent=4)
            except:
                pass

    # 开始消费队列
    while not pending_queue.empty() or active_threads:
        for gm in gpu_managers:
            gm.try_scale_up(print_lock)
            while gm.can_accept_task() and not pending_queue.empty():
                task = pending_queue.get()
                gm.active_tasks += 1
                res = {}
                t = threading.Thread(target=inference_worker_resume, args=(task, gm, res))
                t.task = task;
                t.gm = gm;
                t.res = res
                t.start()
                active_threads.append(t)

        done = [t for t in active_threads if not t.is_alive()]
        for t in done:
            active_threads.remove(t)
            t.gm.active_tasks -= 1
            if t.res.get('status') == 'oom':
                t.gm.handle_oom(print_lock)
                pending_queue.put(t.task)  # 回收失败任务
        time.sleep(0.5)

    print("\n✅ 所有失败片段补救完成！您可以重新运行主程序的合并逻辑，或手动触发合并脚本。")


if __name__ == "__main__":
    main()