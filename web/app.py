from flask import Flask, render_template, request, jsonify, send_file
from flask_cors import CORS
import os
import sys
import logging
from werkzeug.utils import secure_filename
import threading
from datetime import datetime
import uuid
import json
import glob
import cv2
import base64
import subprocess
from sam_helper import SAMService

app = Flask(__name__)
CORS(app)

# =================================================================
# 【核心改造】全自动路径解析 (不再写死任何绝对路径)
# =================================================================
# WEB_DIR 就是当前 app.py 所在的文件夹 (web)
WEB_DIR = os.path.dirname(os.path.abspath(__file__))
# PROPAINTER_ROOT 就是 WEB_DIR 的上一级 (ProPainter)
PROPAINTER_ROOT = os.path.dirname(WEB_DIR)
# PROJECT_DIR 就是后端的目录 (project)
PROJECT_DIR = os.path.join(PROPAINTER_ROOT, 'project')

app.config['SECRET_KEY'] = 'your-secret-key-change-in-production'
app.config['UPLOAD_FOLDER'] = os.path.join(PROPAINTER_ROOT, 'uploads')
app.config['VIDEO_FOLDER'] = os.path.join(PROPAINTER_ROOT, 'videos')
app.config['RESULTS_FOLDER'] = os.path.join(PROPAINTER_ROOT, 'results')
app.config['MAX_CONTENT_LENGTH'] = 2000 * 1024 * 1024

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['VIDEO_FOLDER'], exist_ok=True)
os.makedirs(app.config['RESULTS_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# 初始化 SAM 服务 (全局单例，启动时加载)
SAM_WEIGHT_PATH = os.path.join(PROPAINTER_ROOT, 'weights', 'sam_vit_b_01ec64.pth')
try:
    sam_service = SAMService(checkpoint_path=SAM_WEIGHT_PATH)
except Exception as e:
    logger.error(f"SAM 加载失败: {e}")
    sam_service = None

tasks = {}


class ProcessingTask:
    def __init__(self, task_id, script_params):
        self.task_id = task_id
        self.script_params = script_params
        self.status = "pending"
        self.progress = 0
        self.current_stage = ""
        self.logs = []
        self.start_time = None
        self.end_time = None
        self.thread = None
        self.final_video_path = None

        # 将前端传来的纯文件名，拼接成本地绝对路径
        self.source_video = os.path.join(app.config['VIDEO_FOLDER'],
                                         os.path.basename(self.script_params['source_video']))
        self.mask_file = os.path.join(app.config['UPLOAD_FOLDER'], os.path.basename(self.script_params['mask_file']))
        self.workspace = app.config['RESULTS_FOLDER']

        logger.info(f"创建新任务对象: {task_id}")

    def start(self):
        self.status = "running"
        self.start_time = datetime.now()
        self.thread = threading.Thread(target=self._run_task)
        self.thread.daemon = True
        self.thread.start()

    def _run_task(self):
        try:
            self._add_log("准备本地环境，开始处理...")

            # 【这是针对你 5台机器、4显卡 的推荐启动配置】
            cmd = [
                sys.executable,
                os.path.join(PROJECT_DIR, 'main_pipeline.py'),
                '--video', self.source_video,
                '--mask_json', self.mask_file,
                '--workspace', self.workspace,
                '--padding', '150',  # 保持高外扩，防重叠算法会保驾护航
                '--gpus', '0,1,2,3',  # 明确告诉它使用这 4 张显卡
                '--scale_delay', '3',  # 稳定处理 3 分钟后，尝试在某张卡上加第二个任务
                '--cooldown', '10'  # 如果某张卡爆显存，该卡 10 分钟内禁止新增并发
            ]

            self._add_log(f"执行命令: {' '.join(cmd)}")

            # 使用 Popen 直接在本地运行，替代 SSH
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=PROJECT_DIR  # 切换到 project 目录下执行
            )

            # 实时读取日志
            while True:
                line = process.stdout.readline()
                if line == '' and process.poll() is not None:
                    break
                if line.strip():
                    if self.status == "cancelled":
                        process.terminate()
                        break

                    clean_line = line.strip()
                    if clean_line.startswith("PROGRESS:"):
                        try:
                            parts = clean_line.split(":")
                            if len(parts) >= 3:
                                self.current_stage = parts[1]
                                self.progress = int(parts[2])
                        except:
                            pass
                    else:
                        self._add_log(clean_line)

            return_code = process.wait()

            if self.status == "cancelled":
                self._add_log("任务已被用户取消")
            elif return_code == 0:
                self.status = "completed"
                self.progress = 100
                self._find_final_video()
                self._add_log(f"✓ 完成. 视频: {self.final_video_path}")
            else:
                self.status = "failed"
                self._add_log(f"✗ 脚本执行异常，退出码: {return_code}")

        except Exception as e:
            self.status = "failed"
            self._add_log(f"执行失败: {str(e)}")
            logger.error(str(e))
        finally:
            self.end_time = datetime.now()

    def _find_final_video(self):
        try:
            # 去 workspace 找最新的 final_output.mp4
            files = glob.glob(os.path.join(self.workspace, '**', 'final_output.mp4'), recursive=True)
            if files:
                self.final_video_path = max(files, key=os.path.getmtime)
        except:
            pass

    def _add_log(self, msg):
        self.logs.append({'timestamp': datetime.now().strftime("%H:%M:%S"), 'message': msg})

    def cancel(self):
        self.status = "cancelled"


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/api/tasks', methods=['POST'])
def create_task():
    data = request.json
    task_id = str(uuid.uuid4())
    # 移除了 server_config，仅传入参数
    tasks[task_id] = ProcessingTask(task_id, data['script_params'])
    tasks[task_id].start()
    return jsonify({'task_id': task_id, 'status': 'started'})


@app.route('/api/tasks/<task_id>', methods=['GET'])
def get_task(task_id):
    if task_id not in tasks: return jsonify({'error': 'Not found'}), 404
    t = tasks[task_id]
    return jsonify({'status': t.status, 'progress': t.progress, 'current_stage': t.current_stage, 'logs': t.logs[-50:],
                    'final_video_path': t.final_video_path})


@app.route('/api/tasks/<task_id>/cancel', methods=['POST'])
def cancel_task(task_id):
    if task_id in tasks: tasks[task_id].cancel()
    return jsonify({'status': 'cancelled'})


@app.route('/api/upload', methods=['POST'])
def upload():
    f = request.files['file']
    fname = secure_filename(f.filename)
    path = os.path.join(app.config['VIDEO_FOLDER'], fname)
    f.save(path)
    # 只返回文件名
    return jsonify({'filename': fname})


@app.route('/api/save-mask', methods=['POST'])
def save_mask_api():
    data = request.json
    vname = os.path.splitext(os.path.basename(data.get('video_filename', 'unknown')))[0]
    fname = f"{vname}_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(app.config['UPLOAD_FOLDER'], fname)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(data['mask_data'], f, indent=2, ensure_ascii=False)
    # 只返回文件名
    return jsonify({'mask_filename': fname})


@app.route('/api/extract-frame', methods=['POST'])
def extract_frame_api():
    data = request.json
    vid_filename = data.get('video_filename')
    vid_path = os.path.join(app.config['VIDEO_FOLDER'], os.path.basename(vid_filename))

    if not os.path.exists(vid_path):
        return jsonify({'error': 'Video not found'}), 404

    timestamp = data.get('timestamp', '00:00:00')
    h, m, s = map(int, timestamp.split(':'))
    seconds = h * 3600 + m * 60 + s

    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
    ret, frame = cap.read()
    cap.release()

    if ret:
        _, buf = cv2.imencode('.jpg', frame)
        b64 = base64.b64encode(buf).decode('utf-8')
        return jsonify({'success': True, 'image_data': f"data:image/jpeg;base64,{b64}"})
    return jsonify({'error': 'Extract failed'}), 500


@app.route('/api/tasks/<task_id>/download-video', methods=['GET'])
def download_video(task_id):
    if task_id not in tasks or not tasks[task_id].final_video_path:
        return jsonify({'error': 'Not ready'}), 404
    return send_file(tasks[task_id].final_video_path, as_attachment=True)


@app.route('/api/auto-segment', methods=['POST'])
def auto_segment_api():
    if sam_service is None:
        return jsonify({'error': 'SAM 模型未加载成功'}), 500

    data = request.json
    b64_img = data.get('image_data')
    click_x = data.get('x')
    click_y = data.get('y')
    box = data.get('box')

    if not b64_img or click_x is None or click_y is None:
        return jsonify({'error': '缺少必要参数'}), 400

    try:
        point_coords = [[click_x, click_y]]
        point_labels = [1]

        polygon = sam_service.predict_from_b64(
            b64_img,
            box=box,
            point_coords=point_coords,
            point_labels=point_labels
        )

        return jsonify({'success': True, 'polygon': polygon})
    except Exception as e:
        logger.error(f"SAM 预测出错: {e}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)