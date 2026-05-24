import os
import sys
import logging
import threading
from datetime import datetime
import uuid
import json
import glob
import cv2
import base64
import subprocess
import tempfile
from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from sam_helper import SAMService

WEB_DIR = os.path.dirname(os.path.abspath(__file__))
PROPAINTER_ROOT = os.path.dirname(WEB_DIR)
PROJECT_DIR = os.path.join(PROPAINTER_ROOT, 'project')

UPLOAD_FOLDER = os.path.join(PROPAINTER_ROOT, 'uploads')
VIDEO_FOLDER = os.path.join(PROPAINTER_ROOT, 'videos')
RESULTS_FOLDER = os.path.join(PROPAINTER_ROOT, 'results')

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(VIDEO_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="智能视频清洗系统 API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory=os.path.join(WEB_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(WEB_DIR, "templates"))

SAM_WEIGHT_PATH = os.path.join(PROPAINTER_ROOT, 'weights', 'sam_vit_b_01ec64.pth')
try:
    sam_service = SAMService(checkpoint_path=SAM_WEIGHT_PATH)
except Exception as e:
    logger.error(f"SAM 加载失败: {e}")
    sam_service = None

tasks = {}

class ScriptParams(BaseModel):
    source_video: str
    mask_file: str

class TaskCreateRequest(BaseModel):
    script_params: ScriptParams

class SaveMaskRequest(BaseModel):
    video_filename: str
    mask_data: Dict[str, Any]

class ExtractFrameRequest(BaseModel):
    video_filename: str
    timestamp: str = Field(default="00:00:00")

# 【核心修改】增加了 extract_mode 接收前端的双引擎指令
class AutoSegmentRequest(BaseModel):
    image_data: str
    pos_points: List[List[int]] = Field(default_factory=list)
    neg_points: List[List[int]] = Field(default_factory=list)
    box: Optional[List[int]] = None
    extract_mode: str = Field(default="semantic", description="semantic 或 color")

class ProcessingTask:
    def __init__(self, task_id, script_params_dict):
        self.task_id = task_id
        self.script_params = script_params_dict
        self.status = "pending"
        self.progress = 0
        self.current_stage = ""
        self.logs = []
        self.start_time = None
        self.end_time = None
        self.thread = None
        self.final_video_path = None

        self.source_video = os.path.join(VIDEO_FOLDER, os.path.basename(self.script_params['source_video']))
        self.mask_file = os.path.join(UPLOAD_FOLDER, os.path.basename(self.script_params['mask_file']))
        self.workspace = RESULTS_FOLDER

    def start(self):
        self.status = "running"
        self.start_time = datetime.now()
        self.thread = threading.Thread(target=self._run_task)
        self.thread.daemon = True
        self.thread.start()

    def _run_task(self):
        try:
            self._add_log("准备本地环境，开始处理...")
            cmd = [
                sys.executable,
                os.path.join(PROJECT_DIR, 'main_pipeline.py'),
                '--video', self.source_video,
                '--mask_json', self.mask_file,
                '--workspace', self.workspace,
                '--padding', '150',
                '--gpus', '0,1,2,3',
                '--scale_delay', '3',
                '--cooldown', '5'

            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True, cwd=PROJECT_DIR)
            while True:
                line = process.stdout.readline()
                if line == '' and process.poll() is not None: break
                if line.strip():
                    if self.status == "cancelled":
                        process.terminate(); break
                    clean_line = line.strip()
                    if clean_line.startswith("PROGRESS:"):
                        try:
                            parts = clean_line.split(":")
                            if len(parts) >= 3:
                                self.current_stage = parts[1]; self.progress = int(parts[2])
                        except: pass
                    else:
                        self._add_log(clean_line)
            return_code = process.wait()
            if self.status == "cancelled": self._add_log("任务已被取消")
            elif return_code == 0:
                self.status = "completed"; self.progress = 100
                self._find_final_video()
            else:
                self.status = "failed"
        except Exception as e:
            self.status = "failed"; self._add_log(f"执行失败: {str(e)}")
        finally:
            self.end_time = datetime.now()

    def _find_final_video(self):
        try:
            files = glob.glob(os.path.join(self.workspace, '**', 'final_output.mp4'), recursive=True)
            if files: self.final_video_path = max(files, key=os.path.getmtime)
        except: pass

    def _add_log(self, msg):
        self.logs.append({'timestamp': datetime.now().strftime("%H:%M:%S"), 'message': msg})

    def cancel(self):
        self.status = "cancelled"

@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# 【核心修改】恢复 async，并使用 1MB 异步分块安全读取！
@app.post("/api/upload", tags=["文件处理"])
async def upload(file: UploadFile = File(...)):
    """异步安全上传，防卡顿、防内存溢出"""
    fname = "".join(c for c in file.filename if c.isalnum() or c in " ._-")
    path = os.path.join(VIDEO_FOLDER, fname)
    with open(path, "wb") as buffer:
        while True:
            chunk = await file.read(1024 * 1024) # 每次读取 1MB
            if not chunk:
                break
            buffer.write(chunk)
    return {"filename": fname}

@app.post("/api/save-mask", tags=["掩码标注"])
async def save_mask_api(request_data: SaveMaskRequest):
    vname = os.path.splitext(os.path.basename(request_data.video_filename))[0]
    fname = f"{vname}_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(UPLOAD_FOLDER, fname)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(request_data.mask_data, f, indent=2, ensure_ascii=False)
    return {"mask_filename": fname}

@app.post("/api/extract-frame", tags=["视频截帧"])
async def extract_frame_api(request_data: ExtractFrameRequest):
    vid_path = os.path.join(VIDEO_FOLDER, os.path.basename(request_data.video_filename))
    if not os.path.exists(vid_path): raise HTTPException(status_code=404, detail="Video not found")

    # =================================================================
    # 【核心修复】：废弃缓慢的 OpenCV 逐帧解码，改用 FFmpeg 关键帧极速闪现
    # =================================================================
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp:
        temp_out = tmp.name

    try:
        # 注意：-ss 参数必须放在 -i 的前面，这代表开启 O(1) 级别的急速跳转 (Fast Seek)
        cmd = [
            'ffmpeg', '-y',
            '-ss', request_data.timestamp,  # 直接使用前端传来的 HH:MM:SS 字符串
            '-i', vid_path,
            '-vframes', '1',
            '-q:v', '2',
            temp_out
        ]

        # 执行命令，屏蔽底层日志输出
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if os.path.exists(temp_out) and os.path.getsize(temp_out) > 0:
            with open(temp_out, 'rb') as f:
                buf = f.read()
            b64 = base64.b64encode(buf).decode('utf-8')
            return {"success": True, "image_data": f"data:image/jpeg;base64,{b64}"}
        else:
            raise HTTPException(status_code=500, detail="Extract failed (FFmpeg returned empty)")

    finally:
        # 兜底清理：截帧完成后自动删除临时图片文件
        if os.path.exists(temp_out):
            os.remove(temp_out)

@app.post("/api/auto-segment", tags=["AI 抠图"])
async def auto_segment_api(request_data: AutoSegmentRequest):
    if sam_service is None: raise HTTPException(status_code=500, detail="SAM 模型未加载成功")
    try:
        point_coords = request_data.pos_points + request_data.neg_points
        point_labels = [1] * len(request_data.pos_points) + [0] * len(request_data.neg_points)
        if not point_coords: raise HTTPException(status_code=400, detail="未提供交互点")

        # 【核心修改】将前端选定的模式传给底层，并将返回值改名为 polygons (多边形矩阵)
        polygons = sam_service.predict_from_b64(
            request_data.image_data,
            box=request_data.box,
            point_coords=point_coords,
            point_labels=point_labels,
            extract_mode=request_data.extract_mode
        )
        return {"success": True, "polygons": polygons}
    except Exception as e:
        logger.error(f"SAM 预测出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/tasks", tags=["任务控制"])
async def create_task(request_data: TaskCreateRequest):
    task_id = str(uuid.uuid4())
    tasks[task_id] = ProcessingTask(task_id, request_data.script_params.dict())
    tasks[task_id].start()
    return {"task_id": task_id, "status": "started"}

@app.get("/api/tasks/{task_id}", tags=["任务控制"])
async def get_task(task_id: str):
    if task_id not in tasks: raise HTTPException(status_code=404, detail="Task Not found")
    t = tasks[task_id]
    return { "status": t.status, "progress": t.progress, "current_stage": t.current_stage, "logs": t.logs[-50:], "final_video_path": t.final_video_path }

@app.post("/api/tasks/{task_id}/cancel", tags=["任务控制"])
async def cancel_task(task_id: str):
    if task_id in tasks: tasks[task_id].cancel()
    return {"status": "cancelled"}

@app.get("/api/tasks/{task_id}/download-video", tags=["文件处理"])
async def download_video(task_id: str):
    if task_id not in tasks or not tasks[task_id].final_video_path: raise HTTPException(status_code=404, detail="Video Not ready")
    return FileResponse(tasks[task_id].final_video_path, media_type='video/mp4', filename=f"cleaned_video.mp4")