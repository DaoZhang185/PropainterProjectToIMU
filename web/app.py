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
import shutil

from fastapi import FastAPI, UploadFile, File, Request, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any

from sam_helper import SAMService

# =================================================================
# 【核心改造】全自动路径解析 (不再写死任何绝对路径)
# =================================================================
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

# =================================================================
# FastAPI 应用初始化与配置
# =================================================================
app = FastAPI(
    title="智能视频清洗系统 API",
    description="提供视频上传、切帧、智能选区、清洗任务调度等全套接口，供外部系统调用对接。",
    version="1.0.0"
)

# 配置 CORS跨域
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 挂载静态文件和模板 (保留原有网页界面的功能)
app.mount("/static", StaticFiles(directory=os.path.join(WEB_DIR, "static")), name="static")
templates = Jinja2Templates(directory=os.path.join(WEB_DIR, "templates"))

# 初始化 SAM 服务
SAM_WEIGHT_PATH = os.path.join(PROPAINTER_ROOT, 'weights', 'sam_vit_b_01ec64.pth')
try:
    sam_service = SAMService(checkpoint_path=SAM_WEIGHT_PATH)
except Exception as e:
    logger.error(f"SAM 加载失败: {e}")
    sam_service = None

tasks = {}

# =================================================================
# 定义请求数据模型 (用于自动生成严谨的 Swagger 接口文档)
# =================================================================
class ScriptParams(BaseModel):
    source_video: str = Field(..., description="上传接口返回的视频文件名")
    mask_file: str = Field(..., description="保存遮罩接口返回的 JSON 文件名")
    # 可以根据你的需要加入其他非必填参数，如 threshold 等

class TaskCreateRequest(BaseModel):
    script_params: ScriptParams

class SaveMaskRequest(BaseModel):
    video_filename: str = Field(..., description="原始视频文件名")
    mask_data: Dict[str, Any] = Field(..., description="前端绘制生成的遮罩 JSON 结构")

class ExtractFrameRequest(BaseModel):
    video_filename: str = Field(..., description="视频文件名")
    timestamp: str = Field(default="00:00:00", description="时间戳，格式 HH:MM:SS")

class AutoSegmentRequest(BaseModel):
    image_data: str = Field(..., description="视频帧的 base64 字符串")
    pos_points: List[List[int]] = Field(default_factory=list, description="正向点选坐标列表 (想要选中的区域)")
    neg_points: List[List[int]] = Field(default_factory=list, description="负向点选坐标列表 (想排除的区域)")
    box: Optional[List[int]] = Field(default=None, description="大框坐标 [x1, y1, x2, y2]")
# =================================================================
# 核心任务调度类 (完全保留原有逻辑)
# =================================================================
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
            cmd = [
                sys.executable,
                os.path.join(PROJECT_DIR, 'main_pipeline.py'),
                '--video', self.source_video,
                '--mask_json', self.mask_file,
                '--workspace', self.workspace,
                '--padding', '150',
                '--gpus', '0,1,2,3',
                '--scale_delay', '3',
                '--cooldown', '10'
            ]

            self._add_log(f"执行命令: {' '.join(cmd)}")

            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                universal_newlines=True,
                cwd=PROJECT_DIR
            )

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
            files = glob.glob(os.path.join(self.workspace, '**', 'final_output.mp4'), recursive=True)
            if files:
                self.final_video_path = max(files, key=os.path.getmtime)
        except:
            pass

    def _add_log(self, msg):
        self.logs.append({'timestamp': datetime.now().strftime("%H:%M:%S"), 'message': msg})

    def cancel(self):
        self.status = "cancelled"


# =================================================================
# API 路由接口定义
# =================================================================

@app.get("/", response_class=HTMLResponse, tags=["页面"])
async def index(request: Request):
    """访问系统可视化工作台界面"""
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/api/upload", tags=["文件处理"])
async def upload(file: UploadFile = File(...)):
    """上传原始视频文件"""
    # 替换 werkzeug 的 secure_filename
    fname = "".join(c for c in file.filename if c.isalnum() or c in " ._-")
    path = os.path.join(VIDEO_FOLDER, fname)
    with open(path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    return {"filename": fname}


@app.post("/api/save-mask", tags=["掩码标注"])
async def save_mask_api(request_data: SaveMaskRequest):
    """接收前端 JSON 并生成遮罩文件"""
    vname = os.path.splitext(os.path.basename(request_data.video_filename))[0]
    fname = f"{vname}_mask_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    path = os.path.join(UPLOAD_FOLDER, fname)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(request_data.mask_data, f, indent=2, ensure_ascii=False)
    return {"mask_filename": fname}


@app.post("/api/extract-frame", tags=["视频截帧"])
async def extract_frame_api(request_data: ExtractFrameRequest):
    """根据时间戳提取视频画面，返回 Base64"""
    vid_path = os.path.join(VIDEO_FOLDER, os.path.basename(request_data.video_filename))

    if not os.path.exists(vid_path):
        raise HTTPException(status_code=404, detail="Video not found")

    timestamp = request_data.timestamp
    try:
        h, m, s = map(int, timestamp.split(':'))
        seconds = h * 3600 + m * 60 + s
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid timestamp format, expect HH:MM:SS")

    cap = cv2.VideoCapture(vid_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(seconds * fps))
    ret, frame = cap.read()
    cap.release()

    if ret:
        _, buf = cv2.imencode('.jpg', frame)
        b64 = base64.b64encode(buf).decode('utf-8')
        return {"success": True, "image_data": f"data:image/jpeg;base64,{b64}"}
    raise HTTPException(status_code=500, detail="Extract failed")


@app.post("/api/auto-segment", tags=["AI 抠图"])
async def auto_segment_api(request_data: AutoSegmentRequest):
    """调用 SAM 模型进行智能点选目标分割 (支持正负向多点)"""
    if sam_service is None:
        raise HTTPException(status_code=500, detail="SAM 模型未加载成功")

    try:
        # 合并正负向点，并分配标签 (1为正，0为负)
        point_coords = request_data.pos_points + request_data.neg_points
        point_labels = [1] * len(request_data.pos_points) + [0] * len(request_data.neg_points)

        if not point_coords:
            raise HTTPException(status_code=400, detail="未提供任何交互点")

        polygon = sam_service.predict_from_b64(
            request_data.image_data,
            box=request_data.box,
            point_coords=point_coords,
            point_labels=point_labels
        )
        return {"success": True, "polygon": polygon}
    except Exception as e:
        logger.error(f"SAM 预测出错: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/tasks", tags=["任务控制"])
async def create_task(request_data: TaskCreateRequest):
    """提交参数并启动视频智能清洗算法任务"""
    task_id = str(uuid.uuid4())
    # request_data.script_params.dict() 将 Pydantic 对象转回字典
    tasks[task_id] = ProcessingTask(task_id, request_data.script_params.dict())
    tasks[task_id].start()
    return {"task_id": task_id, "status": "started"}


@app.get("/api/tasks/{task_id}", tags=["任务控制"])
async def get_task(task_id: str):
    """轮询任务当前处理进度与日志"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task Not found")
    t = tasks[task_id]
    return {
        "status": t.status,
        "progress": t.progress,
        "current_stage": t.current_stage,
        "logs": t.logs[-50:],
        "final_video_path": t.final_video_path
    }


@app.post("/api/tasks/{task_id}/cancel", tags=["任务控制"])
async def cancel_task(task_id: str):
    """强行终止正在进行的清洗任务"""
    if task_id in tasks:
        tasks[task_id].cancel()
    return {"status": "cancelled"}


@app.get("/api/tasks/{task_id}/download-video", tags=["文件处理"])
async def download_video(task_id: str):
    """下载处理完成后的无水印纯净视频"""
    if task_id not in tasks or not tasks[task_id].final_video_path:
        raise HTTPException(status_code=404, detail="Video Not ready")
    return FileResponse(
        tasks[task_id].final_video_path,
        media_type='video/mp4',
        filename=f"cleaned_video_{task_id[:8]}.mp4"
    )