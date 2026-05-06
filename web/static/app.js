let currentTaskId = null;
let statusInterval = null;

// Mask制作相关变量
let isDrawingMode = false;
let isDrawing = false;
let startX, startY;
let currentRect = null;
let maskRects = [];
let boxIdCounter = 0;

let currentVideoFilename = '';
let uploadedMaskFilename = null;

// 全局视频时长控制
let globalVideoDuration = 0;
// 记录当前提取参考画面的时间戳（作为自动追踪的种子）
let currentExtractTimeStr = "00:00:00";

// 魔法棒(SAM) 多点交互会话状态
let samSession = { active: false, pos: [], neg: [], polygon: null };

let imageScale = 1;
let imageOffsetX = 0;
let imageOffsetY = 0;

const FIXED_SCRIPT_PARAMS = {
    scene_threshold: 20.0,
    min_duration: 0.5,
    max_duration: 5.0,
    scale: 0.25,
    output_scale: 1.0,
    max_frames: 0,
    output_fps: 0,
    convert_to_mp4: true,
    max_workers: 1,
    enable_parallel: true,
    batch_size: 1,
    keep_intermediate: true
};

function initializePage() {
    const sourceVideo = document.getElementById('source-video');
    if (!sourceVideo.value) { sourceVideo.value = 'sample.mp4'; }
    currentVideoFilename = sourceVideo.value;
    resetCanvas();
}

async function uploadVideo() {
    const fileInput = document.getElementById('video-file');
    if (!fileInput.files[0]) { alert('请选择要上传的视频文件'); return; }

    const file = fileInput.files[0];

    const videoNode = document.createElement('video');
    videoNode.preload = 'metadata';
    videoNode.onloadedmetadata = function() {
        globalVideoDuration = videoNode.duration;
        window.URL.revokeObjectURL(videoNode.src);
        console.log("已成功获取视频总时长:", globalVideoDuration, "秒");
    };
    videoNode.src = URL.createObjectURL(file);

    const formData = new FormData();
    formData.append('file', file);

    const btn = document.getElementById('upload-btn');
    const originalText = btn.innerHTML;
    try {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>上传云端中...';
        const response = await fetch('/api/upload', { method: 'POST', body: formData });
        const result = await response.json();
        if (response.ok) {
            document.getElementById('source-video').value = result.filename;
            currentVideoFilename = result.filename;
            alert(`视频上传成功！\n系统检测到该视频总长为: ${Math.floor(globalVideoDuration)} 秒`);
        } else { alert('上传失败: ' + result.error); }
    } catch (error) { alert('上传错误: ' + error.message); }
    finally {
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

async function startProcessing() {
    const sourceVideo = document.getElementById('source-video').value;
    if (!sourceVideo) { alert('请设置源视频文件名'); return; }

    // 如果有可见图层但没有提交
    const activeMasks = maskRects.filter(b => b.visible);
    if (activeMasks.length > 0 && !uploadedMaskFilename) {
        if(!confirm('检测到您绘制了图层但尚未提交数据。直接处理将忽略这些标注。\n是否继续？')) return;
    }

    const scriptParams = { ...FIXED_SCRIPT_PARAMS, source_video: currentVideoFilename, mask_file: uploadedMaskFilename };
    try {
        const response = await fetch('/api/tasks', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ script_params: scriptParams })
        });
        const result = await response.json();
        if (response.ok) {
            currentTaskId = result.task_id;
            document.getElementById('cancel-btn').disabled = false;
            document.getElementById('download-video-btn').disabled = true;
            startStatusPolling();
        } else { alert('任务启动失败: ' + result.error); }
    } catch (error) { alert('任务启动错误: ' + error.message); }
}

async function cancelProcessing() {
    if (!currentTaskId) return;
    try {
        await fetch(`/api/tasks/${currentTaskId}/cancel`, { method: 'POST' });
        document.getElementById('cancel-btn').disabled = true;
    } catch (error) { console.error('取消任务错误:', error); }
}

async function downloadVideo() {
    if (!currentTaskId) { alert('没有可用的视频文件'); return; }
    try {
        const response = await fetch(`/api/tasks/${currentTaskId}/download-video`);
        if (response.ok) {
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.style.display = 'none';
            a.href = url;
            a.download = '处理后纯净视频.mp4';
            document.body.appendChild(a);
            a.click();
            window.URL.revokeObjectURL(url);
            document.body.removeChild(a);
        } else {
            const error = await response.json();
            alert('下载视频失败: ' + error.error);
        }
    } catch (error) { alert('下载视频错误: ' + error.message); }
}

async function extractFrame() {
    if (!currentVideoFilename) { alert('请先选择或上传视频'); return; }

    const hours = parseInt(document.getElementById('hours').value) || 0;
    const minutes = parseInt(document.getElementById('minutes').value) || 0;
    const seconds = parseInt(document.getElementById('seconds').value) || 0;

    const totalSec = hours * 3600 + minutes * 60 + seconds;
    if (globalVideoDuration > 0 && totalSec > globalVideoDuration) {
        alert(`❌ 提取失败：设定的提取时间点 (${totalSec}秒) 已超出视频的总时长 (${Math.floor(globalVideoDuration)}秒)！`);
        return;
    }

    currentExtractTimeStr = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;

    const btn = document.getElementById('extract-frame-btn');
    const originalText = btn.innerHTML;
    const controller = new AbortController();
    const timeoutId = setTimeout(() => controller.abort(), 30000);

    try {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>提取中...';
        const response = await fetch('/api/extract-frame', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_filename: currentVideoFilename, timestamp: currentExtractTimeStr }),
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        const result = await response.json();

        if (response.ok && result.success) {
            const maskImage = document.getElementById('mask-image');
            const framePlaceholder = document.getElementById('frame-placeholder');

            // 切换画面不清空图层，确保可以看到之前画的
            framePlaceholder.style.display = 'none';
            maskImage.src = result.image_data;
            maskImage.style.display = 'block';
            maskImage.classList.add('active');

            await new Promise((resolve) => {
                maskImage.onload = function() {
                    calculateImageDisplay(maskImage);
                    initDrawing();
                    toggleDrawingMode(true);
                    redrawCanvas();
                    resolve();
                };
                maskImage.onerror = () => { alert('图片加载失败'); resolve(); };
            });
        } else { alert('提取帧失败: ' + (result.error || '未知错误')); }
    } catch (error) {
        if (error.name === 'AbortError') { alert('提取帧超时：服务器在30秒内未响应。'); }
        else { alert('提取帧错误: ' + error.message); }
    } finally {
        clearTimeout(timeoutId);
        btn.disabled = false;
        btn.innerHTML = originalText;
    }
}

function calculateImageDisplay(maskImage) {
    const container = document.getElementById('mask-container');
    const containerWidth = container.clientWidth;
    const containerHeight = container.clientHeight;
    const imageWidth = maskImage.naturalWidth;
    const imageHeight = maskImage.naturalHeight;

    const scaleX = containerWidth / imageWidth;
    const scaleY = containerHeight / imageHeight;
    imageScale = Math.min(scaleX, scaleY, 1);

    const displayWidth = imageWidth * imageScale;
    const displayHeight = imageHeight * imageScale;

    imageOffsetX = (containerWidth - displayWidth) / 2;
    imageOffsetY = (containerHeight - displayHeight) / 2;

    maskImage.style.width = `${displayWidth}px`; maskImage.style.height = `${displayHeight}px`;
    maskImage.style.left = `${imageOffsetX}px`; maskImage.style.top = `${imageOffsetY}px`;

    const maskCanvas = document.getElementById('mask-canvas');
    maskCanvas.style.width = `${displayWidth}px`; maskCanvas.style.height = `${displayHeight}px`;
    maskCanvas.style.left = `${imageOffsetX}px`; maskCanvas.style.top = `${imageOffsetY}px`;

    maskCanvas.width = displayWidth; maskCanvas.height = displayHeight;
    maskCanvas.style.display = 'block'; maskCanvas.classList.add('active');
}

function displayToImageCoordinates(displayX, displayY) {
    return { x: Math.round(displayX / imageScale), y: Math.round(displayY / imageScale) };
}

function toggleDrawingMode(forceState) {
    const btn = document.getElementById('start-drawing-btn');
    const canvas = document.getElementById('mask-canvas');
    isDrawingMode = forceState !== undefined ? forceState : !isDrawingMode;

    if (isDrawingMode) {
        btn.classList.remove('btn-warning'); btn.classList.add('btn-danger');
        btn.innerHTML = '<i class="fas fa-stop me-1"></i>停止绘制';
        if(canvas) canvas.style.cursor = 'crosshair';
    } else {
        btn.classList.remove('btn-danger'); btn.classList.add('btn-warning');
        btn.innerHTML = '<i class="fas fa-pen-nib me-1"></i>进入绘制模式';
        if(canvas) canvas.style.cursor = 'default';
        // 如果退出了绘制模式但还没确认SAM，自动放弃
        if (samSession.active) {
            samSession = { active: false, pos: [], neg: [], polygon: null };
            document.getElementById('commit-sam-btn').style.display = 'none';
            redrawCanvas();
        }
    }
}

// ============== 核心事件绑定与绘制 =================

function initDrawing() {
    const canvas = document.getElementById('mask-canvas');
    const newCanvas = canvas.cloneNode(true);
    canvas.parentNode.replaceChild(newCanvas, canvas);

    // 禁用默认右键菜单，防止点按Alt等快捷键时的误触
    newCanvas.addEventListener('contextmenu', e => e.preventDefault());

    newCanvas.addEventListener('mousedown', async function(e) {
        if (!isDrawingMode) return;

        const rect = newCanvas.getBoundingClientRect();
        startX = e.clientX - rect.left;
        startY = e.clientY - rect.top;

        const type = document.getElementById('regionType').value;
        const mode = document.getElementById('drawMode').value;

        // 【新增多点正负向交互】
        if (mode === 'magic') {
            samSession.active = true;
            document.getElementById('commit-sam-btn').style.display = 'block';

            const realCoords = displayToImageCoordinates(startX, startY);
            if (e.altKey) {
                samSession.neg.push([realCoords.x, realCoords.y]);
            } else {
                samSession.pos.push([realCoords.x, realCoords.y]);
            }

            await triggerSAM(newCanvas);
            return;
        }

        // 普通框选逻辑
        isDrawing = true;
        currentRect = {
            id: ++boxIdCounter,
            x: startX, y: startY,
            width: 0, height: 0,
            type: type, mode: mode,
            visible: true,
            ref_time: currentExtractTimeStr
        };
    });

    newCanvas.addEventListener('mousemove', function(e) {
        if (!isDrawing || !currentRect || currentRect.mode === 'magic') return;
        const rect = newCanvas.getBoundingClientRect();
        currentRect.width = (e.clientX - rect.left) - startX;
        currentRect.height = (e.clientY - rect.top) - startY;
        redrawCanvas(newCanvas);
    });

    newCanvas.addEventListener('mouseup', function(e) {
        if (!isDrawing || !currentRect || currentRect.mode === 'magic') return;
        isDrawing = false;

        // 纠正反向拖拽
        if (currentRect.width < 0) { currentRect.x += currentRect.width; currentRect.width = Math.abs(currentRect.width); }
        if (currentRect.height < 0) { currentRect.y += currentRect.height; currentRect.height = Math.abs(currentRect.height); }

        if (Math.abs(currentRect.width) > 0 && Math.abs(currentRect.height) > 0) {
            maskRects.push(currentRect);
            updateLayerPanel();
        }
        currentRect = null;
        redrawCanvas(newCanvas);
    });
}

// 触发多点SAM预测
async function triggerSAM(canvas) {
    canvas.style.cursor = 'wait';
    try {
        const maskImage = document.getElementById('mask-image');
        const response = await fetch('/api/auto-segment', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                image_data: maskImage.src,
                pos_points: samSession.pos,
                neg_points: samSession.neg
            })
        });
        const result = await response.json();

        if (response.ok && result.success && result.polygon && result.polygon.length > 0) {
            samSession.polygon = result.polygon;
        } else {
            console.warn("SAM无法识别明确对象");
        }
        redrawCanvas(canvas);
    } catch (error) {
        alert('AI推理节点通信失败: ' + error.message);
    } finally {
        canvas.style.cursor = 'crosshair';
    }
}

// 确认并提交当前的SAM结果为图层
document.getElementById('commit-sam-btn').addEventListener('click', () => {
    if (samSession.polygon) {
        maskRects.push({
            id: ++boxIdCounter,
            type: document.getElementById('regionType').value,
            mode: 'magic',
            polygon: samSession.polygon,
            visible: true,
            ref_time: currentExtractTimeStr
        });
        updateLayerPanel();
    }
    // 重置Session
    samSession = { active: false, pos: [], neg: [], polygon: null };
    document.getElementById('commit-sam-btn').style.display = 'none';
    redrawCanvas();
});


// ============== 图层面板管理 =================

function getRegionTypeName(type) {
    const map = { 'logo': '台标', 'subtitle': '字幕', 'other': '自定义区域' };
    return map[type] || type;
}

window.toggleLayer = function(id) {
    const layer = maskRects.find(b => b.id === id);
    if (layer) {
        layer.visible = !layer.visible;
        redrawCanvas();
        updateLayerPanel();
    }
};

window.deleteLayer = function(id) {
    maskRects = maskRects.filter(b => b.id !== id);
    redrawCanvas();
    updateLayerPanel();
};

function clearAllBoxes() {
    if (maskRects.length === 0) return;
    if (confirm('确定要清空画布上的所有图层吗？')) {
        maskRects = [];
        samSession = { active: false, pos: [], neg: [], polygon: null };
        document.getElementById('commit-sam-btn').style.display = 'none';
        redrawCanvas();
        updateLayerPanel();
    }
}

function updateLayerPanel() {
    const panel = document.getElementById('layerPanel');
    if (maskRects.length === 0) {
        panel.innerHTML = '<div class="text-muted text-center" style="font-size:11px; margin-top:30px;">暂无图层</div>';
        return;
    }

    let html = '';
    // 倒序渲染，新的在最上面
    [...maskRects].reverse().forEach(box => {
        const eyeIcon = box.visible ? 'fa-eye text-primary' : 'fa-eye-slash text-muted';
        const typeStr = box.mode === 'magic' ? '✨ 智能蒙版' : '🟥 区域红框';
        // 如果图层被隐藏，加点透明度
        const opacity = box.visible ? '1' : '0.5';

        html += `
        <div class="d-flex justify-content-between align-items-center mb-1 pb-1" style="border-bottom:1px solid #f0f0f0; font-size:12px; opacity:${opacity};">
            <div>
                <i class="fas ${eyeIcon} me-2" style="cursor:pointer;" onclick="toggleLayer(${box.id})"></i>
                <span style="font-weight:600; color:var(--dark);">${getRegionTypeName(box.type)}</span>
                <span style="color:var(--gray-text); margin-left:4px;">${typeStr}</span>
                <span class="text-muted" style="font-size:10px; margin-left:4px;">(帧: ${box.ref_time})</span>
            </div>
            <i class="fas fa-times text-danger" style="cursor:pointer;" onclick="deleteLayer(${box.id})"></i>
        </div>`;
    });
    panel.innerHTML = html;
}

// ============== 核心绘制逻辑 =================

function redrawCanvas(canvasElement) {
    const canvas = canvasElement || document.getElementById('mask-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    // 渲染已经保存的图层 (仅渲染可见部分)
    maskRects.forEach(box => {
        if (box.visible) drawBox(ctx, box);
    });

    // 渲染鼠标正在拖拽的框
    if (isDrawing && currentRect) {
        drawBox(ctx, currentRect, true);
    }

    // 渲染处于编辑态的 SAM 会话点和轮廓
    if (samSession.active) {
        if (samSession.polygon) {
            ctx.strokeStyle = '#9b59b6';
            ctx.lineWidth = 2;
            ctx.fillStyle = 'rgba(155, 89, 182, 0.4)';
            ctx.beginPath();
            samSession.polygon.forEach((pt, index) => {
                const dispX = pt[0] * imageScale;
                const dispY = pt[1] * imageScale;
                if (index === 0) ctx.moveTo(dispX, dispY); else ctx.lineTo(dispX, dispY);
            });
            ctx.closePath(); ctx.fill(); ctx.stroke();
        }

        // 渲染正向特征点 (绿色圆点)
        ctx.fillStyle = '#2ecc71';
        samSession.pos.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt[0] * imageScale, pt[1] * imageScale, 4, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
        });

        // 渲染负向特征点 (红色圆点)
        ctx.fillStyle = '#e74c3c';
        samSession.neg.forEach(pt => {
            ctx.beginPath();
            ctx.arc(pt[0] * imageScale, pt[1] * imageScale, 4, 0, 2 * Math.PI);
            ctx.fill();
            ctx.stroke();
        });
    }
}

function drawBox(ctx, box, isDashed = false) {
    const isLarge = box.mode === 'large';
    const labelTitle = `${getRegionTypeName(box.type)} [L${box.id}]`;

    if (box.mode === 'magic' && box.polygon) {
        ctx.strokeStyle = '#3498db';
        ctx.lineWidth = 2;
        ctx.fillStyle = 'rgba(52, 152, 219, 0.3)';
        ctx.beginPath();
        box.polygon.forEach((pt, index) => {
            const dispX = pt[0] * imageScale;
            const dispY = pt[1] * imageScale;
            if (index === 0) ctx.moveTo(dispX, dispY); else ctx.lineTo(dispX, dispY);
        });
        ctx.closePath(); ctx.fill(); ctx.stroke();
        return;
    }

    const strokeColor = '#e74c3c';
    const fillColor = 'rgba(231, 76, 60, 0.15)';

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2;
    ctx.fillStyle = fillColor;
    if (isDashed) ctx.setLineDash([5, 5]); else ctx.setLineDash([]);
    ctx.fillRect(box.x, box.y, box.width, box.height);
    ctx.strokeRect(box.x, box.y, box.width, box.height);

    if (!isDashed && isLarge) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        const textWidth = ctx.measureText(labelTitle).width + 10;
        ctx.fillRect(box.x, box.y - 20, textWidth, 20);

        ctx.fillStyle = 'rgba(255, 255, 255, 1)';
        ctx.font = '12px Arial';
        ctx.textBaseline = 'bottom';
        ctx.fillText(labelTitle, box.x + 5, box.y - 5);
    }
}

// ============== 数据打包交互 =================

async function generateJSON() {
    // 【核心】过滤掉小眼睛被关掉的图层
    const activeMasks = maskRects.filter(b => b.visible);
    if (activeMasks.length === 0) { alert('当前没有任何可见的图层！无法生成标注集。'); return; }

    const maskImage = document.getElementById('mask-image');
    const jsonData = {
        metadata: {
            timestamp: new Date().toISOString(),
            videoFilename: currentVideoFilename,
            originalResolution: { width: maskImage.naturalWidth, height: maskImage.naturalHeight }
        },
        regions: {}
    };

    activeMasks.forEach(box => {
        // 防止相同的 other 区域覆盖，每个 other 独立一个 ID
        let regionKey = box.type === 'other' ? `other_${box.id}` : box.type;

        if (!jsonData.regions[regionKey]) {
            jsonData.regions[regionKey] = {
                typeName: getRegionTypeName(box.type),
                reference_time: box.ref_time, // 【核心】将绘制帧的时间戳交给后台用于 AI 追踪
                largeBoxes: [],
                smallBoxes: []
            };
        }

        if (box.mode === 'magic') {
            jsonData.regions[regionKey].smallBoxes.push({ id: box.id, polygon: box.polygon });
        } else {
            const realCoords = displayToImageCoordinates(box.x, box.y);
            const boxData = {
                id: box.id,
                x: realCoords.x, y: realCoords.y,
                width: Math.round(box.width / imageScale), height: Math.round(box.height / imageScale)
            };
            jsonData.regions[regionKey].largeBoxes.push(boxData);
        }
    });

    const btn = document.getElementById('generate-json-btn');
    const originalBtnText = btn.innerHTML;
    try {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>正在打包同步...';
        const response = await fetch('/api/save-mask', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mask_data: jsonData, video_filename: currentVideoFilename })
        });
        const result = await response.json();
        if (response.ok) {
            uploadedMaskFilename = result.mask_filename;
            const statusDiv = document.getElementById('mask-upload-status');
            statusDiv.style.display = 'block';
            statusDiv.innerHTML = `<i class="fas fa-check-circle me-1"></i>成功同步 ${activeMasks.length} 个可见图层`;
            alert(`标注提交成功！后台 AI 自动追踪引擎已准备就绪。`);
        } else { throw new Error(result.error || '上传失败'); }
    } catch (error) { alert('云端网络异常: ' + error.message); }
    finally { btn.disabled = false; btn.innerHTML = originalBtnText; }
}

function resetCanvas() {
    maskRects = [];
    currentRect = null;
    boxIdCounter = 0;
    uploadedMaskFilename = null;
    samSession = { active: false, pos: [], neg: [], polygon: null };

    document.getElementById('mask-upload-status').style.display = 'none';
    document.getElementById('commit-sam-btn').style.display = 'none';

    const maskImage = document.getElementById('mask-image');
    const maskCanvas = document.getElementById('mask-canvas');
    const framePlaceholder = document.getElementById('frame-placeholder');

    if(maskImage) { maskImage.src = ''; maskImage.style.display = 'none'; maskImage.classList.remove('active'); }
    if(maskCanvas) { maskCanvas.style.display = 'none'; maskCanvas.classList.remove('active'); }
    if(framePlaceholder) framePlaceholder.style.display = 'block';

    const btn = document.getElementById('start-drawing-btn');
    if(btn) toggleDrawingMode(false);
    updateLayerPanel();
}

function startStatusPolling() {
    if (statusInterval) clearInterval(statusInterval);
    statusInterval = setInterval(async () => {
        if (!currentTaskId) return;
        try {
            const response = await fetch(`/api/tasks/${currentTaskId}`);
            const task = await response.json();
            updateTaskStatus(task);
            if (task.status === 'completed' || task.status === 'failed' || task.status === 'cancelled') {
                clearInterval(statusInterval);
                document.getElementById('cancel-btn').disabled = true;
            }
        } catch (error) { console.error('状态查询错误:', error); }
    }, 2000);
}

function updateTaskStatus(task) {
    const progressBar = document.getElementById('progress-bar');
    progressBar.style.width = task.progress + '%';
    document.getElementById('progress-text').textContent = getStageText(task.current_stage) + ' ' + task.progress + '%';
    document.getElementById('download-video-btn').disabled = (task.status !== 'completed');
}

function getStageText(stage) {
    const stageMap = { 'initialization': '分配算力', 'scene_detection': '光流分析', 'mask_generation': '渲染掩码', 'video_processing': '分布式AI推理', 'finalizing': '编码合成', 'completed': '清洗完毕' };
    return stageMap[stage] || stage || '节点唤醒中';
}

document.addEventListener('DOMContentLoaded', function() {
    initializePage();
    document.getElementById('upload-btn').addEventListener('click', uploadVideo);
    document.getElementById('start-btn').addEventListener('click', startProcessing);
    document.getElementById('cancel-btn').addEventListener('click', cancelProcessing);
    document.getElementById('download-video-btn').addEventListener('click', downloadVideo);
    document.getElementById('extract-frame-btn').addEventListener('click', extractFrame);
    document.getElementById('start-drawing-btn').addEventListener('click', () => toggleDrawingMode());
    document.getElementById('clear-all-btn').addEventListener('click', clearAllBoxes);
    document.getElementById('generate-json-btn').addEventListener('click', generateJSON);

    document.getElementById('source-video').addEventListener('change', function() { currentVideoFilename = this.value; });

    const drawModeSelect = document.getElementById('drawMode');
    drawModeSelect.addEventListener('change', function() {
        document.getElementById('magic-hint').style.display = this.value === 'magic' ? 'block' : 'none';
    });
});