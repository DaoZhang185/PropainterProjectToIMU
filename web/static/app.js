let currentTaskId = null;
let statusInterval = null;

// Mask制作相关变量
let isDrawingMode = false;
let isDrawing = false;
let startX, startY;
let currentRect = null;
let maskRects = [];
let boxIdCounter = 0;
let otherRegionCounter = 0; // 用来追踪产生了几组"其他区域"

let currentVideoFilename = '';
let uploadedMaskFilename = null;

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
    const formData = new FormData();
    formData.append('file', fileInput.files[0]);

    try {
        const response = await fetch('/api/upload', { method: 'POST', body: formData });
        const result = await response.json();
        if (response.ok) {
            document.getElementById('source-video').value = result.filename;
            currentVideoFilename = result.filename;
            alert('视频上传成功！');
        } else { alert('上传失败: ' + result.error); }
    } catch (error) { alert('上传错误: ' + error.message); }
}

async function startProcessing() {
    const sourceVideo = document.getElementById('source-video').value;
    if (!sourceVideo) { alert('请设置源视频文件名'); return; }
    if (maskRects.length > 0 && !uploadedMaskFilename) {
        if(!confirm('检测到您绘制了Mask但尚未上传。直接处理将忽略这些标注。\n是否继续？')) return;
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
            a.download = '处理后视频.mp4';
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
    const timestamp = `${hours.toString().padStart(2, '0')}:${minutes.toString().padStart(2, '0')}:${seconds.toString().padStart(2, '0')}`;

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
            body: JSON.stringify({ video_filename: currentVideoFilename, timestamp: timestamp }),
            signal: controller.signal
        });
        clearTimeout(timeoutId);
        const result = await response.json();

        if (response.ok && result.success) {
            const maskImage = document.getElementById('mask-image');
            const framePlaceholder = document.getElementById('frame-placeholder');
            resetCanvas();
            framePlaceholder.style.display = 'none';
            maskImage.src = result.image_data;
            maskImage.style.display = 'block';
            maskImage.classList.add('active');

            await new Promise((resolve) => {
                maskImage.onload = function() {
                    calculateImageDisplay(maskImage);
                    initDrawing();
                    toggleDrawingMode(true);
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
        btn.innerHTML = '<i class="fas fa-stop me-1"></i>停止画框';
        canvas.style.cursor = 'crosshair';
    } else {
        btn.classList.remove('btn-danger'); btn.classList.add('btn-warning');
        btn.innerHTML = '<i class="fas fa-pen me-1"></i>激活画笔';
        canvas.style.cursor = 'default';
    }
}

// 动态创建极具交互感的三联时间输入框
function addOtherTimeInput(groupId) {
    const container = document.getElementById('dynamicTimeContainers');
    const div = document.createElement('div');
    div.className = 'mt-2 p-2 bg-light border rounded';
    div.id = `time_container_${groupId}`;

    // oninput 绑定重绘，实现输入框更改，画布文本自动刷新！
    div.innerHTML = `
        <div class="d-flex justify-content-between align-items-center mb-1">
            <span class="small text-dark fw-bold">其他区域${groupId} 生效时间</span>
        </div>
        <div class="mb-1"><span class="small text-muted" style="font-size:11px;">起:</span></div>
        <div class="time-input-group mb-1">
            <div class="time-input-wrapper">
                <input type="number" class="form-control form-control-sm" id="start_h_${groupId}" min="0" value="0" oninput="redrawCanvas(); updateAnnotationInfo();">
                <span class="time-input-unit">时</span>
            </div>
            <div class="time-separator">:</div>
            <div class="time-input-wrapper">
                <input type="number" class="form-control form-control-sm" id="start_m_${groupId}" min="0" max="59" value="0" oninput="redrawCanvas(); updateAnnotationInfo();">
                <span class="time-input-unit">分</span>
            </div>
            <div class="time-separator">:</div>
            <div class="time-input-wrapper">
                <input type="number" class="form-control form-control-sm" id="start_s_${groupId}" min="0" max="59" value="0" oninput="redrawCanvas(); updateAnnotationInfo();">
                <span class="time-input-unit">秒</span>
            </div>
        </div>
        <div class="mb-1 mt-2"><span class="small text-muted" style="font-size:11px;">止:</span></div>
        <div class="time-input-group mb-1">
            <div class="time-input-wrapper">
                <input type="number" class="form-control form-control-sm" id="end_h_${groupId}" min="0" value="0" oninput="redrawCanvas(); updateAnnotationInfo();">
                <span class="time-input-unit">时</span>
            </div>
            <div class="time-separator">:</div>
            <div class="time-input-wrapper">
                <input type="number" class="form-control form-control-sm" id="end_m_${groupId}" min="0" max="59" value="0" oninput="redrawCanvas(); updateAnnotationInfo();">
                <span class="time-input-unit">分</span>
            </div>
            <div class="time-separator">:</div>
            <div class="time-input-wrapper">
                <input type="number" class="form-control form-control-sm" id="end_s_${groupId}" min="0" max="59" value="0" oninput="redrawCanvas(); updateAnnotationInfo();">
                <span class="time-input-unit">秒</span>
            </div>
        </div>
    `;
    container.appendChild(div);
}

// 读取页面上的 DOM 时间以供画布或上传使用
function getTimeRangeStr(otherId) {
    const sh = document.getElementById(`start_h_${otherId}`);
    const sm = document.getElementById(`start_m_${otherId}`);
    const ss = document.getElementById(`start_s_${otherId}`);
    const eh = document.getElementById(`end_h_${otherId}`);
    const em = document.getElementById(`end_m_${otherId}`);
    const es = document.getElementById(`end_s_${otherId}`);
    if (sh && sm && ss && eh && em && es) {
        return `${sh.value}:${sm.value}:${ss.value}-${eh.value}:${em.value}:${es.value}`;
    }
    return "";
}

function initDrawing() {
    const canvas = document.getElementById('mask-canvas');
    const newCanvas = canvas.cloneNode(true);
    canvas.parentNode.replaceChild(newCanvas, canvas);
    const ctx = newCanvas.getContext('2d');

    newCanvas.addEventListener('mousedown', async function(e) {
        if (!isDrawingMode) return;

        const rect = newCanvas.getBoundingClientRect();
        startX = e.clientX - rect.left;
        startY = e.clientY - rect.top;

        const type = document.getElementById('regionType').value;
        const mode = document.getElementById('drawMode').value;

        // 如果是"其他区域"，且当前试图点魔法棒或小框，必须先有大框做容器才能产生 UI 面板
        if (type === 'other') {
            if (mode !== 'large' && otherRegionCounter === 0) {
                alert('请先使用【区域框】(红色整体) 圈定该"其他区域"的大致范围，系统将为您自动生成对应的时间设置项！');
                isDrawingMode = false;
                toggleDrawingMode(false);
                return;
            }
        }

        // ====== 魔法棒模式 ======
        if (mode === 'magic') {
            const btn = document.getElementById('start-drawing-btn');
            const originalText = btn.innerHTML;

            try {
                btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>智能识别中...';
                newCanvas.style.cursor = 'wait';

                const realCoords = displayToImageCoordinates(startX, startY);
                const maskImage = document.getElementById('mask-image');

                const response = await fetch('/api/auto-segment', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ image_data: maskImage.src, x: realCoords.x, y: realCoords.y })
                });
                const result = await response.json();

                if (response.ok && result.success && result.polygon && result.polygon.length > 0) {
                    maskRects.push({
                        id: ++boxIdCounter,
                        type: type,
                        mode: mode,
                        polygon: result.polygon,
                        otherId: type === 'other' ? otherRegionCounter : null // 魔法棒自动认领最新的红框父亲
                    });
                    updateAnnotationInfo();
                    redrawCanvas(newCanvas);
                } else { alert('未识别到明显对象，请换个点重试'); }
            } catch (error) { alert('智能识别请求失败: ' + error.message); }
            finally {
                btn.innerHTML = originalText;
                newCanvas.style.cursor = 'crosshair';
            }
            return;
        }

        // ======= 矩形框逻辑 =======
        isDrawing = true;
        currentRect = { id: ++boxIdCounter, x: startX, y: startY, width: 0, height: 0, type: type, mode: mode };
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

        if (currentRect.width < 0) { currentRect.x += currentRect.width; currentRect.width = Math.abs(currentRect.width); }
        if (currentRect.height < 0) { currentRect.y += currentRect.height; currentRect.height = Math.abs(currentRect.height); }

        if (Math.abs(currentRect.width) > 0 && Math.abs(currentRect.height) > 0) {
            if (currentRect.type === 'other') {
                if (currentRect.mode === 'large') {
                    otherRegionCounter++;
                    currentRect.otherId = otherRegionCounter;
                    addOtherTimeInput(otherRegionCounter);
                } else {
                    currentRect.otherId = otherRegionCounter;
                }
            }
            maskRects.push(currentRect);
            updateAnnotationInfo();
        }
        currentRect = null;
        redrawCanvas(newCanvas);
    });
}

function redrawCanvas(canvasElement) {
    const canvas = canvasElement || document.getElementById('mask-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    maskRects.forEach(box => { drawBox(ctx, box); });
    if (isDrawing && currentRect) { drawBox(ctx, currentRect, true); }
}

function drawBox(ctx, box, isDashed = false) {
    // 动态拉取当前 UI 上填写的最新时间
    let timeStr = "";
    if (box.type === 'other' && box.otherId) {
        timeStr = getTimeRangeStr(box.otherId);
    }

    // 【完美体验】只在大红框展示时间和编号
    const isLarge = box.mode === 'large';
    const labelTitle = (box.type === 'other')
        ? `其他区域${box.otherId} [${timeStr.replace('-', ' 至 ')}]`
        : `${getRegionTypeName(box.type)}-${box.id}`;

    // 魔法棒多边形：纯粹的半透明紫，【坚决不画任何文字】防止画面被遮挡
    if (box.mode === 'magic' && box.polygon) {
        ctx.strokeStyle = '#9b59b6';
        ctx.lineWidth = 2;
        ctx.fillStyle = 'rgba(155, 89, 182, 0.3)';
        ctx.beginPath();
        box.polygon.forEach((pt, index) => {
            const dispX = pt[0] * imageScale;
            const dispY = pt[1] * imageScale;
            if (index === 0) ctx.moveTo(dispX, dispY); else ctx.lineTo(dispX, dispY);
        });
        ctx.closePath(); ctx.fill(); ctx.stroke();
        return;
    }

    const strokeColor = isLarge ? '#e74c3c' : '#27ae60';
    const fillColor = isLarge ? 'rgba(231, 76, 60, 0.2)' : 'rgba(39, 174, 96, 0.2)';

    ctx.strokeStyle = strokeColor;
    ctx.lineWidth = 2;
    ctx.fillStyle = fillColor;
    if (isDashed) ctx.setLineDash([5, 5]); else ctx.setLineDash([]);
    ctx.fillRect(box.x, box.y, box.width, box.height);
    ctx.strokeRect(box.x, box.y, box.width, box.height);

    // 只有红色大区域框才显示半透明的高级文字标签
    if (!isDashed && isLarge) {
        ctx.fillStyle = 'rgba(0, 0, 0, 0.4)'; // 柔和的半透明磨砂黑底
        const textWidth = ctx.measureText(labelTitle).width + 10;
        ctx.fillRect(box.x, box.y - 20, textWidth, 20);

        ctx.fillStyle = 'rgba(255, 255, 255, 0.95)'; // 高清晰白字
        ctx.font = '12px Arial';
        ctx.textBaseline = 'bottom';
        ctx.fillText(labelTitle, box.x + 5, box.y - 5);
    }
}

function getRegionTypeName(type) {
    const map = { 'logo': '台标', 'subtitle': '字幕', 'title': '剧名', 'other': '其他区域' };
    return map[type] || type;
}

function updateAnnotationInfo() {
    const details = document.getElementById('annotationDetails');
    if (maskRects.length === 0) { details.innerHTML = '尚未开始标注'; return; }

    const counts = {};
    maskRects.forEach(box => {
        let typeKey = box.type;
        let displayName = getRegionTypeName(box.type);

        if (box.type === 'other') {
            typeKey = `other_${box.otherId}`;
            const timeStr = getTimeRangeStr(box.otherId);
            displayName = `其他区域${box.otherId} [${timeStr.replace('-', ' 至 ')}]`;
        }

        if (!counts[typeKey]) counts[typeKey] = { name: displayName, large: 0, small: 0, magic: 0 };
        if (box.mode === 'large') counts[typeKey].large++;
        else if (box.mode === 'magic') counts[typeKey].magic++;
        else counts[typeKey].small++;
    });

    let html = '';
    Object.keys(counts).forEach(key => {
        const c = counts[key];
        html += `<div class="mb-1">
            <span class="badge bg-secondary me-1">${c.name}</span>
            <span class="text-danger small">大框:${c.large}</span> 
            <span class="text-success small">小框:${c.small}</span>
            ${c.magic > 0 ? `<span class="text-primary small">智能:${c.magic}</span>` : ''}
        </div>`;
    });
    details.innerHTML = html;
}

function undoLastBox() {
    if (maskRects.length > 0) {
        const removed = maskRects.pop();
        // 撤销连坐机制：如果撤销了红框，连带撤销左侧生成的UI时间面板
        if (removed.type === 'other' && removed.mode === 'large') {
            const container = document.getElementById(`time_container_${removed.otherId}`);
            if (container) container.remove();
            if (removed.otherId === otherRegionCounter) otherRegionCounter--;
        }
        redrawCanvas();
        updateAnnotationInfo();
    }
}

function clearAllBoxes() {
    if (maskRects.length === 0) return;
    if (confirm('确定要清空所有标注框吗？')) {
        maskRects = [];
        otherRegionCounter = 0;
        document.getElementById('dynamicTimeContainers').innerHTML = '';
        redrawCanvas();
        updateAnnotationInfo();
    }
}

// ====== JSON 生成（自动归拢时间） ======
async function generateJSON() {
    if (maskRects.length === 0) { alert('没有标注数据，请先画框标注'); return; }

    const maskImage = document.getElementById('mask-image');
    const jsonData = {
        metadata: {
            timestamp: new Date().toISOString(),
            videoFilename: currentVideoFilename,
            originalResolution: { width: maskImage.naturalWidth, height: maskImage.naturalHeight }
        },
        regions: {}
    };

    maskRects.forEach(box => {
        let regionKey = box.type;
        let typeName = getRegionTypeName(box.type);
        let timeRange = "";

        if (box.type === 'other') {
            regionKey = `other_${box.otherId}`;
            typeName = `其他区域${box.otherId}`;
            timeRange = getTimeRangeStr(box.otherId); // 取出 "0:3:23-0:4:23" 给后端解析
        }

        if (!jsonData.regions[regionKey]) {
            jsonData.regions[regionKey] = { typeName: typeName, largeBoxes: [], smallBoxes: [] };
            if (box.type === 'other') jsonData.regions[regionKey].timeRange = timeRange;
        }

        if (box.mode === 'magic') {
            jsonData.regions[regionKey].smallBoxes.push({ id: box.id, polygon: box.polygon });
        } else {
            const realCoords = displayToImageCoordinates(box.x, box.y);
            const boxData = { id: box.id, x: realCoords.x, y: realCoords.y, width: Math.round(box.width / imageScale), height: Math.round(box.height / imageScale) };
            if (box.mode === 'large') jsonData.regions[regionKey].largeBoxes.push(boxData);
            else jsonData.regions[regionKey].smallBoxes.push(boxData);
        }
    });

    const btn = document.getElementById('generate-json-btn');
    const originalBtnText = btn.innerHTML;
    try {
        btn.disabled = true;
        btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>正在上传...';
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
            statusDiv.innerHTML = `<i class="fas fa-check-circle me-1"></i>Mask已保存至服务器`;
            alert(`Mask文件已成功上传至服务器！`);
        } else { throw new Error(result.error || '上传失败'); }
    } catch (error) { alert('Mask上传失败: ' + error.message); }
    finally { btn.disabled = false; btn.innerHTML = originalBtnText; }
}

function resetCanvas() {
    maskRects = [];
    currentRect = null;
    boxIdCounter = 0;
    otherRegionCounter = 0;
    uploadedMaskFilename = null;

    document.getElementById('mask-upload-status').style.display = 'none';
    document.getElementById('dynamicTimeContainers').innerHTML = '';

    const maskImage = document.getElementById('mask-image');
    const maskCanvas = document.getElementById('mask-canvas');
    const framePlaceholder = document.getElementById('frame-placeholder');

    if(maskImage) { maskImage.src = ''; maskImage.style.display = 'none'; maskImage.classList.remove('active'); }
    if(maskCanvas) { maskCanvas.style.display = 'none'; maskCanvas.classList.remove('active'); }
    if(framePlaceholder) framePlaceholder.style.display = 'block';

    const btn = document.getElementById('start-drawing-btn');
    if(btn) toggleDrawingMode(false);
    updateAnnotationInfo();
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
    progressBar.textContent = task.progress + '%';
    document.getElementById('progress-text').textContent = getStageText(task.current_stage) + ' ' + task.progress + '%';
    document.getElementById('download-video-btn').disabled = (task.status !== 'completed');
}

function getStageText(stage) {
    const stageMap = { 'initialization': '初始化', 'scene_detection': '场景检测', 'mask_generation': '掩码生成', 'video_processing': '视频处理', 'finalizing': '最终处理', 'completed': '完成' };
    return stageMap[stage] || stage || '准备开始';
}

document.addEventListener('DOMContentLoaded', function() {
    initializePage();
    document.getElementById('upload-btn').addEventListener('click', uploadVideo);
    document.getElementById('start-btn').addEventListener('click', startProcessing);
    document.getElementById('cancel-btn').addEventListener('click', cancelProcessing);
    document.getElementById('download-video-btn').addEventListener('click', downloadVideo);
    document.getElementById('extract-frame-btn').addEventListener('click', extractFrame);
    document.getElementById('start-drawing-btn').addEventListener('click', () => toggleDrawingMode());
    document.getElementById('undo-btn').addEventListener('click', undoLastBox);
    document.getElementById('clear-all-btn').addEventListener('click', clearAllBoxes);
    document.getElementById('generate-json-btn').addEventListener('click', generateJSON);

    document.getElementById('source-video').addEventListener('change', function() { currentVideoFilename = this.value; });

    const regionSelect = document.getElementById('regionType');
    const modeSelect = document.getElementById('drawMode');

    regionSelect.value = 'logo'; modeSelect.value = 'large';
    regionSelect.addEventListener('change', function() { modeSelect.value = 'large'; });
});