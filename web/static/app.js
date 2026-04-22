let currentTaskId = null;
let statusInterval = null;

// Mask制作相关变量
let isDrawingMode = false;
let isDrawing = false;
let startX, startY;
let currentRect = null;
let maskRects = [];
let boxIdCounter = 0;
let otherRegionCounter = 0;

let currentVideoFilename = '';
let uploadedMaskFilename = null;

// 【核心新增】全局保存服务器传回（或前端解析）的视频真实时长
let globalVideoDuration = 0;

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

    // 【核心新增】文件选择后，立刻在前端隐式获取其真实时长
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
    if (maskRects.length > 0 && !uploadedMaskFilename) {
        if(!confirm('检测到您绘制了Mask但尚未提交同步。直接处理将忽略这些标注。\n是否继续？')) return;
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

    // 【核心新增】越界拦截校验
    const totalSec = hours * 3600 + minutes * 60 + seconds;
    if (globalVideoDuration > 0 && totalSec > globalVideoDuration) {
        alert(`❌ 提取失败：您设定的提取时间点 (${totalSec}秒) 已超出视频的总时长 (${Math.floor(globalVideoDuration)}秒)！`);
        return;
    }

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
        btn.innerHTML = '<i class="fas fa-stop me-1"></i>停止绘制';
        canvas.style.cursor = 'crosshair';
    } else {
        btn.classList.remove('btn-danger'); btn.classList.add('btn-warning');
        btn.innerHTML = '<i class="fas fa-pen-nib me-1"></i>进入绘制模式';
        canvas.style.cursor = 'default';
    }
}

// 【重构】生成绝美的高级三联输入面板，并且绑定 onblur 校验机制
function addOtherTimeInput(groupId) {
    const container = document.getElementById('dynamicTimeContainers');
    const div = document.createElement('div');
    div.className = 'mt-2 p-2 bg-light border rounded';
    div.style.borderColor = '#dcdde1';
    div.id = `time_container_${groupId}`;

    div.innerHTML = `
        <div class="d-flex justify-content-between align-items-center mb-2">
            <span class="text-primary fw-bold" style="font-size: 13px;">■ 其他区域${groupId} 生效时效</span>
        </div>
        <div class="mb-1"><span class="small text-muted" style="font-size:11px; font-weight:bold;">起始时间:</span></div>
        <div class="time-input-group mb-2">
            <div class="time-input-wrapper">
                <input type="number" class="form-control form-control-sm" id="start_h_${groupId}" min="0" value="0" oninput="redrawCanvas();" onblur="validateTimeRange(${groupId})">
                <span class="time-input-unit">时</span>
            </div>
            <div class="time-separator">:</div>
            <div class="time-input-wrapper">
                <input type="number" class="form-control form-control-sm" id="start_m_${groupId}" min="0" max="59" value="0" oninput="redrawCanvas();" onblur="validateTimeRange(${groupId})">
                <span class="time-input-unit">分</span>
            </div>
            <div class="time-separator">:</div>
            <div class="time-input-wrapper">
                <input type="number" class="form-control form-control-sm" id="start_s_${groupId}" min="0" max="59" value="0" oninput="redrawCanvas();" onblur="validateTimeRange(${groupId})">
                <span class="time-input-unit">秒</span>
            </div>
        </div>
        <div class="mb-1 mt-1"><span class="small text-muted" style="font-size:11px; font-weight:bold;">终止时间:</span></div>
        <div class="time-input-group mb-1">
            <div class="time-input-wrapper">
                <input type="number" class="form-control form-control-sm" id="end_h_${groupId}" min="0" value="0" oninput="redrawCanvas();" onblur="validateTimeRange(${groupId})">
                <span class="time-input-unit">时</span>
            </div>
            <div class="time-separator">:</div>
            <div class="time-input-wrapper">
                <input type="number" class="form-control form-control-sm" id="end_m_${groupId}" min="0" max="59" value="0" oninput="redrawCanvas();" onblur="validateTimeRange(${groupId})">
                <span class="time-input-unit">分</span>
            </div>
            <div class="time-separator">:</div>
            <div class="time-input-wrapper">
                <input type="number" class="form-control form-control-sm" id="end_s_${groupId}" min="0" max="59" value="0" oninput="redrawCanvas();" onblur="validateTimeRange(${groupId})">
                <span class="time-input-unit">秒</span>
            </div>
        </div>
    `;
    container.appendChild(div);
}

// 【核心新增】当用户移出输入框时(blur)，严格校验时长关系
window.validateTimeRange = function(groupId) {
    // 1. 获取输入值
    const sh = parseInt(document.getElementById(`start_h_${groupId}`).value) || 0;
    const sm = parseInt(document.getElementById(`start_m_${groupId}`).value) || 0;
    const ss = parseInt(document.getElementById(`start_s_${groupId}`).value) || 0;
    let eh = parseInt(document.getElementById(`end_h_${groupId}`).value) || 0;
    let em = parseInt(document.getElementById(`end_m_${groupId}`).value) || 0;
    let es = parseInt(document.getElementById(`end_s_${groupId}`).value) || 0;

    const startTotal = sh * 3600 + sm * 60 + ss;
    let endTotal = eh * 3600 + em * 60 + es;

    // 如果都没填，先放过
    if (startTotal === 0 && endTotal === 0) return;

    // 2. 终止时间越界拦截 & 自动 Clamp 钳制修正
    if (globalVideoDuration > 0 && endTotal > globalVideoDuration) {
        endTotal = Math.floor(globalVideoDuration);
        document.getElementById(`end_h_${groupId}`).value = Math.floor(endTotal / 3600);
        document.getElementById(`end_m_${groupId}`).value = Math.floor((endTotal % 3600) / 60);
        document.getElementById(`end_s_${groupId}`).value = endTotal % 60;
        alert(`温馨提示：终止时间超出了视频总长度，系统已为您自动修正为视频片尾时刻 (${endTotal}秒)。`);
    }

    // 3. 起始时间越界拦截
    if (globalVideoDuration > 0 && startTotal >= globalVideoDuration) {
        alert(`❌ 错误：您的开始时间不能超出视频的总时长 (${Math.floor(globalVideoDuration)}秒)！`);
    }

    // 4. 逻辑悖论拦截：开始 >= 结束
    if (startTotal > 0 && endTotal > 0 && startTotal >= endTotal) {
        alert("❌ 错误：生效的【起始时间】必须严格小于【终止时间】！请重新输入。");
    }

    // 强制触发画布同步和统计刷新
    redrawCanvas();
    updateAnnotationInfo();
};

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

        if (type === 'other') {
            if (mode !== 'large' && otherRegionCounter === 0) {
                alert('请先使用【区域框】(红框) 圈定该"其他区域"的大致范围，系统将为您自动生成对应的时间设置面板！');
                isDrawingMode = false;
                toggleDrawingMode(false);
                return;
            }
        }

        if (mode === 'magic') {
            const btn = document.getElementById('start-drawing-btn');
            const originalText = btn.innerHTML;

            try {
                btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>正在提交AI运算...';
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
                        otherId: type === 'other' ? otherRegionCounter : null
                    });
                    updateAnnotationInfo();
                    redrawCanvas(newCanvas);
                } else { alert('未识别到明显对象，请尝试在目标边缘点选'); }
            } catch (error) { alert('AI推理节点通信失败: ' + error.message); }
            finally {
                btn.innerHTML = originalText;
                newCanvas.style.cursor = 'crosshair';
            }
            return;
        }

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
    let timeStr = "";
    if (box.type === 'other' && box.otherId) {
        timeStr = getTimeRangeStr(box.otherId);
    }

    const isLarge = box.mode === 'large';
    const labelTitle = (box.type === 'other')
        ? `其他区域${box.otherId} [${timeStr.replace('-', ' 至 ')}]`
        : `${getRegionTypeName(box.type)}-${box.id}`;

    if (box.mode === 'magic' && box.polygon) {
        ctx.strokeStyle = '#9b59b6';
        ctx.lineWidth = 2;
        ctx.fillStyle = 'rgba(155, 89, 182, 0.35)';
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
    const fillColor = isLarge ? 'rgba(231, 76, 60, 0.15)' : 'rgba(39, 174, 96, 0.2)';

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

function getRegionTypeName(type) {
    const map = { 'logo': '台标', 'subtitle': '字幕', 'title': '剧名', 'other': '其他区域' };
    return map[type] || type;
}

function updateAnnotationInfo() {
    const details = document.getElementById('annotationDetails');
    if (maskRects.length === 0) { details.innerHTML = '尚未生成有效标注'; return; }

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
            <span class="badge" style="background-color: var(--primary); margin-right:4px;">${c.name}</span>
            <span class="text-danger small fw-bold">红框:${c.large}</span> 
            ${c.magic > 0 ? `<span class="text-primary small fw-bold" style="margin-left:4px;">魔法棒:${c.magic}</span>` : ''}
        </div>`;
    });
    details.innerHTML = html;
}

function undoLastBox() {
    if (maskRects.length > 0) {
        const removed = maskRects.pop();
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
    if (confirm('危险操作：确定要清空画布上所有的标注框吗？')) {
        maskRects = [];
        otherRegionCounter = 0;
        document.getElementById('dynamicTimeContainers').innerHTML = '';
        redrawCanvas();
        updateAnnotationInfo();
    }
}

async function generateJSON() {
    if (maskRects.length === 0) { alert('画布为空，无法生成标注集'); return; }

    const maskImage = document.getElementById('mask-image');
    const jsonData = {
        metadata: {
            timestamp: new Date().toISOString(),
            videoFilename: currentVideoFilename,
            originalResolution: { width: maskImage.naturalWidth, height: maskImage.naturalHeight }
        },
        regions: {}
    };

    let hasTimeError = false;

    maskRects.forEach(box => {
        let regionKey = box.type;
        let typeName = getRegionTypeName(box.type);
        let timeRange = "";

        if (box.type === 'other') {
            regionKey = `other_${box.otherId}`;
            typeName = `其他区域${box.otherId}`;
            timeRange = getTimeRangeStr(box.otherId);
            // 提交时进行最终严格校验
            const parts = timeRange.split('-');
            if (parts.length !== 2 || parts[0] === "0:0:0" && parts[1] === "0:0:0") {
                hasTimeError = true;
            }
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

    if (hasTimeError) {
        alert('上传拦截：您有"其他区域"未设置有效的起止时间。请设定正确的时效再试！');
        return;
    }

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
            statusDiv.innerHTML = `<i class="fas fa-check-circle me-1"></i>标注集已成功挂载至引擎`;
            alert(`标注提交成功！可以启动分布式引擎进行清洗了。`);
        } else { throw new Error(result.error || '上传失败'); }
    } catch (error) { alert('云端网络异常: ' + error.message); }
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
    document.getElementById('undo-btn').addEventListener('click', undoLastBox);
    document.getElementById('clear-all-btn').addEventListener('click', clearAllBoxes);
    document.getElementById('generate-json-btn').addEventListener('click', generateJSON);

    document.getElementById('source-video').addEventListener('change', function() { currentVideoFilename = this.value; });

    const regionSelect = document.getElementById('regionType');
    const modeSelect = document.getElementById('drawMode');

    regionSelect.value = 'logo'; modeSelect.value = 'large';
    regionSelect.addEventListener('change', function() { modeSelect.value = 'large'; });
});