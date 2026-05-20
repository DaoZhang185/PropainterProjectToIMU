let currentTaskId = null;
let statusInterval = null;

let isDrawingMode = false;
let isDrawing = false;
let startX, startY;
let currentRect = null;
let maskRects = [];
let boxIdCounter = 0;
let otherRegionCounter = 0;

let currentVideoFilename = '';
let uploadedMaskFilename = null;

let globalVideoDuration = 0;
let currentExtractTimeStr = "00:00:00";
let isMnSubEnabled = false; // 【新增】记录是否开启了蒙文字幕模式
let samSession = { active: false, pos: [], neg: [], polygons: null, history: [] };

let imageScale = 1;
let imageOffsetX = 0;
let imageOffsetY = 0;

const FIXED_SCRIPT_PARAMS = {
    scene_threshold: 20.0, min_duration: 0.5, max_duration: 5.0,
    scale: 0.25, output_scale: 1.0, max_frames: 0, output_fps: 0,
    convert_to_mp4: true, max_workers: 1, enable_parallel: true,
    batch_size: 1, keep_intermediate: true
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
    finally { btn.disabled = false; btn.innerHTML = originalText; }
}

async function startProcessing() {
    const sourceVideo = document.getElementById('source-video').value;
    if (!sourceVideo) { alert('请设置源视频文件名'); return; }

    const activeMasks = maskRects.filter(b => b.visible);
    if (activeMasks.length > 0 && !uploadedMaskFilename) {
        if(!confirm('检测到您绘制了图层但尚未提交数据。\n是否继续？')) return;
    }

    const scriptParams = { ...FIXED_SCRIPT_PARAMS, source_video: currentVideoFilename, mask_file: uploadedMaskFilename };
    try {
        const response = await fetch('/api/tasks', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ script_params: scriptParams })
        });
        const result = await response.json();
        if (response.ok) {
            currentTaskId = result.task_id;
            document.getElementById('cancel-btn').disabled = false;
            document.getElementById('download-video-btn').disabled = true;
            startStatusPolling();
        } else { alert('任务启动失败'); }
    } catch (error) { alert('任务错误: ' + error.message); }
}

async function cancelProcessing() {
    if (!currentTaskId) return;
    try { await fetch(`/api/tasks/${currentTaskId}/cancel`, { method: 'POST' }); document.getElementById('cancel-btn').disabled = true; } catch (error) {}
}

async function downloadVideo() {
    if (!currentTaskId) return;
    try {
        const response = await fetch(`/api/tasks/${currentTaskId}/download-video`);
        if (response.ok) {
            const blob = await response.blob(); const url = window.URL.createObjectURL(blob);

            const a = document.createElement('a'); a.href = url;
            a.download = isMnSubEnabled ? '处理后蒙文字幕视频.mp4' : '处理后纯净视频.mp4';
            document.body.appendChild(a); a.click(); window.URL.revokeObjectURL(url); document.body.removeChild(a);
        }
    } catch (error) {}
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
    try {
        btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>提取中...';
        const response = await fetch('/api/extract-frame', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ video_filename: currentVideoFilename, timestamp: currentExtractTimeStr })
        });
        const result = await response.json();
        if (response.ok && result.success) {
            const maskImage = document.getElementById('mask-image');
            document.getElementById('frame-placeholder').style.display = 'none';
            maskImage.src = result.image_data;
            maskImage.style.display = 'block'; maskImage.classList.add('active');
            await new Promise((resolve) => {
                maskImage.onload = function() { calculateImageDisplay(maskImage); initDrawing(); toggleDrawingMode(true); resolve(); };
            });
        }
    } catch (error) {} finally { btn.disabled = false; btn.innerHTML = originalText; }
}

function calculateImageDisplay(maskImage) {
    const container = document.getElementById('mask-container');
    const cw = container.clientWidth, ch = container.clientHeight;
    const iw = maskImage.naturalWidth, ih = maskImage.naturalHeight;
    imageScale = Math.min(cw / iw, ch / ih, 1);
    const dw = iw * imageScale, dh = ih * imageScale;
    imageOffsetX = (cw - dw) / 2; imageOffsetY = (ch - dh) / 2;

    maskImage.style.width = `${dw}px`; maskImage.style.height = `${dh}px`;
    maskImage.style.left = `${imageOffsetX}px`; maskImage.style.top = `${imageOffsetY}px`;
    const maskCanvas = document.getElementById('mask-canvas');
    maskCanvas.style.width = `${dw}px`; maskCanvas.style.height = `${dh}px`;
    maskCanvas.style.left = `${imageOffsetX}px`; maskCanvas.style.top = `${imageOffsetY}px`;
    maskCanvas.width = dw; maskCanvas.height = dh;
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
        btn.classList.replace('btn-warning', 'btn-danger'); btn.innerHTML = '<i class="fas fa-stop me-1"></i>停止绘制';
        document.getElementById('undo-btn').style.display = 'block';
        if(canvas) canvas.style.cursor = 'crosshair';
    } else {
        btn.classList.replace('btn-danger', 'btn-warning'); btn.innerHTML = '<i class="fas fa-pen-nib me-1"></i>进入绘制模式';
        document.getElementById('undo-btn').style.display = 'none';
        if(canvas) canvas.style.cursor = 'default';
        if (samSession.active) {
            samSession = { active: false, pos: [], neg: [], polygons: null, history: [] };
            document.getElementById('commit-sam-btn').style.display = 'none';
            document.getElementById('undo-btn').innerHTML = '<i class="fas fa-rotate-left me-1"></i>撤销';
            redrawCanvas();
        }
    }
}

function initDrawing() {
    const canvas = document.getElementById('mask-canvas');
    const newCanvas = canvas.cloneNode(true);
    canvas.parentNode.replaceChild(newCanvas, canvas);
    newCanvas.addEventListener('contextmenu', e => e.preventDefault());

    newCanvas.addEventListener('mousedown', async function(e) {
        if (!isDrawingMode) return;
        const rect = newCanvas.getBoundingClientRect();
        startX = e.clientX - rect.left; startY = e.clientY - rect.top;
        const type = document.getElementById('regionType').value;
        const mode = document.getElementById('drawMode').value;

        // 【保留】字幕位置框唯一性拦截
        if (type === 'sub_pos') {
            if (maskRects.some(b => b.type === 'sub_pos')) {
                alert('⚠️ 规则限制：只能绘制一个【蒙文字幕位置】的框！如果想修改，请先撤销或删除旧框。');
                isDrawingMode = false; toggleDrawingMode(false); return;
            }
        }

        if (mode === 'magic' && type !== 'sub_pos') {
            samSession.active = true;
            document.getElementById('commit-sam-btn').style.display = 'block';
            document.getElementById('undo-btn').innerHTML = '<i class="fas fa-rotate-left me-1"></i>撤销点击';

            const realCoords = displayToImageCoordinates(startX, startY);
            if (e.altKey) { samSession.neg.push([realCoords.x, realCoords.y]); samSession.history.push('neg'); }
            else { samSession.pos.push([realCoords.x, realCoords.y]); samSession.history.push('pos'); }
            await triggerSAM(newCanvas);
            return;
        }

        isDrawing = true;
        currentRect = { id: ++boxIdCounter, x: startX, y: startY, width: 0, height: 0, type: type, mode: mode, visible: true, ref_time: currentExtractTimeStr };
    });

    newCanvas.addEventListener('mousemove', function(e) {
        if (!isDrawing || !currentRect || currentRect.mode === 'magic') return;
        const rect = newCanvas.getBoundingClientRect();
        currentRect.width = (e.clientX - rect.left) - startX; currentRect.height = (e.clientY - rect.top) - startY;
        redrawCanvas(newCanvas);
    });

    newCanvas.addEventListener('mouseup', function(e) {
        if (!isDrawing || !currentRect || currentRect.mode === 'magic') return;
        isDrawing = false;
        if (currentRect.width < 0) { currentRect.x += currentRect.width; currentRect.width = Math.abs(currentRect.width); }
        if (currentRect.height < 0) { currentRect.y += currentRect.height; currentRect.height = Math.abs(currentRect.height); }
        if (currentRect.width > 0 && currentRect.height > 0) {
            if (currentRect.type === 'other') {
                otherRegionCounter++; currentRect.otherId = otherRegionCounter;
            }
            maskRects.push(currentRect);
            updateLayerPanel();
        }
        currentRect = null; redrawCanvas(newCanvas);
    });
}

document.getElementById('undo-btn').addEventListener('click', async () => {
    if (samSession.active) {
        if (samSession.history.length > 0) {
            const lastType = samSession.history.pop();
            if (lastType === 'pos') samSession.pos.pop(); else samSession.neg.pop();
            if (samSession.pos.length === 0 && samSession.neg.length === 0) {
                samSession.polygons = null; redrawCanvas();
            } else { await triggerSAM(document.getElementById('mask-canvas')); }
        }
    } else {
        if (maskRects.length > 0) {
            const removed = maskRects.pop();
            if (removed.type === 'other' && removed.otherId === otherRegionCounter) otherRegionCounter--;
            redrawCanvas(); updateLayerPanel();
        }
    }
});

async function triggerSAM(canvas) {
    canvas.style.cursor = 'wait';
    try {
        const maskImage = document.getElementById('mask-image');
        const extractMode = document.getElementById('extractMode').value;

        const response = await fetch('/api/auto-segment', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ image_data: maskImage.src, pos_points: samSession.pos, neg_points: samSession.neg, extract_mode: extractMode })
        });
        const result = await response.json();
        if (response.ok && result.success && result.polygons && result.polygons.length > 0) {
            samSession.polygons = result.polygons;
        } else { samSession.polygons = null; }
        redrawCanvas(canvas);
    } catch (error) {} finally { canvas.style.cursor = 'crosshair'; }
}

document.getElementById('start-drawing-btn').insertAdjacentHTML('afterend', `
    <button class="btn btn-success mt-1" id="commit-sam-btn" style="display:none;"><i class="fas fa-check-double me-1"></i>确认抠图</button>
`);

document.getElementById('commit-sam-btn').addEventListener('click', () => {
    if (samSession.polygons) {
        const type = document.getElementById('regionType').value;
        const mode = document.getElementById('extractMode').value;
        maskRects.push({
            id: ++boxIdCounter, type: type, mode: 'magic', extractMode: mode, polygons: samSession.polygons,
            visible: true, ref_time: currentExtractTimeStr,
            otherId: type === 'other' ? ++otherRegionCounter : null
        });
        updateLayerPanel();
    }
    samSession = { active: false, pos: [], neg: [], polygons: null, history: [] };
    document.getElementById('commit-sam-btn').style.display = 'none';
    document.getElementById('undo-btn').innerHTML = '<i class="fas fa-rotate-left me-1"></i>撤销';
    redrawCanvas();
});

function getRegionTypeName(type) {
    const map = { 'logo': '台标', 'title': '剧名', 'subtitle': '字幕', 'other': '自定义区域', 'sub_pos': '蒙文字幕投放区' };
    return map[type] || type;
}

window.toggleGroup = function(type) {
    const groupBoxes = maskRects.filter(b => b.type === type);
    if (groupBoxes.length > 0) {
        const newState = !groupBoxes[0].visible;
        groupBoxes.forEach(b => b.visible = newState);
        redrawCanvas(); updateLayerPanel();
    }
};

window.deleteGroup = function(type) {
    if(confirm(`确定要删除所有的【${getRegionTypeName(type)}】吗？`)){
        maskRects = maskRects.filter(b => b.type !== type);
        redrawCanvas(); updateLayerPanel();
    }
};

function updateLayerPanel() {
    const panel = document.getElementById('layerPanel');
    if (maskRects.length === 0) { panel.innerHTML = '<div class="text-muted text-center" style="font-size:11px; margin-top:30px;">暂无图层</div>'; return; }

    const grouped = {};
    maskRects.forEach(box => {
        if (!grouped[box.type]) grouped[box.type] = { type: box.type, ref_time: box.ref_time, visible: box.visible, elements: [] };
        grouped[box.type].elements.push(box);
        if(!box.visible) grouped[box.type].visible = false;
    });

    let html = '';
    Object.values(grouped).forEach(group => {
        const eyeIcon = group.visible ? 'fa-eye text-primary' : 'fa-eye-slash text-muted';
        const opacity = group.visible ? '1' : '0.5';
        html += `
        <div class="d-flex justify-content-between align-items-center mb-1 pb-1" style="border-bottom:1px solid #f0f0f0; font-size:12px; opacity:${opacity};">
            <div>
                <i class="fas ${eyeIcon} me-2" style="cursor:pointer;" onclick="toggleGroup('${group.type}')"></i>
                <span style="font-weight:600; color:var(--dark);">${getRegionTypeName(group.type)}</span>
                <span style="color:var(--gray-text); margin-left:4px;">(共 ${group.elements.length} 笔)</span>
            </div>
            <i class="fas fa-times text-danger" style="cursor:pointer;" onclick="deleteGroup('${group.type}')"></i>
        </div>`;
    });
    panel.innerHTML = html;
}

function redrawCanvas(canvasElement) {
    const canvas = canvasElement || document.getElementById('mask-canvas');
    if (!canvas) return;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    maskRects.forEach(box => { if (box.visible) drawBox(ctx, box); });
    if (isDrawing && currentRect) drawBox(ctx, currentRect, true);

    if (samSession.active) {
        if (samSession.polygons) {
            ctx.strokeStyle = '#9b59b6'; ctx.lineWidth = 2; ctx.fillStyle = 'rgba(155, 89, 182, 0.4)';
            samSession.polygons.forEach(poly => {
                ctx.beginPath();
                poly.forEach((pt, index) => {
                    const dispX = pt[0] * imageScale; const dispY = pt[1] * imageScale;
                    if (index === 0) ctx.moveTo(dispX, dispY); else ctx.lineTo(dispX, dispY);
                });
                ctx.closePath(); ctx.fill(); ctx.stroke();
            });
        }
        ctx.fillStyle = '#2ecc71';
        samSession.pos.forEach(pt => { ctx.beginPath(); ctx.arc(pt[0] * imageScale, pt[1] * imageScale, 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); });
        ctx.fillStyle = '#e74c3c';
        samSession.neg.forEach(pt => { ctx.beginPath(); ctx.arc(pt[0] * imageScale, pt[1] * imageScale, 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke(); });
    }
}

function drawBox(ctx, box, isDashed = false) {
    if (box.mode === 'magic' && box.polygons) {
        const color = box.extractMode === 'solid' ? '52, 152, 219' : '230, 126, 34';
        ctx.strokeStyle = `rgb(${color})`; ctx.lineWidth = 2; ctx.fillStyle = `rgba(${color}, 0.3)`;
        box.polygons.forEach(poly => {
            ctx.beginPath();
            poly.forEach((pt, index) => {
                const dispX = pt[0] * imageScale; const dispY = pt[1] * imageScale;
                if (index === 0) ctx.moveTo(dispX, dispY); else ctx.lineTo(dispX, dispY);
            });
            ctx.closePath(); ctx.fill(); ctx.stroke();
        });
        return;
    }
    const isLarge = box.mode === 'large';

    const strokeColor = box.type === 'sub_pos' ? '#9b59b6' : '#e74c3c';
    const fillColor = box.type === 'sub_pos' ? 'rgba(155, 89, 182, 0.15)' : 'rgba(231, 76, 60, 0.15)';

    ctx.strokeStyle = strokeColor; ctx.lineWidth = 2; ctx.fillStyle = fillColor;
    if (isDashed) ctx.setLineDash([5, 5]); else ctx.setLineDash([]);
    ctx.fillRect(box.x, box.y, box.width, box.height); ctx.strokeRect(box.x, box.y, box.width, box.height);

    if (!isDashed && isLarge) {
        const labelTitle = getRegionTypeName(box.type);
        ctx.fillStyle = 'rgba(0, 0, 0, 0.55)';
        ctx.fillRect(box.x, box.y - 20, ctx.measureText(labelTitle).width + 10, 20);
        ctx.fillStyle = 'white'; ctx.font = '12px Arial'; ctx.textBaseline = 'bottom';
        ctx.fillText(labelTitle, box.x + 5, box.y - 5);
    }
}

function clearAllBoxes() {
    if (confirm('危险操作：确定要清空画布上所有的标注框吗？')) {
        maskRects = []; otherRegionCounter = 0;
        samSession = { active: false, pos: [], neg: [], polygons: null, history: [] };
        document.getElementById('commit-sam-btn').style.display = 'none';
        redrawCanvas(); updateLayerPanel();
    }
}

async function generateJSON() {
    const activeMasks = maskRects.filter(b => b.visible);
    if (activeMasks.length === 0) { alert('画布为空'); return; }

    const maskImage = document.getElementById('mask-image');
    const jsonData = {
        metadata: { timestamp: new Date().toISOString(), videoFilename: currentVideoFilename, originalResolution: { width: maskImage.naturalWidth, height: maskImage.naturalHeight } },
        regions: {}
    };

    activeMasks.forEach(box => {
        let regionKey = box.type === 'other' ? `other_${box.otherId}` : box.type;

        if (!jsonData.regions[regionKey]) {
            jsonData.regions[regionKey] = { typeName: getRegionTypeName(box.type), reference_time: box.ref_time, largeBoxes: [], smallBoxes: [] };
        }

        if (box.mode === 'magic') {
            box.polygons.forEach((poly, idx) => { jsonData.regions[regionKey].smallBoxes.push({ id: `${box.id}_${idx}`, polygon: poly }); });
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
        btn.disabled = true; btn.innerHTML = '<i class="fas fa-spinner fa-spin me-1"></i>正在打包同步...';
        const response = await fetch('/api/save-mask', {
            method: 'POST', headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ mask_data: jsonData, video_filename: currentVideoFilename })
        });
        const result = await response.json();
        if (response.ok) {
            uploadedMaskFilename = result.mask_filename;
            document.getElementById('mask-upload-status').style.display = 'block';
            alert(`标注提交成功！`);
        } else { throw new Error(result.error); }
    } catch (error) { alert('异常: ' + error.message); } finally { btn.disabled = false; btn.innerHTML = originalBtnText; }
}

function resetCanvas() {
    maskRects = []; currentRect = null; boxIdCounter = 0; otherRegionCounter = 0; uploadedMaskFilename = null;
    samSession = { active: false, pos: [], neg: [], polygons: null, history: [] };
    document.getElementById('mask-upload-status').style.display = 'none';
    const maskImage = document.getElementById('mask-image'); const maskCanvas = document.getElementById('mask-canvas');
    if(maskImage) { maskImage.src = ''; maskImage.style.display = 'none'; maskImage.classList.remove('active'); }
    if(maskCanvas) { maskCanvas.style.display = 'none'; maskCanvas.classList.remove('active'); }
    document.getElementById('frame-placeholder').style.display = 'block';
    toggleDrawingMode(false); updateLayerPanel();
}

function startStatusPolling() {
    if (statusInterval) clearInterval(statusInterval);
    statusInterval = setInterval(async () => {
        if (!currentTaskId) return;
        try {
            const response = await fetch(`/api/tasks/${currentTaskId}`);
            const task = await response.json();
            document.getElementById('progress-bar').style.width = task.progress + '%';

            // 【新增】加入字幕生成阶段中文提示
            const stageMap = { 'initialization': '分配算力', 'scene_detection': '光流分析', 'mask_generation': '渲染掩码', 'video_processing': '分布式推理', 'finalizing': '画面合成', 'subtitle_burning': '语音提取翻译及烧录', 'completed': '全自动产线清洗完毕' };
            document.getElementById('progress-text').textContent = (stageMap[task.current_stage] || '处理中') + ' ' + task.progress + '%';
            document.getElementById('download-video-btn').disabled = (task.status !== 'completed');
            if (task.status === 'completed' || task.status === 'failed' || task.status === 'cancelled') { clearInterval(statusInterval); document.getElementById('cancel-btn').disabled = true; }
        } catch (error) {}
    }, 2000);
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

    regionSelect.addEventListener('change', function() {
        if (this.value === 'sub_pos') { modeSelect.value = 'large'; }
        else { modeSelect.value = 'large'; }
        modeSelect.dispatchEvent(new Event('change'));
    });

    modeSelect.addEventListener('change', function() {
        const isMagic = this.value === 'magic';
        document.getElementById('magic-hint').style.display = isMagic ? 'block' : 'none';
        document.getElementById('magic-mode-container').style.display = isMagic ? 'block' : 'none';

        if (isMagic && regionSelect.value === 'sub_pos') {
            alert('⚠️ 蒙文字幕投放区仅用于定位，不支持魔法棒抠图，已为您自动切回红框。');
            this.value = 'large';
            this.dispatchEvent(new Event('change'));
        }
    });
    // =================================================================
    // 【新增】蒙文字幕开关按钮的交互逻辑
    // =================================================================
    document.getElementById('enable-mn-sub-btn').addEventListener('click', function() {
        isMnSubEnabled = !isMnSubEnabled;
        const subOption = document.getElementById('sub-pos-option');
        const downloadText = document.getElementById('download-text');
        const regionSelect = document.getElementById('regionType');

        if (isMnSubEnabled) {
            // 开启状态：显示下拉选项，改变下载文案，按钮变色
            subOption.style.display = 'block';
            subOption.removeAttribute('hidden');
            downloadText.innerText = '下载处理后的蒙文字幕视频';
            this.classList.replace('btn-outline-success', 'btn-success');
            this.innerHTML = '<i class="fas fa-times-circle me-1"></i>取消蒙文字幕';
        } else {
            // 关闭状态：隐藏下拉选项，恢复纯净文案，按钮复原
            subOption.style.display = 'none';
            subOption.setAttribute('hidden', 'true');
            downloadText.innerText = '下载处理后的纯净视频';
            this.classList.replace('btn-success', 'btn-outline-success');
            this.innerHTML = '<i class="fas fa-language me-1"></i>添加蒙文字幕';

            // 防呆机制 1：如果当前正好选中了蒙文选项，自动切回台标区域
            if (regionSelect.value === 'sub_pos') {
                regionSelect.value = 'logo';
                regionSelect.dispatchEvent(new Event('change'));
            }

            // 防呆机制 2：如果用户已经画了蒙文框，关闭开关时自动清除那个框
            const hasSubBox = maskRects.some(b => b.type === 'sub_pos');
            if (hasSubBox) {
                maskRects = maskRects.filter(b => b.type !== 'sub_pos');
                redrawCanvas();
                updateLayerPanel();
            }
        }
    });
});