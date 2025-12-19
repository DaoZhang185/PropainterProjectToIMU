#!/bin/bash

# =================================================================
# ProPainter 视频处理流程 - 音画分离 (修复音频编码兼容性)
# =================================================================

# 参数变量
SOURCE_VIDEO=""
SCENE_THRESHOLD="10.0"
MIN_SCENE_DURATION="0.2"
MAX_SCENE_DURATION="12.0"
SCALE="0.5"
OUTPUT_SCALE="1.0"
BASE_OUTPUT="/home/FeiLong_Grp/ZhangMengHui/Model/Model/Model/ProPainter/results/finalresults03"
FINAL_OUTPUT=""
MAX_FRAMES=""
OUTPUT_FPS=""
CONVERT_TO_MP4="true"
MAX_WORKERS="1"
ENABLE_PARALLEL="true"
BATCH_SIZE="1"
KEEP_INTERMEDIATE="false"
WORKING_DIR="/home/FeiLong_Grp/ZhangMengHui/Model/Model/Model/ProPainter"
CONDA_ENV="prolabel-env"
MASK_JSON=""

TIMESTAMP=$(date '+%Y%m%d_%H%M%S')
VIDEO_NAME="unknown"

# 解析参数
while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-video) SOURCE_VIDEO="$2"; VIDEO_NAME=$(basename "$SOURCE_VIDEO" | sed 's/\.[^.]*$//'); shift 2 ;;
    --base-output) BASE_OUTPUT="$2"; shift 2 ;;
    --final-output) FINAL_OUTPUT="$2"; shift 2 ;;
    --working-dir) WORKING_DIR="$2"; shift 2 ;;
    --conda-env) CONDA_ENV="$2"; shift 2 ;;
    --mask-json) MASK_JSON="$2"; shift 2 ;;
    --scene-threshold) SCENE_THRESHOLD="$2"; shift 2 ;;
    --min-duration) MIN_SCENE_DURATION="$2"; shift 2 ;;
    --max-duration) MAX_SCENE_DURATION="$2"; shift 2 ;;
    --scale) SCALE="$2"; shift 2 ;;
    --output-scale) OUTPUT_SCALE="$2"; shift 2 ;;
    --max-frames) MAX_FRAMES="$2"; shift 2 ;;
    --output-fps) OUTPUT_FPS="$2"; shift 2 ;;
    --no-convert) CONVERT_TO_MP4="false"; shift ;;
    --max-workers) MAX_WORKERS="$2"; shift 2 ;;
    --no-parallel) ENABLE_PARALLEL="false"; shift ;;
    --batch-size) BATCH_SIZE="$2"; shift 2 ;;
    --keep-intermediate) KEEP_INTERMEDIATE="true"; shift ;;
    *) echo "Unknown arg $1"; exit 1 ;;
  esac
done

LOG_FILE="$BASE_OUTPUT/process_${TIMESTAMP}.log"
log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG_FILE"; }
report_progress() { echo "PROGRESS:$1:$2"; }

setup_environment() {
    log "激活环境: $CONDA_ENV"
    source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || source /opt/conda/etc/profile.d/conda.sh 2>/dev/null || source ~/anaconda3/etc/profile.d/conda.sh 2>/dev/null
    conda activate "$CONDA_ENV" || return 1
}

validate_parameters() {
    if [ ! -f "$SOURCE_VIDEO" ]; then
        log "错误: 源视频文件不存在: $SOURCE_VIDEO"
        return 1
    fi
    if [ -z "$MASK_JSON" ]; then
        log "错误: Mask JSON 文件未指定。"
        return 1
    fi
    return 0
}

get_accurate_fps() {
    FPS_RATIO=$(ffprobe -v error -select_streams v:0 -show_entries stream=avg_frame_rate -of default=noprint_wrappers=1:nokey=1 "$SOURCE_VIDEO" 2>/dev/null)
    if [ -z "$FPS_RATIO" ]; then
        echo "25"
        return
    fi
    FPS=$(echo "$FPS_RATIO" | awk -F/ '{if ($2 == 0) print 25; else printf "%.0f\n", $1/$2}')
    FPS=${OUTPUT_FPS:-$FPS}
    echo "$FPS"
}

main() {
    mkdir -p "$BASE_OUTPUT"
    cd "$WORKING_DIR" || exit 1

    echo "=====================================" > "$LOG_FILE"
    echo "ProPainter 视频处理日志 (音画分离模式)" >> "$LOG_FILE"
    echo "视频: $VIDEO_NAME" >> "$LOG_FILE"
    echo "Mask: $MASK_JSON" >> "$LOG_FILE"
    echo "=====================================" >> "$LOG_FILE"

    setup_environment || exit 1

    log "开始处理: $VIDEO_NAME"

    # 【步骤 0】分离音频
    AUDIO_FILE="$BASE_OUTPUT/original_audio.m4a"
    HAS_AUDIO="false"

    log "步骤 0: 正在提取原始音频流..."
    # 【修复点】使用 -c:a aac 强制转码为 AAC 格式，解决 MP3 到 MP4 容器的不兼容问题
    if ffmpeg -i "$SOURCE_VIDEO" -vn -c:a aac -y "$AUDIO_FILE" 2>>"$LOG_FILE"; then
        if [ -s "$AUDIO_FILE" ]; then
            HAS_AUDIO="true"
            log "✓ 音频提取成功: $AUDIO_FILE"
        else
            log "⚠ 警告: 音频文件为空，源视频可能没有音频轨道。"
        fi
    else
        log "⚠ 警告: 音频提取失败，最终视频将无声。"
    fi

    report_progress "scene_detection" "10"
    SCENE_DIR="$BASE_OUTPUT/scenes"
    if [ -d "$SCENE_DIR" ]; then rm -rf "$SCENE_DIR"; fi

    log "执行场景检测..."
    python scene_detector.py --input "$SOURCE_VIDEO" --output "$SCENE_DIR" --threshold "$SCENE_THRESHOLD" --min-duration "$MIN_SCENE_DURATION" --mask-json "$MASK_JSON" 2>&1 | tee -a "$LOG_FILE"

    SCENE_INFO="$SCENE_DIR/segment_info.json"
    if [ ! -f "$SCENE_INFO" ]; then log "场景检测/分割失败"; exit 1; fi

    SEGMENT_COUNT=$(python -c "import json; print(json.load(open('$SCENE_INFO'))['total_segments'])")
    log "场景数量: $SEGMENT_COUNT"

    FPS=$(get_accurate_fps)
    log "使用的视频帧率(整数): $FPS FPS"

    FILELIST="$BASE_OUTPUT/filelist.txt"
    rm -f "$FILELIST" && touch "$FILELIST"
    SUCCESS_COUNT=0

    report_progress "processing" "40"

    for i in $(seq 0 $((SEGMENT_COUNT - 1))); do
        SEGMENT_PATH=$(python -c "import json; print(json.load(open('$SCENE_INFO'))['segments'][$i]['path'])")
        SCENE_OUT="$BASE_OUTPUT/scene_${i}"

        log "--- 处理场景 $i/$((SEGMENT_COUNT - 1)) ---"

        # 1. 提取 (清理旧文件)
        if [ -d "$SCENE_OUT/inputs/imgs" ]; then rm -rf "$SCENE_OUT/inputs/imgs"; fi
        mkdir -p "$SCENE_OUT/inputs/imgs"

        ffmpeg -i "$SEGMENT_PATH" -r "$FPS" -qscale:v 1 "$SCENE_OUT/inputs/imgs/frame_%08d.png" >/dev/null 2>&1

        # 2. 生成Mask (清理旧文件)
        if [ -d "$SCENE_OUT/inputs/masks" ]; then rm -rf "$SCENE_OUT/inputs/masks"; fi
        mkdir -p "$SCENE_OUT/inputs/masks"
        python make_mask.py --input "$SCENE_OUT/inputs/imgs" --output "$SCENE_OUT/inputs/masks" --mask_json "$MASK_JSON" 2>&1 | tee -a "$LOG_FILE"

        # 3. 缩放 (清理旧文件)
        if [ -d "$SCENE_OUT/inputs/imgs_resize" ]; then rm -rf "$SCENE_OUT/inputs/imgs_resize"; fi
        if [ -d "$SCENE_OUT/inputs/masks_resize" ]; then rm -rf "$SCENE_OUT/inputs/masks_resize"; fi
        mkdir -p "$SCENE_OUT/inputs/imgs_resize" "$SCENE_OUT/inputs/masks_resize"

        python resize_img.py --input_dir "$SCENE_OUT/inputs/imgs" --output_dir "$SCENE_OUT/inputs/imgs_resize" --scale "$SCALE" 2>&1 | tee -a "$LOG_FILE"
        python resize_img.py --input_dir "$SCENE_OUT/inputs/masks" --output_dir "$SCENE_OUT/inputs/masks_resize" --scale "$SCALE" 2>&1 | tee -a "$LOG_FILE"

        # 4. 推理 (清理旧文件)
        if [ -d "$SCENE_OUT/results" ]; then rm -rf "$SCENE_OUT/results"; fi
        mkdir -p "$SCENE_OUT/results"

        CMD="python inference_propainter.py --video \"$SCENE_OUT/inputs/imgs_resize\" --mask \"$SCENE_OUT/inputs/masks_resize\" --output \"$SCENE_OUT/results\" --save_fps \"$FPS\" --fp16"
        if [ -n "$MAX_FRAMES" ]; then CMD="$CMD --max_frames \"$MAX_FRAMES\""; fi

        log "    执行推理..."
        eval "$CMD" 2>&1 | tee -a "$LOG_FILE"

        # 5. 后处理 (【核心】纯画面重组)
        INFER_OUT="$SCENE_OUT/results/inference_output.mp4"
        FINAL_SCENE="$SCENE_OUT/final_scene.mp4"

        if [ -f "$INFER_OUT" ]; then
             log "    后处理: 生成无声片段..."

             # 使用 -r 指定输出帧率
             ffmpeg -i "$INFER_OUT" -vf "scale=960:544,setpts=N/($FPS*TB)" -r "$FPS" -c:v mpeg4 -qscale:v 2 -an -y "$FINAL_SCENE" 2>>"$LOG_FILE"

             if [ $? -eq 0 ] && [ -f "$FINAL_SCENE" ]; then
                 echo "file '$FINAL_SCENE'" >> "$FILELIST"
                 SUCCESS_COUNT=$((SUCCESS_COUNT+1))
                 log "    ✓ 场景 $i 处理完成"
             else
                 log "    错误: 场景 $i 最终视频生成失败"
             fi
        else
             log "    错误: 推理脚本未生成输出文件"
        fi

        if [ "$KEEP_INTERMEDIATE" = "false" ]; then
            rm -rf "$SCENE_OUT/inputs" "$SCENE_OUT/results"
        fi

        PROGRESS=$((40 + (i * 50 / SEGMENT_COUNT)))
        report_progress "processing" "$PROGRESS"
    done

    report_progress "finalizing" "90"
    if [ $SUCCESS_COUNT -gt 0 ]; then
        log "正在合并 $SUCCESS_COUNT 个纯画面片段..."

        SILENT_MERGED_OUTPUT="$BASE_OUTPUT/final_silent.mp4"
        ffmpeg -f concat -safe 0 -i "$FILELIST" -c copy -an -y "$SILENT_MERGED_OUTPUT" 2>> "$LOG_FILE"

        if [ $? -eq 0 ]; then
            log "✓ 纯画面合并完成: $SILENT_MERGED_OUTPUT"

            if [ "$HAS_AUDIO" = "true" ]; then
                log "正在执行最终音画合成..."
                ffmpeg -i "$SILENT_MERGED_OUTPUT" -i "$AUDIO_FILE" -c copy -map 0:v:0 -map 1:a:0 -shortest -y "$FINAL_OUTPUT" 2>> "$LOG_FILE"

                if [ $? -eq 0 ]; then
                    log "✓ ★★★ 最终视频已生成(含音频): $FINAL_OUTPUT"
                else
                    log "⚠ 音画合成失败，回退到无声视频..."
                    cp "$SILENT_MERGED_OUTPUT" "$FINAL_OUTPUT"
                fi
            else
                log "无音频文件，直接输出无声视频..."
                mv "$SILENT_MERGED_OUTPUT" "$FINAL_OUTPUT"
            fi

        else
             log "合并失败，尝试重新编码合并..."
             ffmpeg -f concat -safe 0 -i "$FILELIST" -c:v mpeg4 -qscale:v 2 -an -y "$FINAL_OUTPUT" 2>> "$LOG_FILE"
        fi
    else
        log "错误: 没有成功的片段可合并"
        exit 1
    fi

    report_progress "completed" "100"
}

if ! validate_parameters; then exit 1; fi
main