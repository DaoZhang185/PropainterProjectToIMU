#!/bin/bash

# ================= 配置区域 =================
# 1. 指定包含文件列表的文本文件路径
LIST_FILE="/home/FeiLong_Grp/ZhangMengHui/Model/Model/Model/ProPainter/results/finalresults11/filelist.txt"

# 2. 指定输出视频路径
OUTPUT_FILE="/home/FeiLong_Grp/ZhangMengHui/Model/Model/Model/ProPainter/results/finalresults11/final_output_simple.mp4"
# ===========================================

if [ -f "$LIST_FILE" ]; then
    echo "正在读取列表合并: $LIST_FILE"
    
    # -f concat: 拼接模式
    # -safe 0: 允许读取任意路径
    # -c copy: 不重编码，直接复制流（速度快，画质无损）
    ffmpeg -f concat -safe 0 -i "$LIST_FILE" -c copy -y "$OUTPUT_FILE"
    
    echo "完成: $OUTPUT_FILE"
else
    echo "错误: 找不到列表文件 $LIST_FILE"
fi