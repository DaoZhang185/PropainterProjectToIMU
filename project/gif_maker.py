#!/usr/bin/env python3
"""
解决花屏问题的简单有效版本
5到4平滑过渡 + 432帧倒序
"""

import os
import subprocess


def main():
    if not os.path.exists("input.mp4"):
        print("找不到input.mp4")
        return

    print("生成无花屏的平滑GIF...")

    # 方案1：分步处理，避免复杂filter_complex
    # 步骤1：提取前5秒作为12345
    print("1. 提取12345部分（前5秒）...")
    subprocess.run(['ffmpeg', '-i', 'input.mp4', '-t', '5', '-vf', 'fps=24,scale=1920:1080', '-y', 'temp_12345.mp4'],
                   capture_output=True)

    # 步骤2：提取5-8秒作为432（倒序）
    print("2. 提取432部分（5-8秒，准备倒序）...")
    subprocess.run(
        ['ffmpeg', '-i', 'input.mp4', '-ss', '5', '-t', '3', '-vf', 'fps=24,scale=1920:1080', '-y', 'temp_432_raw.mp4'],
        capture_output=True)

    # 步骤3：将432倒序
    print("3. 将432部分倒序...")
    subprocess.run(['ffmpeg', '-i', 'temp_432_raw.mp4', '-vf', 'reverse', '-y', 'temp_432_rev.mp4'],
                   capture_output=True)

    # 步骤4：处理5到4的过渡
    print("4. 处理5到4的平滑过渡...")

    # 创建过渡文件：5的最后0.5秒 + 4的开始0.5秒 混合
    # 先提取5的最后0.5秒
    subprocess.run(['ffmpeg', '-i', 'temp_12345.mp4', '-ss', '4.5', '-t', '0.5', '-y', 'temp_5_end.mp4'],
                   capture_output=True)

    # 提取4的开始0.5秒（来自倒序后的432）
    subprocess.run(['ffmpeg', '-i', 'temp_432_rev.mp4', '-t', '0.5', '-y', 'temp_4_start.mp4'],
                   capture_output=True)

    # 创建过渡：混合最后0.5秒
    filter_transition = (
        "[0:v][1:v]"
        "blend=all_expr="
        "'A*(0.5-0.5*cos(PI*T/0.5))+B*(0.5+0.5*cos(PI*T/0.5))'"
    )

    subprocess.run([
        'ffmpeg',
        '-i', 'temp_5_end.mp4',
        '-i', 'temp_4_start.mp4',
        '-filter_complex', filter_transition,
        '-t', '0.5',
        '-y', 'temp_transition.mp4'
    ], capture_output=True)

    # 步骤5：拼接所有部分
    print("5. 拼接所有部分...")

    # 创建文件列表
    with open('concat_list.txt', 'w') as f:
        f.write("file 'temp_12345_trimmed.mp4'\n")
        f.write("file 'temp_transition.mp4'\n")
        f.write("file 'temp_432_rest.mp4'\n")

    # 截取12345的前4.5秒（去掉最后0.5秒用于过渡）
    subprocess.run(['ffmpeg', '-i', 'temp_12345.mp4', '-t', '4.5', '-c', 'copy', '-y', 'temp_12345_trimmed.mp4'],
                   capture_output=True)

    # 截取432的剩余部分（去掉前0.5秒用于过渡）
    subprocess.run(['ffmpeg', '-i', 'temp_432_rev.mp4', '-ss', '0.5', '-c', 'copy', '-y', 'temp_432_rest.mp4'],
                   capture_output=True)

    # 拼接：前4.5秒 + 0.5秒过渡 + 2.5秒倒序
    subprocess.run([
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'concat_list.txt',
        '-c', 'copy',
        '-y', 'temp_one_cycle.mp4'
    ], capture_output=True)

    # 步骤6：创建循环（再复制一次）
    print("6. 创建完整循环...")
    with open('concat_final.txt', 'w') as f:
        f.write("file 'temp_one_cycle.mp4'\n")
        f.write("file 'temp_one_cycle.mp4'\n")

    subprocess.run([
        'ffmpeg',
        '-f', 'concat',
        '-safe', '0',
        '-i', 'concat_final.txt',
        '-c', 'copy',
        '-y', 'temp_final_loop.mp4'
    ], capture_output=True)

    # 步骤7：转GIF（优化大小）
    print("7. 转换为GIF...")
    subprocess.run([
        'ffmpeg', '-i', 'temp_final_loop.mp4',
        '-vf', 'fps=20,scale=1280:720,split[p][b];[p]palettegen=96[p];[b][p]paletteuse',
        '-loop', '0',
        '-y', 'final_car.gif'
    ], capture_output=True)

    # 清理临时文件
    temp_files = [
        'temp_12345.mp4', 'temp_432_raw.mp4', 'temp_432_rev.mp4',
        'temp_5_end.mp4', 'temp_4_start.mp4', 'temp_transition.mp4',
        'temp_12345_trimmed.mp4', 'temp_432_rest.mp4', 'temp_one_cycle.mp4',
        'temp_final_loop.mp4', 'concat_list.txt', 'concat_final.txt'
    ]

    for temp_file in temp_files:
        if os.path.exists(temp_file):
            os.remove(temp_file)

    if os.path.exists('final_car.gif'):
        size = os.path.getsize('final_car.gif') / (1024 * 1024)
        print(f"\n✅ 生成成功！文件大小: {size:.1f} MB")
        print("🎯 循环模式: 12345432 12345432")
        print("🔄 5到4: 0.5秒余弦函数平滑过渡")
        print("📱 分辨率: 1280x720 (优化大小)")
    else:
        print("\n❌ 生成失败，尝试简化版...")
        create_simple_version()


def create_simple_version():
    """如果分步失败，用这个最简单的版本"""
    print("\n使用最简版本...")

    # 最简单的命令，避免所有复杂处理
    cmd = [
        'ffmpeg',
        '-i', 'input.mp4',
        '-vf',
        # 截取前8秒
        'trim=end=8,'
        # 分割为12345和432
        'split=2[v1][v2];'
        # 12345（前5秒）
        '[v1]trim=end=5,setpts=PTS-STARTPTS[forward];'
        # 432（后3秒倒序）
        '[v2]trim=start=5,reverse,setpts=PTS-STARTPTS[reverse];'
        # 连接
        '[forward][reverse]concat,'
        # 循环一次
        'loop=loop=1:size=1000,'
        # 转GIF
        'fps=20,scale=1280:720,split[p][b];[p]palettegen[p];[b][p]paletteuse',
        '-loop', '0',
        '-y', 'simple_car.gif'
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode == 0:
        print("✅ 简版生成: simple_car.gif")


if __name__ == "__main__":
    main()