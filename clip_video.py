import argparse

from moviepy.editor import VideoFileClip


def clip_video(input_video_path, output_video_path, start_time, end_time):
    """
    从输入视频中按照指定的起始时间和结束时间截取视频片段，并保存为新的视频文件。

    参数:
    input_video_path (str): 输入视频文件的路径。
    output_video_path (str): 输出视频文件的路径，即切片后保存的文件路径。
    start_time (float): 切片的起始时间（单位：秒）。
    end_time (float): 切片的结束时间（单位：秒）。
    """
    # 加载视频文件
    video = VideoFileClip(input_video_path)
    # 截取指定时间段的视频片段
    clip = video.subclip(start_time, end_time)
    # 将截取的视频片段写入新的文件
    clip.write_videofile(output_video_path)
    # 关闭视频相关资源
    video.close()
    clip.close()

if __name__ == "__main__":
    # 【add by zhangmh at 2025.10.14：修改参数传递方式 start】

        # # 新增：解析命令行参数
        # parser = argparse.ArgumentParser(description='截取视频片段')
        # parser.add_argument('--input_path', required=True, help='输入视频路径')
        # parser.add_argument('--output_path', required=True, help='输出视频路径')
        # parser.add_argument('--start', type=float, required=True, help='起始时间（秒）')
        # parser.add_argument('--end', type=float, required=True, help='结束时间（秒）')
        # args = parser.parse_args()
        #
        # # 使用命令行参数调用函数，替代硬编码
        # clip_video(args.input_path, args.output_path, args.start, args.end)
    # 【add by zhangmh at 2025.10.14：修改参数传递方式 end】
    #
    # 使用示例 zhangmh修改传递参数前原本代码
    input_path = "/home/FeiLong_Grp/ZhangMengHui/Model/Model/Model/ProPainter/videos/test_0/test_clip.mp4"  # 替换为实际的输入视频路径
    output_path = "/home/FeiLong_Grp/ZhangMengHui/Model/Model/Model/ProPainter/videos/test_0/test_clip3.mp4"  # 替换为实际的输出视频路径
    start = 177  # 这里表示从第10秒开始切片，可按需修改
    end = 240  # 这里表示到第20秒结束切片，可按需修改
    clip_video(input_path, output_path, start, end)
