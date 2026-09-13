# -*- coding: utf-8 -*-
import sys
import os

# =================================================================
# 【路径修复】确保脚本能引用上级目录（根目录）的模块 (core, model, utils)
# =================================================================
current_dir = os.path.dirname(os.path.abspath(__file__))  # .../ProPainter/project
root_dir = os.path.dirname(current_dir)  # .../ProPainter
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
# =================================================================

import traceback
import cv2
import argparse
import imageio
import numpy as np
import scipy.ndimage
from PIL import Image
from tqdm import tqdm

import torch
import torchvision

# 现在可以正常从根目录导入模块了
from model.modules.flow_comp_raft import RAFT_bi
from model.recurrent_flow_completion import RecurrentFlowCompleteNet
from model.propainter import InpaintGenerator
from utils.download_util import load_file_from_url
from core.utils import to_tensors
from model.misc import get_device

import warnings

warnings.filterwarnings("ignore")

pretrain_model_url = 'https://github.com/sczhou/ProPainter/releases/download/v0.1.0/'


def imwrite(img, file_path, params=None, auto_mkdir=True):
    if auto_mkdir:
        dir_name = os.path.abspath(os.path.dirname(file_path))
        os.makedirs(dir_name, exist_ok=True)
    return cv2.imwrite(file_path, img, params)


# resize frames
def resize_frames(frames, size=None):
    if size is not None:
        out_size = size
        process_size = (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)
        frames = [f.resize(process_size) for f in frames]
    else:
        out_size = frames[0].size
        process_size = (out_size[0] - out_size[0] % 8, out_size[1] - out_size[1] % 8)
        if not out_size == process_size:
            frames = [f.resize(process_size) for f in frames]

    return frames, process_size, out_size


#  read frames from video
def read_frame_from_videos(frame_root, max_frames=None):
    """读取视频帧，支持多种视频格式"""
    video_extensions = {'.mp4', '.mov', '.avi', '.mkv', '.wmv', '.flv', '.webm',
                        '.MP4', '.MOV', '.AVI', '.MKV', '.WMV', '.FLV', '.WEBM'}

    if any(frame_root.lower().endswith(ext) for ext in video_extensions):
        video_name = os.path.basename(frame_root)
        video_name = os.path.splitext(video_name)[0]

        try:
            vframes, aframes, info = torchvision.io.read_video(filename=frame_root, pts_unit='sec')
            frames = list(vframes.numpy())
            frames = [Image.fromarray(f) for f in frames]
            fps = info['video_fps']
        except Exception as e:
            print(f"torchvision读取失败，尝试使用moviepy: {e}")
            try:
                from moviepy.editor import VideoFileClip
                clip = VideoFileClip(frame_root)
                fps = clip.fps
                frames = []
                for frame in clip.iter_frames():
                    frames.append(Image.fromarray(frame))
                clip.close()
            except Exception as e2:
                print(f"moviepy读取也失败: {e2}")
                try:
                    cap = cv2.VideoCapture(frame_root)
                    frames = []
                    fps = cap.get(cv2.CAP_PROP_FPS)
                    while True:
                        ret, frame = cap.read()
                        if not ret: break
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        frames.append(Image.fromarray(frame_rgb))
                        if max_frames and len(frames) >= max_frames: break
                    cap.release()
                except Exception as e3:
                    print(f"所有视频读取方法都失败: {e3}")
                    raise
    else:
        video_name = os.path.basename(frame_root)
        frames = []
        fr_lst = sorted(os.listdir(frame_root))

        if max_frames is not None and max_frames > 0:
            fr_lst = fr_lst[:max_frames]
            print(f"处理前 {max_frames} 帧")
        else:
            print(f"处理所有 {len(fr_lst)} 帧")

        for fr in fr_lst:
            if not fr.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')): continue
            frame = cv2.imread(os.path.join(frame_root, fr))
            if frame is None: continue
            frame = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frames.append(frame)
        fps = None

    size = frames[0].size if frames else (0, 0)
    return frames, fps, size, video_name


def binary_mask(mask, th=0.1):
    mask[mask > th] = 1
    mask[mask <= th] = 0
    return mask


# read frame-wise masks
def read_mask(mpath, length, size, max_frames=None, flow_mask_dilates=8, mask_dilates=5):
    masks_img = []
    masks_dilated = []
    flow_masks = []

    print(f"正在读取掩码从: {mpath}")
    try:
        if mpath.endswith(('jpg', 'jpeg', 'png', 'JPG', 'JPEG', 'PNG')):
            mask_img = Image.open(mpath)
            masks_img.append(mask_img)
        else:
            mnames = sorted([f for f in os.listdir(mpath) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
            print(f"找到 {len(mnames)} 个掩码文件")
            if max_frames is not None and max_frames > 0:
                mnames = mnames[:max_frames]
                print(f"限制为前 {max_frames} 个掩码")
            for mp in mnames:
                mask_path = os.path.join(mpath, mp)
                try:
                    mask_img = Image.open(mask_path).convert('L')
                    masks_img.append(mask_img)
                except Exception as e:
                    print(f"读取掩码失败 {mp}: {e}")
                    continue
        print(f"最终成功读取 {len(masks_img)} 个掩码")
    except Exception as e:
        print(f"读取掩码目录失败: {e}")
        return [], []

    if len(masks_img) == 0:
        print("错误：没有成功读取任何掩码文件！")
        return [], []

    for mask_img in masks_img:
        try:
            if size is not None:
                mask_img = mask_img.resize(size, Image.NEAREST)
            mask_img = np.array(mask_img.convert('L'))

            if mask_dilates > 0:
                extra_dilated = scipy.ndimage.binary_dilation(mask_img, iterations=1).astype(np.uint8)
                mask_img = scipy.ndimage.binary_dilation(extra_dilated, iterations=mask_dilates).astype(np.uint8)
            else:
                mask_img = binary_mask(mask_img).astype(np.uint8)
            masks_dilated.append(Image.fromarray(mask_img * 255))

            if flow_mask_dilates > 0:
                flow_mask_img = scipy.ndimage.binary_dilation(mask_img, iterations=flow_mask_dilates + 2).astype(
                    np.uint8)
            else:
                flow_mask_img = binary_mask(mask_img).astype(np.uint8)
            flow_masks.append(Image.fromarray(flow_mask_img * 255))
        except Exception as e:
            print(f"处理掩码时出错: {e}")
            continue

    print(f"处理完成: flow_masks={len(flow_masks)}, masks_dilated={len(masks_dilated)}")

    if len(masks_img) == 1:
        flow_masks = flow_masks * length
        masks_dilated = masks_dilated * length

    return flow_masks, masks_dilated


def extrapolation(video_ori, scale):
    """Prepares the data for video outpainting."""
    nFrame = len(video_ori)
    imgW, imgH = video_ori[0].size

    imgH_extr = int(scale[0] * imgH)
    imgW_extr = int(scale[1] * imgW)
    imgH_extr = imgH_extr - imgH_extr % 8
    imgW_extr = imgW_extr - imgW_extr % 8
    H_start = int((imgH_extr - imgH) / 2)
    W_start = int((imgW_extr - imgW) / 2)

    frames = []
    for v in video_ori:
        frame = np.zeros(((imgH_extr, imgW_extr, 3)), dtype=np.uint8)
        frame[H_start: H_start + imgH, W_start: W_start + imgW, :] = v
        frames.append(Image.fromarray(frame))

    masks_dilated = []
    flow_masks = []
    dilate_h = 4 if H_start > 10 else 0
    dilate_w = 4 if W_start > 10 else 0
    mask = np.ones(((imgH_extr, imgW_extr)), dtype=np.uint8)
    mask[H_start + dilate_h: H_start + imgH - dilate_h, W_start + dilate_w: W_start + imgW - dilate_w] = 0
    flow_masks.append(Image.fromarray(mask * 255))
    mask[H_start: H_start + imgH, W_start: W_start + imgW] = 0
    masks_dilated.append(Image.fromarray(mask * 255))

    flow_masks = flow_masks * nFrame
    masks_dilated = masks_dilated * nFrame
    return frames, flow_masks, masks_dilated, (imgW_extr, imgH_extr)


def get_ref_index(mid_neighbor_id, neighbor_ids, length, ref_stride=5, ref_num=-1):
    ref_index = []
    if ref_num == -1:
        for i in range(0, length, ref_stride):
            if i not in neighbor_ids: ref_index.append(i)
    else:
        start_idx = max(0, mid_neighbor_id - ref_stride * (ref_num // 2))
        end_idx = min(length, mid_neighbor_id + ref_stride * (ref_num // 2))
        for i in range(start_idx, end_idx, ref_stride):
            if i not in neighbor_ids:
                if len(ref_index) > ref_num: break
                ref_index.append(i)
    return ref_index


if __name__ == '__main__':
    device = get_device()
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--video', type=str, default='inputs/object_removal/bmx-trees',
                        help='Path of the input video')
    parser.add_argument('-m', '--mask', type=str, default='inputs/object_removal/bmx-trees_mask',
                        help='Path of the mask(s)')
    parser.add_argument('-o', '--output', type=str, default='results', help='Output folder')
    parser.add_argument("--resize_ratio", type=float, default=1.0, help='Resize scale')
    parser.add_argument('--height', type=int, default=-1, help='Height')
    parser.add_argument('--width', type=int, default=-1, help='Width')
    parser.add_argument('--mask_dilation', type=int, default=4, help='Mask dilation')
    parser.add_argument('--flow_mask_dilation', type=int, default=10, help='Flow mask dilation')
    parser.add_argument("--ref_stride", type=int, default=10, help='Stride of global reference frames')
    parser.add_argument("--neighbor_length", type=int, default=10, help='Length of local neighboring frames')
    parser.add_argument("--subvideo_length", type=int, default=80, help='Length of sub-video')
    parser.add_argument("--raft_iter", type=int, default=25, help='Iterations for RAFT')
    parser.add_argument('--mode', default='video_inpainting', choices=['video_inpainting', 'video_outpainting'])
    parser.add_argument('--save_fps', type=int, default=12, help='Frame per second')
    parser.add_argument('--save_frames', action='store_true', help='Save output frames')
    parser.add_argument('--fp16', action='store_true', help='Use fp16')
    parser.add_argument('--max_frames', type=int, default=None, help='Max frames')
    parser.add_argument('--success_flag', type=str, default=None, help='Success flag file path')
    parser.add_argument('--scale_h', type=float, default=1.0)
    parser.add_argument('--scale_w', type=float, default=1.0)

    # 注意：移除了 --enhance_masks 参数，因为它包含不安全的硬编码逻辑

    args = parser.parse_args()

    use_half = True if args.fp16 else False
    if device == torch.device('cpu'): use_half = False

    frames, fps, size, video_name = read_frame_from_videos(args.video, args.max_frames)
    if not args.width == -1 and not args.height == -1: size = (args.width, args.height)
    if not args.resize_ratio == 1.0: size = (int(args.resize_ratio * size[0]), int(args.resize_ratio * size[1]))
    frames, size, out_size = resize_frames(frames, size)

    fps = args.save_fps if fps is None else fps
    save_root = args.output
    if not os.path.exists(save_root): os.makedirs(save_root, exist_ok=True)

    flow_mask_dilates = args.flow_mask_dilation if hasattr(args, 'flow_mask_dilation') else args.mask_dilation + 2
    mask_dilates = args.mask_dilation

    if args.mode == 'video_inpainting':
        frames_len = len(frames)
        # 移除 enhance_masks 逻辑，直接读取处理
        flow_masks, masks_dilated = read_mask(args.mask, frames_len, size, args.max_frames,
                                              flow_mask_dilates=flow_mask_dilates,
                                              mask_dilates=mask_dilates)

        min_len = min(len(frames), len(flow_masks), len(masks_dilated))
        if len(frames) != min_len:
            print(f"警告：帧数量({len(frames)})与有效掩码数量({len(flow_masks)})不匹配！已裁剪至 {min_len}。")
            frames = frames[:min_len]
        w, h = size
    elif args.mode == 'video_outpainting':
        assert args.scale_h is not None and args.scale_w is not None, 'Please provide a outpainting scale (s_h, s_w).'
        frames, flow_masks, masks_dilated, size = extrapolation(frames, (args.scale_h, args.scale_w))
        w, h = size
    else:
        raise NotImplementedError

    min_len = min(len(frames), len(flow_masks), len(masks_dilated))
    if len(frames) != min_len or len(flow_masks) != min_len or len(masks_dilated) != min_len:
        print(f"警告：序列长度不一致，统一裁剪至: {min_len}")
        frames = frames[:min_len]
        flow_masks = flow_masks[:min_len]
        masks_dilated = masks_dilated[:min_len]

    masked_frame_for_save = []
    for i in range(len(frames)):
        mask_ = np.expand_dims(np.array(masks_dilated[i]), 2).repeat(3, axis=2) / 255.
        img = np.array(frames[i])
        green = np.zeros([h, w, 3])
        green[:, :, 1] = 255
        alpha = 0.6
        fuse_img = (1 - alpha) * img + alpha * green
        fuse_img = mask_ * fuse_img + (1 - mask_) * img
        masked_frame_for_save.append(fuse_img.astype(np.uint8))

    frames_inp = [np.array(f).astype(np.uint8) for f in frames]
    if not frames_inp:
        print("\n致命错误：有效视频帧列表为空。")
        exit(1)

    frames = to_tensors()(frames).unsqueeze(0) * 2 - 1
    flow_masks = to_tensors()(flow_masks).unsqueeze(0)
    masks_dilated = to_tensors()(masks_dilated).unsqueeze(0)
    frames, flow_masks, masks_dilated = frames.to(device), flow_masks.to(device), masks_dilated.to(device)

    # Load weights
    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'raft-things.pth'),
                                   model_dir='../weights', progress=True, file_name=None)
    fix_raft = RAFT_bi(ckpt_path, device)

    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'recurrent_flow_completion.pth'),
                                   model_dir='../weights', progress=True, file_name=None)
    fix_flow_complete = RecurrentFlowCompleteNet(ckpt_path)
    for p in fix_flow_complete.parameters(): p.requires_grad = False
    fix_flow_complete.to(device)
    fix_flow_complete.eval()

    ckpt_path = load_file_from_url(url=os.path.join(pretrain_model_url, 'ProPainter.pth'),
                                   model_dir='../weights', progress=True, file_name=None)
    model = InpaintGenerator(model_path=ckpt_path).to(device)
    model.eval()

    video_length = frames.size(1)
    print(f'\nProcessing: {video_name} [{video_length} frames]...')

    with torch.no_grad():
        # RAFT
        if frames.size(-1) <= 640:
            short_clip_len = 12
        elif frames.size(-1) <= 720:
            short_clip_len = 8
        elif frames.size(-1) <= 1280:
            short_clip_len = 4
        else:
            short_clip_len = 2

        if frames.size(1) > short_clip_len:
            gt_flows_f_list, gt_flows_b_list = [], []
            for f in range(0, video_length, short_clip_len):
                end_f = min(video_length, f + short_clip_len)
                if f == 0:
                    flows_f, flows_b = fix_raft(frames[:, f:end_f], iters=args.raft_iter)
                else:
                    flows_f, flows_b = fix_raft(frames[:, f - 1:end_f], iters=args.raft_iter)
                gt_flows_f_list.append(flows_f)
                gt_flows_b_list.append(flows_b)
                torch.cuda.empty_cache()
            gt_flows_f = torch.cat(gt_flows_f_list, dim=1)
            gt_flows_b = torch.cat(gt_flows_b_list, dim=1)
            gt_flows_bi = (gt_flows_f, gt_flows_b)
        else:
            gt_flows_bi = fix_raft(frames, iters=args.raft_iter)
            torch.cuda.empty_cache()

        if use_half:
            frames, flow_masks, masks_dilated = frames.half(), flow_masks.half(), masks_dilated.half()
            gt_flows_bi = (gt_flows_bi[0].half(), gt_flows_bi[1].half())
            fix_flow_complete = fix_flow_complete.half()
            model = model.half()

        # Flow Completion
        flow_length = gt_flows_bi[0].size(1)
        if flow_length > args.subvideo_length:
            pred_flows_f, pred_flows_b = [], []
            pad_len = 5
            for f in range(0, flow_length, args.subvideo_length):
                s_f = max(0, f - pad_len)
                e_f = min(flow_length, f + args.subvideo_length + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(flow_length, f + args.subvideo_length)
                pred_flows_bi_sub, _ = fix_flow_complete.forward_bidirect_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), flow_masks[:, s_f:e_f + 1])
                pred_flows_bi_sub = fix_flow_complete.combine_flow(
                    (gt_flows_bi[0][:, s_f:e_f], gt_flows_bi[1][:, s_f:e_f]), pred_flows_bi_sub,
                    flow_masks[:, s_f:e_f + 1])
                pred_flows_f.append(pred_flows_bi_sub[0][:, pad_len_s:e_f - s_f - pad_len_e])
                pred_flows_b.append(pred_flows_bi_sub[1][:, pad_len_s:e_f - s_f - pad_len_e])
                torch.cuda.empty_cache()
            pred_flows_f = torch.cat(pred_flows_f, dim=1)
            pred_flows_b = torch.cat(pred_flows_b, dim=1)
            pred_flows_bi = (pred_flows_f, pred_flows_b)
        else:
            pred_flows_bi, _ = fix_flow_complete.forward_bidirect_flow(gt_flows_bi, flow_masks)
            pred_flows_bi = fix_flow_complete.combine_flow(gt_flows_bi, pred_flows_bi, flow_masks)
            torch.cuda.empty_cache()

        # Image Propagation
        masked_frames = frames * (1 - masks_dilated)
        subvideo_length_img_prop = min(100, args.subvideo_length)
        if video_length > subvideo_length_img_prop:
            updated_frames, updated_masks = [], []
            pad_len = 10
            for f in range(0, video_length, subvideo_length_img_prop):
                s_f = max(0, f - pad_len)
                e_f = min(video_length, f + subvideo_length_img_prop + pad_len)
                pad_len_s = max(0, f) - s_f
                pad_len_e = e_f - min(video_length, f + subvideo_length_img_prop)
                b, t, _, _, _ = masks_dilated[:, s_f:e_f].size()
                pred_flows_bi_sub = (pred_flows_bi[0][:, s_f:e_f - 1], pred_flows_bi[1][:, s_f:e_f - 1])
                prop_imgs_sub, updated_local_masks_sub = model.img_propagation(masked_frames[:, s_f:e_f],
                                                                               pred_flows_bi_sub,
                                                                               masks_dilated[:, s_f:e_f], 'nearest')
                updated_frames_sub = frames[:, s_f:e_f] * (1 - masks_dilated[:, s_f:e_f]) + prop_imgs_sub.view(b, t, 3,
                                                                                                               h,
                                                                                                               w) * masks_dilated[
                                                                                                                    :,
                                                                                                                    s_f:e_f]
                updated_masks_sub = updated_local_masks_sub.view(b, t, 1, h, w)
                updated_frames.append(updated_frames_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                updated_masks.append(updated_masks_sub[:, pad_len_s:e_f - s_f - pad_len_e])
                torch.cuda.empty_cache()
            updated_frames = torch.cat(updated_frames, dim=1)
            updated_masks = torch.cat(updated_masks, dim=1)
        else:
            b, t, _, _, _ = masks_dilated.size()
            prop_imgs, updated_local_masks = model.img_propagation(masked_frames, pred_flows_bi, masks_dilated,
                                                                   'nearest')
            updated_frames = frames * (1 - masks_dilated) + prop_imgs.view(b, t, 3, h, w) * masks_dilated
            updated_masks = updated_local_masks.view(b, t, 1, h, w)
            torch.cuda.empty_cache()

    ori_frames = frames_inp
    comp_frames = [None] * video_length
    neighbor_stride = args.neighbor_length // 2
    ref_num = args.subvideo_length // args.ref_stride if video_length > args.subvideo_length else -1

    # Feature Propagation + Transformer
    for f in tqdm(range(0, video_length, neighbor_stride)):
        neighbor_ids = [i for i in range(max(0, f - neighbor_stride), min(video_length, f + neighbor_stride + 1))]
        ref_ids = get_ref_index(f, neighbor_ids, video_length, args.ref_stride, ref_num)
        selected_imgs = updated_frames[:, neighbor_ids + ref_ids, :, :, :]
        selected_masks = masks_dilated[:, neighbor_ids + ref_ids, :, :, :]
        selected_update_masks = updated_masks[:, neighbor_ids + ref_ids, :, :, :]
        selected_pred_flows_bi = (
        pred_flows_bi[0][:, neighbor_ids[:-1], :, :, :], pred_flows_bi[1][:, neighbor_ids[:-1], :, :, :])

        with torch.no_grad():
            l_t = len(neighbor_ids)
            pred_img = model(selected_imgs, selected_pred_flows_bi, selected_masks, selected_update_masks, l_t)
            pred_img = pred_img.view(-1, 3, h, w)
            pred_img = (pred_img + 1) / 2
            pred_img = pred_img.cpu().permute(0, 2, 3, 1).numpy() * 255
            binary_masks = masks_dilated[0, neighbor_ids, :, :, :].cpu().permute(0, 2, 3, 1).numpy().astype(np.uint8)
            for i in range(len(neighbor_ids)):
                idx = neighbor_ids[i]
                img = np.array(pred_img[i]).astype(np.uint8) * binary_masks[i] + ori_frames[idx] * (1 - binary_masks[i])
                if comp_frames[idx] is None:
                    comp_frames[idx] = img
                else:
                    comp_frames[idx] = comp_frames[idx].astype(np.float32) * 0.5 + img.astype(np.float32) * 0.5
                comp_frames[idx] = comp_frames[idx].astype(np.uint8)
        torch.cuda.empty_cache()

    if args.save_frames:
        for idx in range(video_length):
            f = comp_frames[idx]
            f = cv2.resize(f, out_size, interpolation=cv2.INTER_CUBIC)
            f = cv2.cvtColor(f, cv2.COLOR_BGR2RGB)
            img_save_root = os.path.join(save_root, 'frames', str(idx).zfill(4) + '.png')
            imwrite(f, img_save_root)

    print(f"\n准备保存修复后的视频...")
    if comp_frames and comp_frames[0] is not None:
        actual_height, actual_width = comp_frames[0].shape[:2]
        save_size = (actual_width, actual_height)
    else:
        save_size = out_size

    print(f"视频信息: {len(comp_frames)} 帧, 保存分辨率: {save_size}, 帧率: {fps} FPS")

    comp_frames_resized = []
    for f in comp_frames:
        if f is not None:
            if f.shape[1] != save_size[0] or f.shape[0] != save_size[1]:
                comp_frames_resized.append(cv2.resize(f, save_size))
            else:
                comp_frames_resized.append(f)
    comp_frames_resized = [f for f in comp_frames_resized if f is not None]

    if not comp_frames_resized:
        print("错误：没有有效的帧可以保存")
        exit(1)

    os.makedirs(save_root, exist_ok=True)

    print("生成绿色mask覆盖的预览视频...")
    try:
        masked_preview_resized = []
        for f in masked_frame_for_save:
            if f.shape[1] != save_size[0] or f.shape[0] != save_size[1]:
                masked_preview_resized.append(cv2.resize(f, save_size))
            else:
                masked_preview_resized.append(f)
        imageio.mimwrite(os.path.join(save_root, 'masked_in.mp4'), masked_preview_resized, fps=fps, quality=7)
        print(f"✓ 绿色mask预览视频保存成功")
    except Exception as e:
        print(f"✗ 绿色mask预览视频保存失败: {e}")

    output_video_path = os.path.join(save_root, 'inference_output.mp4')
    success = False
    try:
        imageio.mimwrite(output_video_path, comp_frames_resized, fps=fps, quality=7)
        print(f"✓ 修复视频保存成功: {output_video_path}")
        success = True
    except Exception as e:
        print(f"✗ imageio保存失败: {e}, 尝试OpenCV...")
        try:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_video_path, fourcc, fps, save_size)
            for frame in comp_frames_resized:
                out.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
            out.release()
            print(f"✓ OpenCV保存成功")
            success = True
        except Exception as e2:
            print(f"✗ 备选保存失败: {e2}")

    if success and args.success_flag:
        try:
            with open(args.success_flag, 'w') as f:
                f.write('SUCCESS')
            print(f"✓ 成功标志文件已创建")
        except:
            pass

    print("\n视频处理完成！")
    torch.cuda.empty_cache()