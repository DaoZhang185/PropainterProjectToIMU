import cv2
import numpy as np
import torch
import base64
from segment_anything import sam_model_registry, SamPredictor


class SAMService:
    def __init__(self, checkpoint_path, model_type="vit_b", device="cuda"):
        print("[SAM] 正在加载 Segment Anything 模型...")
        self.sam = sam_model_registry[model_type](checkpoint=checkpoint_path)
        self.sam.to(device=device)
        self.predictor = SamPredictor(self.sam)
        print("[SAM] 模型加载完成！")

    def predict_from_b64(self, b64_img, box=None, point_coords=None, point_labels=None, extract_mode="solid"):
        """
        extract_mode:
          - "solid": 实心整体覆盖 (默认SAM)
          - "hollow": 仅提取空心边框 (形态学梯度)
          - "text": 仅提纯单色文字 (剔除底色与外框)
        """
        img_data = base64.b64decode(b64_img.split(',')[1] if ',' in b64_img else b64_img)
        nparr = np.frombuffer(img_data, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(img_rgb)

        input_box = np.array(box) if box else None
        input_point = np.array(point_coords) if point_coords else None
        input_label = np.array(point_labels) if point_labels else None

        # 1. 使用 SAM 拿到大基准实心掩码
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False
        )
        sam_mask_uint8 = (masks[0] * 255).astype(np.uint8)

        # 2. 根据前端指令，执行绝对独立的过滤逻辑
        if extract_mode == "hollow":
            # 【模式 A】: 仅提取空心外边框
            kernel = np.ones((5, 5), np.uint8)
            final_mask = cv2.morphologyEx(sam_mask_uint8, cv2.MORPH_GRADIENT, kernel)

        elif extract_mode == "text" and point_coords and point_labels:
            # 【模式 B】: 仅提取点击颜色的文字
            pos_idx = point_labels.index(1) if 1 in point_labels else -1
            if pos_idx != -1:
                px, py = point_coords[pos_idx]
                px, py = min(px, img_rgb.shape[1] - 1), min(py, img_rgb.shape[0] - 1)
                target_color = img_rgb[py, px].astype(np.int32)

                color_mask = np.zeros_like(sam_mask_uint8)
                # 色差严苛扫描 (容差收紧到 45)
                diff = np.sum(np.abs(img_rgb.astype(np.int32) - target_color), axis=-1)
                color_mask[diff < 45] = 255

                # 与 SAM 掩码做交集，绝不跑到框外面去，且完全不要框本身
                final_mask = cv2.bitwise_and(color_mask, sam_mask_uint8)

                # 轻微闭运算，连结断裂的文字像素
                final_mask = cv2.morphologyEx(final_mask, cv2.MORPH_CLOSE, np.ones((2, 2), np.uint8))
            else:
                final_mask = sam_mask_uint8
        else:
            # 【模式 C】: 实心整体覆盖
            final_mask = sam_mask_uint8

        # 3. 生成多边形阵列
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 5:
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) >= 3:
                    polygons.append(approx.squeeze().tolist())

        return polygons