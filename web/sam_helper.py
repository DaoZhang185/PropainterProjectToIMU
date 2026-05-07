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

    def predict_from_b64(self, b64_img, box=None, point_coords=None, point_labels=None, extract_mode="semantic"):
        """
        extract_mode:
          - "semantic": 普通模式，框选完整的物体
          - "color": 镂空模式，自动剥离实心底色，仅保留空心外边框和文字像素
        """
        img_data = base64.b64decode(b64_img.split(',')[1] if ',' in b64_img else b64_img)
        nparr = np.frombuffer(img_data, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        self.predictor.set_image(img_rgb)

        input_box = np.array(box) if box else None
        input_point = np.array(point_coords) if point_coords else None
        input_label = np.array(point_labels) if point_labels else None

        # 1. 使用 SAM 拿到大基准实心框
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False
        )
        sam_mask = masks[0]
        sam_mask_uint8 = (sam_mask * 255).astype(np.uint8)

        if extract_mode == "color" and point_coords and point_labels:
            # === 好莱坞级非破坏性遮罩 (Non-destructive Masking) ===

            # 第一步：提取空心外边框 (利用形态学梯度)
            kernel = np.ones((5, 5), np.uint8)
            edge_mask = cv2.morphologyEx(sam_mask_uint8, cv2.MORPH_GRADIENT, kernel)

            # 第二步：获取纯色文字
            pos_idx = point_labels.index(1) if 1 in point_labels else -1
            color_mask = np.zeros_like(sam_mask_uint8)
            if pos_idx != -1:
                px, py = point_coords[pos_idx]
                px, py = min(px, img_rgb.shape[1] - 1), min(py, img_rgb.shape[0] - 1)
                target_color = img_rgb[py, px].astype(np.int32)

                # 色差欧氏距离扫描 (容差为 60)
                diff = np.sum(np.abs(img_rgb.astype(np.int32) - target_color), axis=-1)
                color_mask[diff < 60] = 255
                # 严禁扩散到大框外部
                color_mask = cv2.bitwise_and(color_mask, sam_mask_uint8)

            # 第三步：缝合边框与文字
            final_mask = cv2.bitwise_or(edge_mask, color_mask)
        else:
            final_mask = sam_mask_uint8

        # 4. 生成多边形阵列 (因为空心和文字是离散的，会生成多个多边形)
        contours, _ = cv2.findContours(final_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        polygons = []
        for cnt in contours:
            if cv2.contourArea(cnt) > 5:  # 过滤极小噪点
                epsilon = 0.002 * cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, epsilon, True)
                if len(approx) >= 3:  # 保证可以围成面
                    polygons.append(approx.squeeze().tolist())

        return polygons  # 返回包含多个多边形的列表