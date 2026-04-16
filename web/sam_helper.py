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

    def predict_from_b64(self, b64_img, box=None, point_coords=None, point_labels=None):
        """
        接收前端传来的 Base64 图片、框坐标(可选)和点击坐标(可选)，返回多边形轮廓
        """
        # 1. Base64 转 OpenCV 图像
        img_data = base64.b64decode(b64_img.split(',')[1] if ',' in b64_img else b64_img)
        nparr = np.frombuffer(img_data, np.uint8)
        img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)

        # 2. 载入图像到 SAM 编码器 (这一步相对耗时，但单帧可以接受)
        self.predictor.set_image(img_rgb)

        # 3. 整理 Prompt
        input_box = np.array(box) if box else None
        input_point = np.array(point_coords) if point_coords else None
        input_label = np.array(point_labels) if point_labels else None

        # 4. 执行预测
        masks, scores, logits = self.predictor.predict(
            point_coords=input_point,
            point_labels=input_label,
            box=input_box,
            multimask_output=False  # 我们只需要最高置信度的一个 Mask
        )

        mask = masks[0]  # bool 类型的 2D 数组

        # 5. 将布尔 Mask 转换为多边形轮廓 (Polygon) 返回给前端
        mask_uint8 = (mask * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            return []

        # 找最大的轮廓，并做平滑近似，减少前端渲染压力和 JSON 体积
        max_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.002 * cv2.arcLength(max_contour, True)
        approx_contour = cv2.approxPolyDP(max_contour, epsilon, True)

        # 展平为 [[x1,y1], [x2,y2]...] 格式
        polygon = approx_contour.squeeze().tolist()
        return polygon