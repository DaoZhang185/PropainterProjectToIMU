import cv2
import numpy as np
import os


# x1,y1,x2,y2 = 57,35,140,70

def process_image(image):
    histogram = cv2.calcHist([image], [0], None, [256], [0, 256])

    # 找到灰度直方图中的最大值对应的灰度级
    max_value_index = np.argmax(histogram)

    # 定义片段的左右边界
    left_index = max(0, max_value_index - 30)
    right_index = min(255, max_value_index + 30)

    # 创建与灰度图大小相同的全零矩阵
    processed_image = np.zeros_like(image)

    # 将灰度值在片段内的像素值置255，其他置0
    processed_image[(image >= left_index) & (image <= right_index)] = 255
    return processed_image


def make_mask(src_img, poses):
    x1, y1, x2, y2 = poses[0]
    cropped_image = src_img[y1:y2, x1:x2]
    result = np.zeros_like(cropped_image)
    for i in range(1, len(poses)):
        x1, y1, x2, y2 = poses[i]
        temp = src_img[y1:y2, x1:x2]
        mean = temp.mean()
        max_ = temp.max()
        min_ = temp.min()
        # result[(cropped_image >=min_+20) & (cropped_image <= max_-20)] = 255
        result[(cropped_image >= mean - 15) & (cropped_image <= mean + 15)] = 255

    return result


# 阈值生成方法
def process_and_place_image_with_threshold(image_path, poses, threshold):
    x1, y1, x2, y2 = poses
    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise ValueError("Image not found or unable to load")
    # 检查坐标是否有效
    if x1 < 0 or y1 < 0 or x2 > original_image.shape[1] or y2 > original_image.shape[0]:
        raise ValueError("Invalid coordinates")

    # 截取指定区域的图像
    cropped_image = original_image[y1:y2, x1:x2]

    _, binary_image = cv2.threshold(cropped_image, threshold, 255, cv2.THRESH_BINARY)  # 105为阈值，可以自己调整
    kernel = np.ones((3, 3), np.uint8)  # 3x3的核，可以自己调整

    # 进行膨胀处理
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

    # 创建一个与原始图像相同大小的全0图片
    result_image = np.zeros_like(original_image)

    # 将处理后的图像放入到结果图像的指定位置
    result_image[y1:y2, x1:x2] = dilated_image
    return result_image


# 大小框生成方法
def process_and_place_image(image_path, poses, index):
    # 读取原始图像
    # print(image_path)
    # for i in range(1,5):

    # 【修正1】 poses现在是单层列表，直接解包，不要用poses[0]
    x1, y1, x2, y2 = poses

    original_image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if original_image is None:
        raise ValueError("Image not found or unable to load")

    # 检查坐标是否有效
    if x1 < 0 or y1 < 0 or x2 > original_image.shape[1] or y2 > original_image.shape[0]:
        raise ValueError("Invalid coordinates")

    # 截取指定区域的图像
    if int(index) == 2:  # 需要变化的框 果要新增j !=2 or j!=3 等等
        cropped_image = original_image[y1:y2, x1:x2]

        # 二值化处理
        # blur_img = cv2.medianBlur(cropped_image,5)
        _, binary_image = cv2.threshold(cropped_image, 200, 255, cv2.THRESH_BINARY)  # 105为阈值，可以自己调整

        # binary_image = cv2.adaptiveThreshold(blur_img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)
        # binary_image = cv2.adaptiveThreshold(cropped_image,255,cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY,11,2)

        # _, binary_image = cv2.threshold(cropped_image, 125,255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) #105为阈值，可以自己调整
    else:
        binary_image = make_mask(original_image, poses)

    # # 定义膨胀操作的核
    kernel = np.ones((3, 3), np.uint8)  # 2x2的核，可以自己调整

    # 进行膨胀处理
    dilated_image = cv2.dilate(binary_image, kernel, iterations=1)

    # 创建一个与原始图像相同大小的全0图片
    result_image = np.zeros_like(original_image)

    # 将处理后的图像放入到结果图像的指定位置
    result_image[y1:y2, x1:x2] = dilated_image

    return result_image


# 像素点帧差法
# def process_and_place_image_with_frame(image_path, poses):

def make_frame1(img_path, poses):
    #     frame = None
    #     for i, img_path in enumerate(img_paths):
    result = None
    print(img_path)
    for j in range(1, 5):
        # for pos in poses:
        pos = poses[str(j)]
        if j == 1:
            result_img = process_and_place_image_with_threshold(img_path, pos, 130)
        elif j == 3:
            result_img = process_and_place_image_with_threshold(img_path, pos, 85)
        elif j == 4:
            result_img = process_and_place_image_with_threshold(img_path, pos, 80)
        else:
            result_img = process_and_place_image(img_path, pos, j)
        if result is None:
            result = result_img
        else:
            result = result + result_img
    return result


def process_and_place_images(img_paths, poses, save_path, frame1_mask):
    frame = None
    for i, img_path in enumerate(img_paths):
        result = None
        print(img_path)
        for j in range(1, 5):
            # for pos in poses:
            pos = poses[str(j)]
            # y1:y2, x1:x2

            # 【修正2】 pos现在是单层列表，直接解包，不要用pos[0]
            x1, y1, x2, y2 = pos

            if j != 2:  # 指定需要变化的框如.果要新增j !=2 or j!=3 等等
                result_img = np.zeros_like(frame1_mask).astype(np.uint8)
                result_img[y1:y2, x1:x2] = frame1_mask[y1:y2, x1:x2]
            else:
                result_img = process_and_place_image(img_path, pos, j)

            if result is None:
                result = result_img
            else:
                result = result + result_img
        # if
        save_root = os.path.join(save_path, os.path.basename(img_path))
        cv2.imwrite(save_root, result)


if __name__ == '__main__':
    # 定义区域坐标
    # 注意：区域 4 必须恢复为 [ [大框], [取样框1], [取样框2]... ] 的格式
    # 我这里使用了您之前提供的完整坐标
    poses = {
        '1': [122, 75, 371, 189],
        '2': [185, 958, 1366, 1031],
        '3': [1698, 208, 1761, 773],
        '4': [1645, 939, 1870, 996]
    }

    # 路径配置
    img_path_dir = '/home/FeiLong_Grp/ZhangMengHui/Model/Model/Model/ProPainter/results/imgs'
    save_path_dir = '/home/FeiLong_Grp/ZhangMengHui/Model/Model/Model/ProPainter/results/masks'

    # 请确保此路径下的 frame_00000001.png 存在
    base_frame_path = '/home/FeiLong_Grp/ZhangMengHui/Model/Model/Model/ProPainter/results/finalresults08/scene_26/inputs/imgs/frame_00000009.png'

    if not os.path.exists(img_path_dir):
        print(f"Error: Input directory {img_path_dir} does not exist.")
        exit(1)

    img_files = sorted([f for f in os.listdir(img_path_dir) if f.lower().endswith(('.png', '.jpg'))])
    if not img_files:
        print("No images found.")
        exit(1)

    img_paths = [os.path.join(img_path_dir, f) for f in img_files]

    # 1. 生成第一帧 Mask (区域4在此处通过 make_mask 生成静态掩码)
    frame1_mask = make_frame1(base_frame_path, poses)

    # 2. 批量处理 (区域4复用第一帧结果，区域2动态生成)
    process_and_place_images(img_paths, poses, save_path_dir, frame1_mask)

    print("Done!")