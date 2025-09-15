import os
from PIL import Image
import numpy as np


def combine_gray_vi(gray_folder, vi_folder, out_folder):
    """
    将灰度图 (IR/Gray) 和 可见光图 (VI) 融合成 RGB 图像，并保存到 out_folder。
    融合方式：Gray 替换 VI 的 Y 通道，保留 VI 的 Cb、Cr 通道。

    Args:
        gray_folder (str): 灰度图所在目录  e.g './datasets/M3FD/images/fused/'
        vi_folder (str): 可见光图所在目录 e.g './datasets/M3FD/images/vi/'
        out_folder (str): 输出保存目录 e.g 
    """
    os.makedirs(out_folder, exist_ok=True)

    for gray_name in os.listdir(gray_folder):
        if gray_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
            gray_base_name = os.path.splitext(gray_name)[0]

            for vi_name in os.listdir(vi_folder):
                if vi_name.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp')):
                    vi_base_name = os.path.splitext(vi_name)[0]
                    if gray_base_name == vi_base_name:
                        # 读灰度图
                        gray_img_path = os.path.join(gray_folder, gray_name)
                        gray_img = Image.open(gray_img_path).convert('L')

                        # 读可见光图
                        vi_img_path = os.path.join(vi_folder, vi_name)
                        vi_img = Image.open(vi_img_path).convert('YCbCr')
                        _, cb, cr = vi_img.split()

                        # 用灰度替换Y通道
                        combine_img = Image.merge("YCbCr", (gray_img, cb, cr)).convert('RGB')

                        # 保存结果
                        output_path = os.path.join(out_folder, f"{gray_base_name}.png")
                        combine_img.save(output_path)
                        print(f"Saved combined image to {output_path}")
                        break

    print("✅ All images have been processed and saved.")


def rgb_to_ycbcr(image):
    rgb_array = np.array(image)

    transform_matrix = np.array([[0.299, 0.587, 0.114],
                                 [-0.169, -0.331, 0.5],
                                 [0.5, -0.419, -0.081]])

    ycbcr_array = np.dot(rgb_array, transform_matrix.T)

    y_channel = ycbcr_array[:, :, 0]
    cb_channel = ycbcr_array[:, :, 1]
    cr_channel = ycbcr_array[:, :, 2]
    
    y_channel = np.clip(y_channel, 0, 255)
    return y_channel, cb_channel, cr_channel


def ycbcr_to_rgb(y, cb, cr):
    ycbcr_array = np.stack((y, cb, cr), axis=-1)

    transform_matrix = np.array([[1, 0, 1.402],
                                 [1, -0.344136, -0.714136],
                                 [1, 1.772, 0]])
    rgb_array = np.dot(ycbcr_array, transform_matrix.T)
    rgb_array = np.clip(rgb_array, 0, 255)

    rgb_array = np.round(rgb_array).astype(np.uint8)
    rgb_image = Image.fromarray(rgb_array, mode='RGB')

    return rgb_image


def fuse_cb_cr(Cb1, Cr1, Cb2, Cr2):
    # [0, 255] 范围内的 Cb, Cr 通道
    # Cb 通道
    w1 = np.abs(Cb1 - 128)
    w2 = np.abs(Cb2 - 128)
    Cb = np.where((w1 == 0) & (w2 == 0), 128,
                  (Cb1 * w1 + Cb2 * w2) / (w1 + w2 + 1e-8))  # 防止除 0

    # Cr 通道
    w3 = np.abs(Cr1 - 128)
    w4 = np.abs(Cr2 - 128)
    Cr = np.where((w3 == 0) & (w4 == 0), 128,
                  (Cr1 * w3 + Cr2 * w4) / (w3 + w4 + 1e-8))

    return Cb.astype(np.float32), Cr.astype(np.float32)