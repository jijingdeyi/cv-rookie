import cv2
import torch
from torchvision import utils as vutils
from PIL import Image
import os
import numpy as np
from skimage.io import imsave


def save_image_tensor(input_tensor: torch.Tensor, filename):

    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)

    input_tensor = input_tensor.clone().detach()

    input_tensor = input_tensor.to(torch.device('cpu'))
    vutils.save_image(input_tensor, filename)


def save_image_tensor2cv2(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)

    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))

    input_tensor = input_tensor.squeeze()
    img_np = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    cv2.imwrite(filename, img_np)


def save_image_tensor2pillow(input_tensor: torch.Tensor, filename):
    assert (len(input_tensor.shape) == 4 and input_tensor.shape[0] == 1)
    input_tensor = input_tensor.clone().detach()
    input_tensor = input_tensor.to(torch.device('cpu'))
    input_tensor = input_tensor.squeeze()
    img_np = input_tensor.mul_(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).type(torch.uint8).numpy()
    im = Image.fromarray(img_np)
    im.save(filename)

def read_data(root: str) -> list[list[str]]:
    """
    Read image paths from dataset root.

    Args:
        root (str): dataset root path.

    Returns:
        list[list[str]]: visible paths and infrared paths.
    """
    
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    train_root = root

    supported = [".jpg", ".JPG", ".png", ".PNG", ".bmp", 'tif', 'TIF'] 

    train_visible_root = os.path.join(train_root, "vi")
    train_infrared_root= os.path.join(train_root, "ir")

    train_visible_path = [os.path.join(train_visible_root, i) for i in os.listdir(train_visible_root)
                  if os.path.splitext(i)[-1] in supported]
    train_infrared_path = [os.path.join(train_infrared_root, i) for i in os.listdir(train_infrared_root)
                  if os.path.splitext(i)[-1] in supported]

    train_visible_path.sort()
    train_infrared_path.sort()

    assert len(train_visible_path) == len(train_infrared_path), ' The length of vi and ir images does not match. vi: {}, ir: {}'.\
                                         format(len(train_visible_path), len(train_infrared_path))
    
    # print("Visible and Infrared images check finish")
    # print("{} visible images for training.".format(len(train_visible_path)))
    print("{} image pairs for training.".format(len(train_infrared_path)))

    train_low_light_path_list = [train_visible_path, train_infrared_path]
    return train_low_light_path_list


def image_read(path, mode='RGB'):
    """
    read image from path, and convert to specified mode.
    """
    img_BGR = cv2.imread(path)  
    # 默认 shape: (H, W, 3)，dtype=uint8，通道顺序 BGR
    
    assert mode in ['RGB','GRAY','YCrCb'], 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
        # shape: (H, W, 3)，dtype=uint8，通道顺序 RGB
    elif mode == 'GRAY':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
        # shape: (H, W)，dtype=uint8，单通道灰度
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
        # shape: (H, W, 3)，dtype=uint8，通道顺序 Y, Cr, Cb
    return img


def image_save(image, imagename, savepath, CrCb=None):
    """Save gray image to savepath as imagename. 
    If CrCb whose shape is (H, W, 2), save RGB image to savepath/RGB as imagename."""
    temp = np.squeeze(image)
    path1 = os.path.join(savepath, 'RGB')
    path2 = os.path.join(savepath, 'Gray')
    if not os.path.exists(path2):
        os.makedirs(path2)
    imsave(os.path.join(path2, "{}.png".format(imagename)), temp) # If error, make sure imageio==2.26.0 is installed.

    if CrCb is not None:
        assert len(CrCb.shape) == 3 and CrCb.shape[2] == 2, "CrCb error"
        temp_RGB = cv2.cvtColor(np.concatenate((temp[..., np.newaxis], CrCb), axis=2), cv2.COLOR_YCrCb2RGB)
        if not os.path.exists(path1):
            os.makedirs(path1)
        temp_RGB = np.clip(temp_RGB, 0, 255)
        imsave(os.path.join(path1, "{}.png".format(imagename)), temp_RGB)