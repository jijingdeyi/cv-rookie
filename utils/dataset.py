import os
import cv2
import numpy as np
from skimage.io import imsave

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
    """read image from path, and convert to specified mode.
    """
    img_BGR = cv2.imread(path).astype('float32')
    assert mode in ['RGB','GRAY','YCrCb'], 'mode error'
    if mode == 'RGB':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2RGB)
    elif mode == 'GRAY':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2GRAY)
    elif mode == 'YCrCb':
        img = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb)
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