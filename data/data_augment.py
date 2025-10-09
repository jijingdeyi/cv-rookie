import numpy as np
import cv2
import os
# from skimage.io import imsave
import torch
import torch.nn as nn
import torch.nn.functional as F
import kornia
import random
from collections.abc import Iterable


class Transformer():
    def __init__(self, shift_n, rotate_n, flip_n):
        self.shift_n = shift_n
        self.rotate_n = rotate_n
        self.flip_n = flip_n

    def apply(self, x):
        if self.shift_n > 0:
            x_shift = shift_random(x, self.shift_n)
        if self.rotate_n > 0:
            x_rotate = rotate_random(x, self.rotate_n)
        if self.flip_n > 0:
            x_flip = flip_random(x, self.flip_n)

        if self.shift_n > 0:
            x = torch.cat((x, x_shift), 0)
        if self.rotate_n > 0:
            x = torch.cat((x, x_rotate), 0)
        if self.flip_n > 0:
            x = torch.cat((x, x_flip), 0)
        return x


def shift_random(x, n_trans=5):
    H, W = x.shape[-2], x.shape[-1]
    assert n_trans <= H - 1 and n_trans <= W - \
        1, 'n_shifts should less than {}'.format(H-1)
    shifts_row = random.sample(
        list(np.concatenate([-1*np.arange(1, H), np.arange(1, H)])), n_trans)
    shifts_col = random.sample(
        list(np.concatenate([-1*np.arange(1, W), np.arange(1, W)])), n_trans)
    x = torch.cat([torch.roll(x, shifts=[sx, sy], dims=[-2, -1]).type_as(x)
                  for sx, sy in zip(shifts_row, shifts_col)], dim=0)
    return x


def rotate_random(data, n_trans=5, random_rotate=False):
    if random_rotate:
        theta_list = random.sample(list(np.arange(1, 359)), n_trans)
    else:
        theta_list = np.arange(10, 360, int(360 / n_trans))
    data = torch.cat([kornia.geometry.transform.rotate(data, torch.Tensor(
        [theta]).type_as(data))for theta in theta_list], dim=0)
    return data


def flip_random(data, n_trans=3):
    assert n_trans <= 3, 'n_flip should less than 3'

    if n_trans >= 1:
        data1 = kornia.geometry.transform.hflip(data)
    if n_trans >= 2:
        data2 = kornia.geometry.transform.vflip(data)
        data1 = torch.cat((data1, data2), 0)
    if n_trans == 3:
        data1 = torch.cat((data1, kornia.geometry.transform.hflip(data2)), 0)
    return data1


def is_low_contrast(image, fraction_threshold=0.1, lower_percentile=10,
                    upper_percentile=90):
    """Determine if an image is low contrast."""
    limits = np.percentile(image, [lower_percentile, upper_percentile])
    ratio = (limits[1] - limits[0]) / limits[1]
    return ratio < fraction_threshold


# souce and target augment

def get_2dshape(shape, *, zero=True):
    # return (h, w)
    if not isinstance(shape, Iterable):
        shape = int(shape)
        shape = (shape, shape)
    else:
        h, w = map(int, shape)
        shape = (h, w)
    if zero:
        minv = 0
    else:
        minv = 1

    assert min(shape) >= minv, 'invalid shape: {}'.format(shape)
    return shape


def random_crop_pad_to_shape(img, crop_pos, crop_size, pad_label_value):
    h, w = img.shape[:2]  # img: H, W, C
    start_crop_h, start_crop_w = crop_pos
    assert ((start_crop_h < h) and (start_crop_h >= 0))
    assert ((start_crop_w < w) and (start_crop_w >= 0))

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    img_crop = img[start_crop_h:start_crop_h + crop_h,
                   start_crop_w:start_crop_w + crop_w, ...]

    img_, margin = pad_image_to_shape(img_crop, crop_size, cv2.BORDER_CONSTANT,
                                      pad_label_value)

    return img_, margin


def generate_random_crop_pos(ori_size, crop_size):
    ori_size = get_2dshape(ori_size)
    h, w = ori_size

    crop_size = get_2dshape(crop_size)
    crop_h, crop_w = crop_size

    pos_h, pos_w = 0, 0

    if h > crop_h:
        pos_h = random.randint(0, h - crop_h + 1)

    if w > crop_w:
        pos_w = random.randint(0, w - crop_w + 1)

    return pos_h, pos_w


def pad_image_to_shape(img, shape, border_mode, value):
    margin = np.zeros(4, np.uint32)
    shape = get_2dshape(shape)
    pad_height = shape[0] - img.shape[0] if shape[0] - img.shape[0] > 0 else 0
    pad_width = shape[1] - img.shape[1] if shape[1] - img.shape[1] > 0 else 0

    margin[0] = pad_height // 2
    margin[1] = pad_height // 2 + pad_height % 2
    margin[2] = pad_width // 2
    margin[3] = pad_width // 2 + pad_width % 2

    img = cv2.copyMakeBorder(
        img, margin[0], margin[1], margin[2], margin[3], border_mode, value=value)

    return img, margin


def pad_image_size_to_multiples_of(img, multiple, pad_value):
    h, w = img.shape[:2]
    d = multiple

    def canonicalize(s):
        v = s // d
        return (v + (v * d != s)) * d

    th, tw = map(canonicalize, (h, w))

    return pad_image_to_shape(img, (th, tw), cv2.BORDER_CONSTANT, pad_value)


def resize_ensure_shortest_edge(img, edge_length,
                                interpolation_mode=cv2.INTER_LINEAR):
    assert isinstance(edge_length, int) and edge_length > 0, edge_length
    h, w = img.shape[:2]
    if h < w:
        ratio = float(edge_length) / h
        th, tw = edge_length, max(1, int(ratio * w))
    else:
        ratio = float(edge_length) / w
        th, tw = max(1, int(ratio * h)), edge_length
    img = cv2.resize(img, (tw, th), interpolation=interpolation_mode)

    return img


def random_scale(img, gt, scales):
    scale = random.choice(scales)
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return img, gt, scale


def random_scale_rgbx(img, gt, modal_x, scales):
    scale = random.choice(scales)
    sh = int(img.shape[0] * scale)
    sw = int(img.shape[1] * scale)
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)
    modal_x = cv2.resize(modal_x, (sw, sh), interpolation=cv2.INTER_LINEAR)

    return img, gt, modal_x, scale


def random_scale_with_length(img, gt, length):
    size = random.choice(length)
    sh = size
    sw = size
    img = cv2.resize(img, (sw, sh), interpolation=cv2.INTER_LINEAR)
    gt = cv2.resize(gt, (sw, sh), interpolation=cv2.INTER_NEAREST)

    return img, gt, size


def random_mirror(img, gt):
    if random.random() >= 0.5:
        img = cv2.flip(img, 1)
        gt = cv2.flip(gt, 1)

    return img, gt,


def random_rotation(img, gt):
    angle = random.random() * 20 - 10
    h, w = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
    img = cv2.warpAffine(img, rotation_matrix, (w, h), flags=cv2.INTER_LINEAR)
    gt = cv2.warpAffine(gt, rotation_matrix, (w, h), flags=cv2.INTER_NEAREST)

    return img, gt


def random_gaussian_blur(img):
    gauss_size = random.choice([1, 3, 5, 7])
    if gauss_size > 1:
        # do the gaussian blur
        img = cv2.GaussianBlur(img, (gauss_size, gauss_size), 0)

    return img


def center_crop(img, shape):
    h, w = shape[0], shape[1]
    y = (img.shape[0] - h) // 2
    x = (img.shape[1] - w) // 2
    return img[y:y + h, x:x + w]


def random_crop(img, gt, size):
    if isinstance(size, int or float):
        size = (int(size), int(size))
    else:
        size = size

    h, w = img.shape[:2]
    crop_h, crop_w = size[0], size[1]

    if h > crop_h:
        x = random.randint(0, h - crop_h + 1)
        img = img[x:x + crop_h, :, :]
        gt = gt[x:x + crop_h, :]

    if w > crop_w:
        x = random.randint(0, w - crop_w + 1)
        img = img[:, x:x + crop_w, :]
        gt = gt[:, x:x + crop_w]

    return img, gt
