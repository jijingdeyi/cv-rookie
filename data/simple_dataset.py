from PIL import Image
import torch
from torch.utils.data import Dataset
import os
from glob import glob
from torchvision.transforms import RandomCrop
import torchvision.transforms.functional as F
import torchvision.transforms as transforms


class RandomCropPair:
    def __init__(self, size):
        self.size = size  # 裁剪尺寸 (h, w)

    def __call__(self, vis_img, ir_img):
        # 获取随机裁剪参数
        i, j, h, w = RandomCrop.get_params(vis_img, output_size=self.size)
        # 对可见光和红外图像使用相同的裁剪参数
        vis_img = F.crop(vis_img, i, j, h, w)
        ir_img = F.crop(ir_img, i, j, h, w)

        vis_img = transforms.ToTensor()(vis_img)
        ir_img = transforms.ToTensor()(ir_img)
        return vis_img, ir_img

class SimpleDataSet(Dataset):
    def __init__(self, visible_path, infrared_path, phase="train", transform=None):
        self.phase = phase
        self.visible_files = sorted(glob(os.path.join(visible_path, "*.*")))
        self.infrared_files = sorted(glob(os.path.join(infrared_path, "*.*")))
        self.transform = transform

    def __len__(self):
        l = len(self.infrared_files)
        return l

    def __getitem__(self, item):
        image_A_path = self.visible_files[item]
        image_B_path = self.infrared_files[item]
        image_A = Image.open(image_A_path).convert(mode='RGB')
        image_B = Image.open(image_B_path).convert(mode='L')  

        if self.transform is not None:
            if isinstance(self.transform, RandomCropPair):
                image_A, image_B = self.transform(image_A, image_B)
            else:
                image_A = self.transform(image_A)
                image_B = self.transform(image_B)

        name = image_A_path.replace("\\", "/").split("/")[-1].split(".")[0]

        return image_A, image_B, name

    @staticmethod
    def collate_fn(batch):
        images_A, images_B, name = zip(*batch)
        images_A = torch.stack(images_A, dim=0)
        images_B = torch.stack(images_B, dim=0)
        return images_A, images_B, name


