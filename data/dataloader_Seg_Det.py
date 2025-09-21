import numpy as np
import torch
from PIL import Image
import os
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor

from data.dataset import image_read


class Trainset_Seg(Dataset):

    def __init__(self, dataname):
        if dataname == "FMB":
            self.IR_path = r"data\FMB\train\ir"
            self.VI_path = r"data\FMB\train\vi"
            self.mask_path = r"data\FMB\train\label"
        elif dataname == "MSRS":
            self.IR_path = r"data\MSRS\train\ir"
            self.VI_path = r"data\MSRS\train\vi"
            self.mask_path = r"data\MSRS\train\label"
        else:
            raise ValueError(f"Unknown dataset: {dataname}")

        self.file_name_list = os.listdir(self.IR_path)

    def __len__(self):
        return len(self.file_name_list)

    def __getitem__(self, index):
        # IR, VI 自动转 float32, [0,1], shape=(1,H,W)
        IR = ToTensor()(
            image_read(os.path.join(self.IR_path, self.file_name_list[index]), "GRAY")
        )
        VI = ToTensor()(
            image_read(os.path.join(self.VI_path, self.file_name_list[index]), "GRAY")
        )

        # segmask 转 long，用于分割任务
        segmask = image_read(
            os.path.join(self.mask_path, self.file_name_list[index]), "GRAY"
        )
        segmask = torch.from_numpy(segmask).long()

        return IR, VI, segmask, index


def cvtColor(image):
    image = image.convert("L")
    return image


class Trainset_Det(Dataset):
    def __init__(self, data_name):
        super(Trainset_Det, self).__init__()
        if data_name == "M3FD":
            self.annotation_lines = self.get_annotation("/data/ykx/M3FD/M3FD_Detection/M3FD_train.txt")
            self.ir_path = "/data/ykx/M3FD/M3FD_Detection/ir"
            self.vi_path = "/data/ykx/M3FD/M3FD_Detection/vi"
            self.h = 768
            self.w = 1024
        elif data_name == "LLVIP":
            self.annotation_lines = self.get_annotation(r"data\LLVIP_train.txt")
            self.ir_path = r"data/LLVIP/ir"
            self.vi_path = r"data/LLVIP/vi"
            self.h = 1024
            self.w = 1280
        else:
            raise ValueError(f"Unknown dataset: {data_name}")

        self.length = len(self.annotation_lines)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        line = self.annotation_lines[index].split()

        ir = cvtColor(Image.open(os.path.join(self.ir_path, line[0]))) # shape (H, W)
        vi = cvtColor(Image.open(os.path.join(self.vi_path, line[0]))) # shape (H, W)

        iw, ih = ir.size

        box = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]]) # list(N, 5) -> array(N, 5)

        # resize to target size
        if iw != self.w or ih != self.h:
            ir = ir.resize((self.w, self.h), Image.Resampling.BICUBIC)
            vi = vi.resize((self.w, self.h), Image.Resampling.BICUBIC)

        ir = np.array(ir, dtype=np.float32)[None, ...] / 255.0
        vi = np.array(vi, dtype=np.float32)[None, ...] / 255.0
        box = np.array(box, dtype=np.float32)
        np.random.shuffle(box)

        nL = len(box)
        labels_out = np.zeros((nL, 6))
        if nL:

            box[:, [0, 2]] = box[:, [0, 2]] / self.h
            box[:, [1, 3]] = box[:, [1, 3]] / self.w

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2

            labels_out[:, 1] = box[:, -1]
            labels_out[:, 2:] = box[:, :4]

        return ir, vi, labels_out, index # (1,H,W), (1,H,W), (nL, 6), int

    def get_annotation(self, annotation_path):
        with open(annotation_path, encoding="utf-8") as f:
            train_lines = f.readlines()
            return train_lines

    def get_random_data(
        self,
        annotation_line,
        input_shape,
        jitter=0.3,
        hue=0.1,
        sat=0.7,
        val=0.4,
        random=True,
    ):
        line = annotation_line.split()

        image = Image.open(line[0])
        image = cvtColor(image)

        iw, ih = image.size
        h, w = input_shape

        box = np.array([np.array(list(map(int, box.split(",")))) for box in line[1:]]) # list(N, 5) -> array(N, 5)

        if not random:
            scale = min(w / iw, h / ih)
            nw = int(iw * scale)
            nh = int(ih * scale)
            dx = (w - nw) // 2
            dy = (h - nh) // 2

            image = image.resize((nw, nh), Image.Resampling.BICUBIC)
            new_image = Image.new("RGB", (w, h), (128, 128, 128))
            new_image.paste(image, (dx, dy))
            image_data = np.array(new_image, np.float32)

            if len(box) > 0:
                np.random.shuffle(box)
                box[:, [0, 2]] = box[:, [0, 2]] * nw / iw + dx
                box[:, [1, 3]] = box[:, [1, 3]] * nh / ih + dy
                box[:, 0:2][box[:, 0:2] < 0] = 0
                box[:, 2][box[:, 2] > w] = w
                box[:, 3][box[:, 3] > h] = h
                box_w = box[:, 2] - box[:, 0]
                box_h = box[:, 3] - box[:, 1]
                box = box[np.logical_and(box_w > 1, box_h > 1)]  # discard invalid box

            return image_data, box

def yolo_dataset_collate(batch):
    ir_images = []
    vi_images = []
    bboxes = []
    index_list = []

    for i, (ir, vi, box, index) in enumerate(batch):
        ir_images.append(ir)      # 已经是 torch.Tensor
        vi_images.append(vi)
        box[:, 0] = i             # 修改 batch index
        bboxes.append(box)
        index_list.append(index)

    ir_images = torch.stack(ir_images).float()   # [B,1,H,W]
    vi_images = torch.stack(vi_images).float()   # [B,1,H,W]
    bboxes = torch.cat(bboxes, dim=0).float()    # [N, 5] 或 [N, 4] depending on格式
    index_list = torch.tensor(index_list, dtype=torch.long)

    return ir_images, vi_images, bboxes, index_list


if __name__ == "__main__":
    import sys
    import os
    import matplotlib.pyplot as plt

    sys.path.append(os.getcwd())
    dataset = Trainset_Det('M3FD')
    ir, vi, labels_out, index = dataset[0]
    plt.imshow(ir.squeeze(0), cmap='gray')
    plt.show()
    
