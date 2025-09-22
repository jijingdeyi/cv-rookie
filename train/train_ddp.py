# train_ddp_demo.py
import os, argparse, random
import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

import train.ddp as ddp  # ← 就用你给的那个文件名保存：misc.py

# torch.backends.cudnn.benchmark = True # would speed up training when the input size is fixed

"""
CUDA_VISIBLE_DEVICES=2,3 torchrun --standalone --nproc_per_node=2 -m train.train_ddp --epochs 2 --batch-size 16 --num-workers 2 --lr 1e-3 --log-dir runs/ddp_demo

"""

# ============ 简单可跑的数据集/模型 ============


class RandomImageDataset(Dataset):
    """(N, 3, 32, 32) 随机图 + 10 类标签，用于演示 DDP 流程"""

    def __init__(self, n=666, seed=0):
        rng = np.random.RandomState(seed)
        self.x = rng.randn(n, 3, 32, 32).astype("float32")
        self.y = rng.randint(0, 10, size=(n,), dtype="int64")

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.from_numpy(self.x[idx]), int(self.y[idx])


class TinyCNN(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1) # B x 64
        return self.classifier(x)


# ============ 训练 / 验证 ============


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_loaders(args, world_size, rank):
    train_set = RandomImageDataset(n=512, seed=42)
    val_set = RandomImageDataset(n=64, seed=233)

    train_sampler = DistributedSampler(
        train_set, num_replicas=world_size, rank=rank, shuffle=True
    )
    val_sampler = DistributedSampler(
        val_set, num_replicas=world_size, rank=rank, shuffle=False
    )

    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_set,
        batch_size=args.batch_size,
        sampler=val_sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader


def train_one_epoch(model, loader, optimizer, scaler, device, epoch, writer=None):
    model.train()
    loss_fn = nn.CrossEntropyLoss()
    metric = ddp.MetricLogger(delimiter="  ")
    metric.add_meter("lr", ddp.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    # DDP: 每个 epoch 前重设采样种子，保证各进程 shuffle 一致且每轮不同
    if isinstance(loader.sampler, DistributedSampler):
        loader.sampler.set_epoch(epoch)

    header = f"Epoch: [{epoch}]"
    for step, (x, y) in enumerate(
        metric.log_every(loader, print_freq=50, header=header)
    ):
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device, dtype=torch.long)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast_mode.autocast(enabled=(device.type == "cuda")):
            logits = model(x)
            loss = loss_fn(logits, y)

        # 使用你提供的 AMP scaler 封装（会自动 unscale、step、update，并返回 grad_norm）
        grad_norm = scaler(
            loss=loss,
            optimizer=optimizer,
            clip_grad=None,
            parameters=model.parameters(),
            create_graph=False,
            update_grad=True,
        )

        # 记录
        loss_val = loss.detach().item()
        metric.update(loss=loss_val, lr=optimizer.param_groups[0]["lr"])
        if writer is not None and ddp.is_main_process():
            global_step = (epoch - 1) * len(loader) + step
            writer.add_scalar("train/loss", loss_val, global_step)
            if grad_norm is not None:
                writer.add_scalar("train/grad_norm", grad_norm.item(), global_step)

    # 打印本 epoch 的平均 loss（所有进程平均）
    avg_loss_tensor = torch.tensor(
        metric.meters["loss"].global_avg, device=device, dtype=torch.float32
    )
    avg_loss = ddp.all_reduce_mean(avg_loss_tensor)
    if ddp.is_main_process() and writer is not None:
        writer.add_scalar("train/epoch_loss", avg_loss, epoch)
    return avg_loss


@torch.no_grad()
def evaluate(model, loader, device, epoch, writer=None):
    model.eval()
    loss_fn = nn.CrossEntropyLoss()
    total_loss, total_correct, total_n = 0.0, 0, 0

    for x, y in loader:
        x = x.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device, dtype=torch.long)
        logits = model(x)
        loss = loss_fn(logits, y)
        total_loss += loss.item() * x.size(0)
        total_correct += (logits.argmax(1) == y).sum().item()
        total_n += x.size(0)

    # 跨进程平均
    dev = device if device.type == "cuda" else torch.device("cpu")
    loss_mean = ddp.all_reduce_mean(
        torch.tensor(total_loss / total_n, device=dev, dtype=torch.float32)
    )
    acc_mean = ddp.all_reduce_mean(
        torch.tensor(total_correct / total_n, device=dev, dtype=torch.float32)
    )

    if ddp.is_main_process() and writer is not None:
        writer.add_scalar("val/loss", float(loss_mean), epoch)
        writer.add_scalar("val/acc", float(acc_mean), epoch)
    if ddp.is_main_process():
        print(
            f"[VAL] epoch={epoch}  loss={float(loss_mean):.4f}  acc={float(acc_mean):.4f}"
        )
    return float(loss_mean), float(acc_mean)


# ============ 主函数 ============


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument(
        "--batch-size", type=int, default=128, help="per-GPU batch size"
    )
    parser.add_argument("--num-workers", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-dir", type=str, default=None)

    # 兼容 misc.init_distributed_mode 的参数
    parser.add_argument("--dist_on_itp", action="store_true")
    parser.add_argument("--dist_url", default="env://")
    parser.add_argument("--world_size", default=1, type=int)
    parser.add_argument("--rank", default=0, type=int)
    parser.add_argument("--gpu", default=0, type=int)

    args = parser.parse_args()

    # 分布式初始化（你提供的封装）
    ddp.init_distributed_mode(args)

    # 设备
    device = (
        torch.device("cuda", args.gpu)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    # cuDNN 策略（固定尺寸追求速度）
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True

    set_seed(args.seed)

    # 数据
    train_loader, val_loader = build_loaders(
        args, world_size=ddp.get_world_size(), rank=ddp.get_rank()
    )

    # 模型/优化器
    model = TinyCNN().to(device)
    # 可选：SyncBN（需要时再开）
    # model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = DDP(
        model,
        device_ids=[args.gpu] if device.type == "cuda" else None,
        output_device=args.gpu if device.type == "cuda" else None,
        broadcast_buffers=False,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = ddp.NativeScalerWithGradNormCount()  # 你文件里的 AMP 封装

    # 仅在 master 进程写日志/保存
    writer = None

    best_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model, train_loader, optimizer, scaler, device, epoch, writer
        )
        val_loss, val_acc = evaluate(model, val_loader, device, epoch, writer)

        if ddp.is_main_process():
            to_save = {
                "epoch": epoch,
                "model": model.module.state_dict(),  # DDP 取 module
                "optimizer": optimizer.state_dict(),
                "scaler": scaler.state_dict(),
                "val_acc": val_acc,
            }
            save_path = os.path.join(args.log_dir or ".", "last.pt")
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            ddp.save_on_master(to_save, save_path)
            if val_acc > best_acc:
                best_acc = val_acc
                ddp.save_on_master(
                    to_save, os.path.join(args.log_dir or ".", "best.pt")
                )
                print(f"[CKPT] epoch={epoch}  acc={best_acc:.4f}  (saved)")

    if writer is not None:
        writer.close()


if __name__ == "__main__":
    main()
