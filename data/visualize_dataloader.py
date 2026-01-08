"""
通用 DataLoader 可视化工具

支持可视化任意 DataLoader 的输出，自动检测图像格式并生成可视化结果。

使用示例:
    # 作为函数使用
    from data.visualize_dataloader import inspect_dataloader, visualize_batch
    from your_module import your_dataloader
    
    inspect_dataloader(your_dataloader, num_batches=3)
    
    # 作为命令行工具使用
    python -m data.visualize_dataloader --module your_module --loader your_dataloader
"""
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from pathlib import Path
from typing import Union, Tuple, List, Optional, Any


def tensor_to_pil_image(tensor: torch.Tensor, is_grayscale: Optional[bool] = None) -> Image.Image:
    """
    将 PyTorch tensor 转换为 PIL Image
    
    Args:
        tensor: 输入 tensor，支持 [C, H, W] 或 [H, W] 格式
        is_grayscale: 是否为灰度图，None 时自动检测
        
    Returns:
        PIL Image 对象
    """
    
    tensor = tensor.detach().cpu()
    
    # 处理不同维度
    if tensor.dim() == 4:  # [B, C, H, W]
        tensor = tensor[0]  # 取第一张
    if tensor.dim() == 3:
        if tensor.shape[0] == 1:  # [1, H, W] 灰度图
            tensor = tensor.squeeze(0)
            is_grayscale = True
        else:  # [C, H, W] RGB
            tensor = tensor.permute(1, 2, 0)
            is_grayscale = False
    elif tensor.dim() == 2:  # [H, W]
        is_grayscale = True
    
    # 转换为 numpy
    arr = tensor.numpy()
    
    # 归一化到 [0, 255]
    if arr.max() <= 1.0:
        arr = (arr * 255).astype(np.uint8)
    else:
        arr = arr.clip(0, 255).astype(np.uint8)
    
    # 转换为 PIL Image
    if is_grayscale or arr.ndim == 2:
        return Image.fromarray(arr, mode='L')
    else:
        return Image.fromarray(arr, mode='RGB')


def create_image_grid(
    images: List[Image.Image],
    labels: Optional[List[str]] = None,
    n_cols: int = 2,
    padding: int = 10,
    bg_color: str = 'white'
) -> Image.Image:
    """
    创建图像网格
    
    Args:
        images: PIL Image 列表
        labels: 可选的标签列表
        n_cols: 列数
        padding: 图像间距
        bg_color: 背景颜色
        
    Returns:
        拼接后的 PIL Image
    """
    n_images = len(images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    # 计算每张图像的尺寸
    img_width = max(img.width for img in images)
    img_height = max(img.height for img in images)
    
    # 计算总尺寸（包含标签高度）
    label_height = 30 if labels else 0
    cell_width = img_width + padding * 2
    cell_height = img_height + label_height + padding * 2
    
    # 创建画布
    grid_width = cell_width * n_cols
    grid_height = cell_height * n_rows
    grid = Image.new('RGB', (grid_width, grid_height), color=bg_color)
    
    # 放置图像
    for idx, img in enumerate(images):
        row = idx // n_cols
        col = idx % n_cols
        
        x = col * cell_width + padding
        y = row * cell_height + padding
        
        # 居中放置图像
        offset_x = (cell_width - img.width) // 2
        offset_y = (cell_height - label_height - img.height) // 2 + label_height
        
        grid.paste(img, (x + offset_x, y + offset_y))
        
        # 添加标签
        if labels and idx < len(labels):
            draw = ImageDraw.Draw(grid)
            label_text = labels[idx]
            font = ImageFont.load_default()
            text_bbox = draw.textbbox((0, 0), label_text, font=font)
            text_width = text_bbox[2] - text_bbox[0]
            text_x = x + (cell_width - text_width) // 2
            text_y = y + 5
            draw.text((text_x, text_y), label_text, fill='black', font=font)
    
    return grid


def visualize_batch(
    *tensors: torch.Tensor,
    labels: Optional[List[str]] = None,
    save_path: Optional[Union[str, Path]] = None,
    show: bool = False,
    max_images: int = 4,
    layout: str = 'side_by_side',
    n_cols: int = 2
) -> Image.Image:
    """
    可视化一个 batch 的多个 tensor
    
    Args:
        *tensors: 要可视化的 tensor，每个应该是 [B, C, H, W] 或 [B, H, W] 格式
        labels: tensor 的标签列表，如 ['IR', 'VI']
        save_path: 保存路径
        show: 是否显示图像
        max_images: 最多显示多少张图像
        layout: 布局方式，'side_by_side' (并排) 或 'grid' (网格)
        n_cols: 网格布局时的列数
        
    Returns:
        拼接后的 PIL Image
    """
    if not tensors:
        raise ValueError("至少需要提供一个 tensor")
    
    batch_size = tensors[0].shape[0]
    n_images = min(batch_size, max_images)
    n_tensors = len(tensors)
    
    if labels is None:
        labels = [f'Image {i+1}' for i in range(n_tensors)]
    elif len(labels) != n_tensors:
        labels = labels[:n_tensors] + [f'Image {i+1}' for i in range(len(labels), n_tensors)]
    
    if layout == 'side_by_side':
        # 并排布局：每对图像并排显示
        combined_images = []
        for i in range(n_images):
            pair_images = []
            for tensor, label in zip(tensors, labels):
                img = tensor_to_pil_image(tensor[i])
                pair_images.append(img)
            
            # 水平拼接
            total_width = sum(img.width for img in pair_images)
            max_height = max(img.height for img in pair_images)
            combined = Image.new('RGB', (total_width, max_height), color='white')
            
            x_offset = 0
            for img in pair_images:
                combined.paste(img, (x_offset, 0))
                x_offset += img.width
            
            combined_images.append(combined)
        
        # 垂直拼接所有图像对
        if len(combined_images) > 1:
            total_height = sum(img.height for img in combined_images)
            max_width = max(img.width for img in combined_images)
            final_image = Image.new('RGB', (max_width, total_height), color='white')
            y_offset = 0
            for img in combined_images:
                final_image.paste(img, (0, y_offset))
                y_offset += img.height
        else:
            final_image = combined_images[0]
    
    else:  # grid layout
        # 网格布局：所有图像排列成网格
        all_images = []
        for i in range(n_images):
            for tensor, label in zip(tensors, labels):
                img = tensor_to_pil_image(tensor[i])
                all_images.append(img)
        
        final_image = create_image_grid(all_images, labels=None, n_cols=n_cols)
    
    if save_path:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        final_image.save(save_path)
        print(f"✓ Saved to {save_path}")
        print(f"  Batch size: {batch_size}, Shapes: {[t.shape for t in tensors]}")
    
    if show:
        final_image.show()
    
    return final_image


def inspect_dataloader(
    dataloader: Any,
    num_batches: int = 1,
    save_dir: Union[str, Path] = 'output/dataloader_check',
    labels: Optional[List[str]] = None
) -> None:
    """
    检查 DataLoader 的输出
    
    Args:
        dataloader: DataLoader 对象
        num_batches: 检查多少个 batch
        save_dir: 保存目录
        labels: tensor 的标签列表
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(f"DataLoader Inspection")
    print("=" * 60)
    print(f"Batch size: {dataloader.batch_size}")
    print(f"Number of batches to check: {num_batches}\n")
    
    for batch_idx, batch_data in enumerate(dataloader):
        if batch_idx >= num_batches:
            break
        
        print(f"Batch {batch_idx + 1}:")
        
        # 处理不同的 batch 格式
        if isinstance(batch_data, (list, tuple)):
            tensors = list(batch_data)
        elif isinstance(batch_data, dict):
            tensors = list(batch_data.values())
            if labels is None:
                labels = list(batch_data.keys())
        else:
            tensors = [batch_data]
        
        # 打印信息
        for idx, tensor in enumerate(tensors):
            label = labels[idx] if labels and idx < len(labels) else f'Tensor {idx+1}'
            print(f"  {label}:")
            print(f"    Shape: {tensor.shape}")
            print(f"    Dtype: {tensor.dtype}")
            print(f"    Range: [{tensor.min():.3f}, {tensor.max():.3f}]")
            if tensor.requires_grad:
                print(f"    Requires grad: True")
        print()
        
        # 可视化
        save_path = save_dir / f"batch_{batch_idx + 1}.png"
        try:
            visualize_batch(
                *tensors,
                labels=labels,
                save_path=save_path,
                show=False,
                max_images=min(4, tensors[0].shape[0])
            )
        except Exception as e:
            print(f"  ⚠ Warning: Could not visualize batch {batch_idx + 1}: {e}")
    
    print(f"✓ Visualizations saved to {save_dir}/")
    print("=" * 60)



if __name__ == "__main__":
    
    from data.datasets_MSRS import trainloader, testloader
    inspect_dataloader(trainloader, num_batches=2, save_dir='output/dataloader_check', labels=['IR', 'VI'])
    # inspect_dataloader(testloader, num_batches=1, save_dir='output/dataloader_check', labels=['IR', 'VI'])
