#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
计算多个文件夹中同名图像的ROI区域灰度统计信息
比较不同方法在同一ROI区域的灰度值
"""
from pathlib import Path
import argparse, cv2, numpy as np, sys
import pandas as pd


def imread_any(p: Path):
    """兼容中文路径的图片读取"""
    if p is None or not p.exists():
        return None
    arr = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)


def parse_roi(roi_str: str):
    """解析区域字符串，支持多种格式：
    - 'x,y,w,h' (绝对坐标)
    - 'rx,ry,rw,rh' (相对坐标，相对于每张子图的尺寸，0-1范围)
    - 'cx,cy,size' (中心点+大小，相对坐标)
    """
    parts = [float(x.strip()) for x in roi_str.split(',')]
    if len(parts) == 4:
        # 格式1或2：判断是绝对还是相对
        if all(0 <= p <= 1 for p in parts):
            return {'type': 'relative', 'x': parts[0], 'y': parts[1], 'w': parts[2], 'h': parts[3]}
        else:
            return {'type': 'absolute', 'x': int(parts[0]), 'y': int(parts[1]), 'w': int(parts[2]), 'h': int(parts[3])}
    elif len(parts) == 3:
        # 格式3：中心点+大小
        if all(0 <= p <= 1 for p in parts):
            size = parts[2]
            return {'type': 'center', 'cx': parts[0], 'cy': parts[1], 'size': size}
    raise ValueError(f"Invalid ROI format: {roi_str}. Expected 'x,y,w,h', 'rx,ry,rw,rh', or 'cx,cy,size'")


def get_roi_coords(roi_params, subimg_h, subimg_w):
    """根据ROI参数和子图尺寸计算实际的裁剪坐标"""
    if roi_params['type'] == 'absolute':
        x, y = roi_params['x'], roi_params['y']
        w, h = roi_params['w'], roi_params['h']
        return max(0, x), max(0, y), min(w, subimg_w - x), min(h, subimg_h - y)
    elif roi_params['type'] == 'relative':
        x = int(roi_params['x'] * subimg_w)
        y = int(roi_params['y'] * subimg_h)
        w = int(roi_params['w'] * subimg_w)
        h = int(roi_params['h'] * subimg_h)
        x = max(0, min(x, subimg_w - 1))
        y = max(0, min(y, subimg_h - 1))
        w = min(w, subimg_w - x)
        h = min(h, subimg_h - y)
        return x, y, w, h
    elif roi_params['type'] == 'center':
        cx = roi_params['cx'] * subimg_w
        cy = roi_params['cy'] * subimg_h
        size = roi_params['size']
        if size <= 1:
            # 相对大小
            w = h = int(min(subimg_w, subimg_h) * size)
        else:
            # 绝对大小
            w = h = int(size)
        x = max(0, int(cx - w // 2))
        y = max(0, int(cy - h // 2))
        w = min(w, subimg_w - x)
        h = min(h, subimg_h - y)
        return x, y, w, h
    else:
        # 默认情况（不应该到达这里，但为了类型检查器）
        return 0, 0, subimg_w, subimg_h


def extract_roi(img, roi_params):
    """从图像中提取ROI区域"""
    if img is None:
        return None
    
    h, w = img.shape[:2]
    roi_x, roi_y, roi_w, roi_h = get_roi_coords(roi_params, h, w)
    
    if roi_w <= 0 or roi_h <= 0:
        return None
    
    # 确保不越界
    roi_x = max(0, min(roi_x, w - 1))
    roi_y = max(0, min(roi_y, h - 1))
    roi_w = min(roi_w, w - roi_x)
    roi_h = min(roi_h, h - roi_y)
    
    if roi_w <= 0 or roi_h <= 0:
        return None
    
    # 提取ROI
    roi = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w]
    
    return roi


def compute_gray_statistics(roi, convert_to_gray=True):
    """计算ROI区域的灰度统计信息"""
    if roi is None:
        return None
    
    # 转换为灰度图
    if len(roi.shape) == 3:
        if convert_to_gray:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            # 如果是彩色图，计算三个通道的平均值
            gray = roi.mean(axis=2).astype(np.uint8)
    else:
        gray = roi
    
    # 计算统计信息
    gray_sum = gray.sum()
    gray_mean = gray.mean()
    gray_std = gray.std()
    gray_max = gray.max()
    gray_min = gray.min()
    pixel_count = gray.size
    
    return {
        'sum': gray_sum,
        'mean': gray_mean,
        'std': gray_std,
        'max': gray_max,
        'min': gray_min,
        'count': pixel_count,
        'size': gray.shape  # (h, w)
    }


def main():
    ap = argparse.ArgumentParser(
        description="计算多个文件夹中同名图像的ROI区域灰度统计信息",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 计算ROI区域灰度统计
  python plots/roi_statistics.py \\
    --input-folders method1 method2 method3 method4 method5 method6 \\
    --method-names "Method1" "Method2" "Method3" "Method4" "Method5" "Method6" \\
    --filename 00209D \\
    --roi "230,140,70,160"
  
  # 使用相对坐标
  python plots/roi_statistics.py \\
    --input-folders method1 method2 method3 \\
    --filename 00209D \\
    --roi "0.3,0.3,0.4,0.4"
  
  # 输出CSV文件
  python plots/roi_statistics.py \\
    --input-folders method1 method2 method3 \\
    --filename 00209D \\
    --roi "230,140,70,160" \\
    --output-csv results.csv
        """
    )
    ap.add_argument("--input-folders", nargs="+", required=True,
                    help="多个输入文件夹路径（每个文件夹代表一种方法）")
    ap.add_argument("--method-names", nargs="+", default=None,
                    help="每个文件夹对应的方法名称/标签（数量需与--input-folders一致，留空则使用文件夹名）")
    ap.add_argument("--filename", type=str, required=True,
                    help="要处理的文件名（不含扩展名）")
    ap.add_argument("--roi", type=str, required=True,
                    help="ROI区域，格式: 'x,y,w,h' 或 'rx,ry,rw,rh' 或 'cx,cy,size'")
    ap.add_argument("--exts", nargs="*", default=[".png", ".jpg", ".jpeg", ".bmp"],
                    help="允许的扩展名（含优先级，默认: .png .jpg .jpeg .bmp）")
    ap.add_argument("--output-csv", type=str, default=None,
                    help="输出CSV文件路径（可选）")
    ap.add_argument("--no-convert-gray", action="store_true",
                    help="不将彩色图转换为灰度图，而是计算RGB的平均值")
    
    args = ap.parse_args()
    
    # 验证参数
    folder_paths = [Path(f) for f in args.input_folders]
    for fp in folder_paths:
        if not fp.is_dir():
            print(f"[Error] 文件夹不存在: {fp}", file=sys.stderr)
            sys.exit(1)
    
    # 确定方法名称
    if args.method_names:
        if len(args.method_names) != len(folder_paths):
            print(f"[Error] --method-names 数量 ({len(args.method_names)}) 与 --input-folders 数量 ({len(folder_paths)}) 不一致", file=sys.stderr)
            sys.exit(1)
        method_names = args.method_names
    else:
        method_names = [fp.name for fp in folder_paths]
    
    # 解析ROI
    try:
        roi_params = parse_roi(args.roi)
    except ValueError as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)
    
    print(f"[Info] 处理文件: {args.filename}")
    print(f"[Info] 方法数量: {len(method_names)}")
    print(f"[Info] ROI: {roi_params}")
    print()
    
    # 从每个文件夹中查找指定的文件
    stem = args.filename
    results = []
    
    for idx, folder_path in enumerate(folder_paths):
        img_path = None
        # 按优先级查找文件
        for ext in args.exts:
            ext_low = ext.lower()
            candidate = folder_path / f"{stem}{ext}"
            if candidate.exists() and candidate.is_file():
                img_path = candidate
                break
            # 也尝试小写扩展名
            candidate = folder_path / f"{stem}{ext_low}"
            if candidate.exists() and candidate.is_file():
                img_path = candidate
                break
        
        if not img_path:
            print(f"[Warn] 在 {folder_path} 中未找到文件: {stem}")
            results.append({
                'method': method_names[idx],
                'folder': str(folder_path),
                'file': None,
                'roi_size': None,
                'gray_sum': None,
                'gray_mean': None,
                'gray_std': None,
                'gray_max': None,
                'gray_min': None,
                'pixel_count': None,
            })
            continue
        
        # 读取图像
        img = imread_any(img_path)
        if img is None:
            print(f"[Warn] 无法读取: {img_path}")
            results.append({
                'method': method_names[idx],
                'folder': str(folder_path),
                'file': img_path.name,
                'roi_size': None,
                'gray_sum': None,
                'gray_mean': None,
                'gray_std': None,
                'gray_max': None,
                'gray_min': None,
                'pixel_count': None,
            })
            continue
        
        # 提取ROI
        roi = extract_roi(img, roi_params)
        if roi is None:
            print(f"[Warn] 无法从 {img_path} 提取ROI")
            results.append({
                'method': method_names[idx],
                'folder': str(folder_path),
                'file': img_path.name,
                'roi_size': None,
                'gray_sum': None,
                'gray_mean': None,
                'gray_std': None,
                'gray_max': None,
                'gray_min': None,
                'pixel_count': None,
            })
            continue
        
        # 计算统计信息
        stats = compute_gray_statistics(roi, convert_to_gray=not args.no_convert_gray)
        
        if stats is None:
            print(f"[Warn] 无法计算 {img_path} 的统计信息")
            results.append({
                'method': method_names[idx],
                'folder': str(folder_path),
                'file': img_path.name,
                'roi_size': None,
                'gray_sum': None,
                'gray_mean': None,
                'gray_std': None,
                'gray_max': None,
                'gray_min': None,
                'pixel_count': None,
            })
            continue
        
        results.append({
            'method': method_names[idx],
            'folder': str(folder_path),
            'file': img_path.name,
            'roi_size': f"{roi.shape[0]}x{roi.shape[1]}",
            'gray_sum': stats['sum'],
            'gray_mean': stats['mean'],
            'gray_std': stats['std'],
            'gray_max': stats['max'],
            'gray_min': stats['min'],
            'pixel_count': stats['count'],
        })
        
        print(f"[{idx+1}/{len(folder_paths)}] {method_names[idx]}")
        print(f"  文件: {img_path.name}")
        print(f"  图像尺寸: {img.shape[1]}x{img.shape[0]}")
        print(f"  ROI尺寸: {roi.shape[1]}x{roi.shape[0]}")
        print(f"  灰度总和: {stats['sum']:,.0f}")
        print(f"  平均灰度: {stats['mean']:.2f}")
        print(f"  标准差: {stats['std']:.2f}")
        print(f"  最大/最小: {stats['max']}/{stats['min']}")
        print()
    
    # 找出灰度总和最大的方法
    valid_results = [r for r in results if r['gray_sum'] is not None]
    if valid_results:
        max_sum_result = max(valid_results, key=lambda x: x['gray_sum'])
        print("=" * 60)
        print(f"【灰度总和最大】: {max_sum_result['method']}")
        print(f"  灰度总和: {max_sum_result['gray_sum']:,.0f}")
        print(f"  平均灰度: {max_sum_result['gray_mean']:.2f}")
        print()
        
        # 按灰度总和排序
        sorted_results = sorted(valid_results, key=lambda x: x['gray_sum'], reverse=True)
        print("【排序结果】（按灰度总和降序）:")
        for i, r in enumerate(sorted_results, 1):
            print(f"  {i}. {r['method']:20s}  总和: {r['gray_sum']:>12,.0f}  平均: {r['gray_mean']:>6.2f}")
    
    # 输出CSV文件
    if args.output_csv:
        df = pd.DataFrame(results)
        df.to_csv(args.output_csv, index=False, encoding='utf-8-sig')
        print(f"\n[OK] 结果已保存到: {args.output_csv}")
    else:
        # 如果安装了pandas，也显示DataFrame
        try:
            df = pd.DataFrame(results)
            print("\n【完整统计表】:")
            print(df.to_string(index=False))
        except:
            pass


if __name__ == "__main__":
    main()

"""
python plots/roi_statistics.py --input-folders /data/ykx/MSRS/test/ir  /data/ykx/MSRS/test/vi /home/ykx/ReCoNet/result/msrs/ori /home/ykx/ReCoNet/result/msrs/grad /home/ykx/ReCoNet/result/msrs/tcmoa /home/ykx/ReCoNet/result/msrs/ours --filename 00209D --roi '230, 140, 70, 160'
"""