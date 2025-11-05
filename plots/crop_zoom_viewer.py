#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
对已生成的对比图进行区域放大处理
从每张对比图的相同区域裁剪并放大，重新拼接成新的对比图
"""
from pathlib import Path
import argparse, cv2, numpy as np, sys
from collections import defaultdict


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


def find_subimages(panel_img, title_h=36, pad=8, sep=2, min_cell_size=50):
    """
    从对比图中识别每个子图像的位置
    返回: list of (y0, x0, h, w) 每个子图的位置
    """
    h, w = panel_img.shape[:2]
    panel_area = panel_img[title_h:, :]  # 去掉标题
    
    # 检测水平和垂直分隔线（sep区域通常是128灰度）
    gray = cv2.cvtColor(panel_area, cv2.COLOR_BGR2GRAY)
    
    # 找垂直分隔线
    v_lines = []
    for x in range(0, w, max(1, w // 200)):
        col = gray[:, x]
        if np.std(col) < 10 and np.mean(col) > 100:  # 分隔线特征
            v_lines.append(x)
    
    # 找水平分隔线
    h_lines = []
    for y in range(0, h - title_h, max(1, (h - title_h) // 200)):
        row = gray[y, :]
        if np.std(row) < 10 and np.mean(row) > 100:
            h_lines.append(y)
    
    # 简化：如果检测不到分隔线，尝试基于网格布局估计
    # 假设至少有一个子图，从左上角开始尝试找到第一个子图的边界
    cells = []
    if not v_lines and not h_lines:
        # 无法检测，尝试启发式方法：假设是单行布局
        # 估算每个cell的宽度（基于pad和内容）
        estimated_w = w // 10  # 粗略估计
        x = pad
        while x < w - pad:
            if x + estimated_w < w:
                cells.append((title_h, x, h - title_h, min(estimated_w, w - x)))
            x += estimated_w + sep
        if cells:
            return cells[:1]  # 至少返回第一个
    
    # 基于检测到的分隔线构建网格
    if v_lines or h_lines:
        v_bounds = [0] + sorted(set(v_lines)) + [w]
        h_bounds = [0] + sorted(set(h_lines)) + [h - title_h]
        
        for i in range(len(h_bounds) - 1):
            for j in range(len(v_bounds) - 1):
                y0 = title_h + h_bounds[i]
                x0 = v_bounds[j]
                h_cell = h_bounds[i+1] - h_bounds[i] - sep
                w_cell = v_bounds[j+1] - v_bounds[j] - sep
                if h_cell > min_cell_size and w_cell > min_cell_size:
                    # 去掉pad
                    y0 += pad
                    x0 += pad
                    h_cell -= 2 * pad
                    w_cell -= 2 * pad
                    if h_cell > 0 and w_cell > 0:
                        cells.append((y0, x0, h_cell, w_cell))
    
    return cells if cells else [(title_h + pad, pad, h - title_h - 2*pad, w - 2*pad)]


def extract_roi_from_image(img, roi_params, zoom_factor=2.0):
    """
    从单张图像中提取ROI区域并放大
    返回: 放大后的ROI (numpy array) 或 None
    """
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
    
    # 放大
    zoom_w = int(roi_w * zoom_factor)
    zoom_h = int(roi_h * zoom_factor)
    zoomed = cv2.resize(roi, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)
    
    return zoomed


def extract_and_zoom_roi(panel_img, roi_params, title_h=36, pad=8, sep=2, zoom_factor=2.0):
    """
    从对比图中提取每个子图的ROI区域并放大
    返回: list of zoomed ROIs (numpy arrays)
    """
    cells = find_subimages(panel_img, title_h, pad, sep)
    if not cells or len(cells) == 0:
        return []
    
    # 使用第一个cell的尺寸来标准化ROI（假设所有cell大小相同）
    first_cell = cells[0]
    if len(first_cell) < 4:
        return []
    _, _, ref_h, ref_w = first_cell
    
    # 计算ROI的实际坐标（相对于子图）
    roi_x, roi_y, roi_w, roi_h = get_roi_coords(roi_params, ref_h, ref_w)
    
    zoomed_rois = []
    for y0, x0, cell_h, cell_w in cells:
        # 计算ROI在这个cell中的实际位置
        scale_h = cell_h / ref_h
        scale_w = cell_w / ref_w
        
        actual_x = int(x0 + roi_x * scale_w)
        actual_y = int(y0 + roi_y * scale_h)
        actual_w = int(roi_w * scale_w)
        actual_h = int(roi_h * scale_h)
        
        # 确保不越界
        actual_x = max(x0, min(actual_x, x0 + cell_w - 1))
        actual_y = max(y0, min(actual_y, y0 + cell_h - 1))
        actual_w = min(actual_w, x0 + cell_w - actual_x)
        actual_h = min(actual_h, y0 + cell_h - actual_y)
        
        if actual_w > 0 and actual_h > 0:
            roi = panel_img[actual_y:actual_y+actual_h, actual_x:actual_x+actual_w]
            # 放大
            zoom_w = int(actual_w * zoom_factor)
            zoom_h = int(actual_h * zoom_factor)
            zoomed = cv2.resize(roi, (zoom_w, zoom_h), interpolation=cv2.INTER_CUBIC)
            zoomed_rois.append(zoomed)
        else:
            zoomed_rois.append(None)
    
    return zoomed_rois


def tile_zoomed_rois(zoomed_rois, labels=None, cols=0, pad=8, bg=18, sep=2, font_scale=0.6, keep_aspect=True):
    """将放大的ROI区域拼接成新的对比图"""
    valid = [roi for roi in zoomed_rois if roi is not None]
    if not valid:
        return None
    
    normed = []
    if keep_aspect:
        # 保持原始宽高比，只统一高度
        max_h = max(r.shape[0] for r in valid)
        for roi in zoomed_rois:
            if roi is None:
                # 用占位图，宽度使用平均宽度
                avg_w = int(np.mean([r.shape[1] for r in valid]))
                roi = np.full((max_h, avg_w, 3), 64, np.uint8)
            else:
                orig_h, orig_w = roi.shape[:2]
                # 按比例缩放高度到max_h，保持宽高比
                new_w = int(orig_w * max_h / orig_h) if orig_h > 0 else orig_w
                roi = cv2.resize(roi, (new_w, max_h), interpolation=cv2.INTER_AREA)
            normed.append(roi)
    else:
        # 统一到最大尺寸（原来的行为）
        max_h = max(r.shape[0] for r in valid)
        max_w = max(r.shape[1] for r in valid)
        for roi in zoomed_rois:
            if roi is None:
                roi = np.full((max_h, max_w, 3), 64, np.uint8)
            elif roi.shape[:2] != (max_h, max_w):
                roi = cv2.resize(roi, (max_w, max_h), interpolation=cv2.INTER_AREA)
            normed.append(roi)
    
    # 添加标签（如果有）
    if labels:
        def draw_label(img, text, font_scale=0.6, pad=6):
            h, w = img.shape[:2]
            overlay = img.copy()
            bar_h = int(28 * font_scale + 2*pad)
            cv2.rectangle(overlay, (0,0), (w, bar_h), (0,0,0), -1)
            img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
            thickness = max(1, int(1.2 * font_scale))
            cv2.putText(img, text, (pad, pad + int(20*font_scale)), cv2.FONT_HERSHEY_SIMPLEX,
                        font_scale, (255,255,255), thickness, cv2.LINE_AA)
            return img
        normed = [draw_label(im, lbl, font_scale, pad//2) for im, lbl in zip(normed, labels)]
    
    n = len(normed)
    if cols <= 0 or cols >= n:
        rows, cols = 1, n
    else:
        import math
        rows = math.ceil(n / cols)
    
    if keep_aspect:
        # 当保持宽高比时，每个单元格宽度可能不同，使用最大宽度
        max_w = max(im.shape[1] for im in normed)
        cell_h = max_h + 2*pad
        # 为每列计算最大宽度
        cell_widths = []
        for c in range(cols):
            col_imgs = [normed[i] for i in range(c, n, cols)]
            max_col_w = max(img.shape[1] for img in col_imgs) if col_imgs else max_w
            cell_widths.append(max_col_w + 2*pad)
        
        # 计算总宽度
        W = sum(cell_widths) + (cols - 1) * sep
        H = rows * cell_h + (rows - 1) * sep
        canvas = np.full((H, W, 3), bg, np.uint8)
        
        for idx, roi in enumerate(normed):
            r, c = divmod(idx, cols)
            y0 = r * (cell_h + sep)
            x0 = sum(cell_widths[:c]) + c * sep
            h_roi, w_roi = roi.shape[:2]
            canvas[y0+pad:y0+pad+h_roi, x0+pad:x0+pad+w_roi] = roi
    else:
        cell_h, cell_w = max_h + 2*pad, max_w + 2*pad
        H = rows * cell_h + (rows - 1) * sep
        W = cols * cell_w + (cols - 1) * sep
        canvas = np.full((H, W, 3), bg, np.uint8)
        
        for idx, roi in enumerate(normed):
            r, c = divmod(idx, cols)
            y0 = r * (cell_h + sep)
            x0 = c * (cell_w + sep)
            canvas[y0+pad:y0+pad+max_h, x0+pad:x0+pad+max_w] = roi
    
    # 绘制分隔线
    for r in range(1, rows):
        y = r * cell_h + (r - 1) * sep
        canvas[y:y+sep, :, :] = 128
    
    if keep_aspect:
        # 当保持宽高比时，垂直分隔线需要按列宽计算
        x_pos = 0
        for c in range(1, cols):
            x_pos += cell_widths[c-1] + sep
            canvas[:, x_pos:x_pos+sep, :] = 128
    else:
        # 统一尺寸时，使用固定单元格宽度
        for c in range(1, cols):
            x = c * cell_w + (c - 1) * sep
            canvas[:, x:x+sep, :] = 128
    
    return canvas


def main():
    ap = argparse.ArgumentParser(
        description="对已生成的对比图进行区域放大处理",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例用法:
  # 从多个文件夹读取同名图像并拼接（推荐）
  python plots/crop_zoom_viewer.py \\
    --input-folders method1 method2 method3 method4 method5 method6 \\
    --method-names "Method1" "Method2" "Method3" "Method4" "Method5" "Method6" \\
    --filename 00119D \\
    --roi "0.3,0.3,0.4,0.4" --zoom 3.0 --out-dir output
  
  # 处理单张图片
  python plots/crop_zoom_viewer.py --input-file temp/panels/00119D_cmp.png --roi "0.3,0.3,0.4,0.4" --zoom 3.0
  
  # 处理单张图片并指定输出文件名
  python plots/crop_zoom_viewer.py --input-file temp/panels/00119D_cmp.png --roi "0.5,0.5,0.3" --zoom 2.5 --output-file output.png
  
  # 处理整个目录（使用相对坐标）
  python plots/crop_zoom_viewer.py --input-dir temp/panels --roi "0.3,0.3,0.4,0.4" --zoom 3.0
  
  # 使用绝对坐标（像素）
  python plots/crop_zoom_viewer.py --input-dir temp/panels --roi "100,100,200,200" --zoom 2.0

ROI格式说明:
  - 相对坐标: "rx,ry,rw,rh" (0-1范围，如 "0.3,0.3,0.4,0.4")
  - 绝对坐标: "x,y,w,h" (像素值，如 "100,100,200,200")
  - 中心点: "cx,cy,size" (0-1范围，如 "0.5,0.5,0.3")
        """
    )
    ap.add_argument("--input-folders", nargs="+", default=None,
                    help="多个输入文件夹路径（每个文件夹代表一种方法），与 --input-file/--input-dir 互斥")
    ap.add_argument("--method-names", nargs="+", default=None,
                    help="每个文件夹对应的方法名称/标签（数量需与--input-folders一致，留空则使用文件夹名）")
    ap.add_argument("--filename", type=str, default=None,
                    help="要处理的文件名（不含扩展名，用于--input-folders模式，必填）")
    ap.add_argument("--input-dir", type=str, default=None,
                    help="输入的对比图目录（panels目录），与 --input-file/--input-folders 互斥")
    ap.add_argument("--input-file", type=str, default=None,
                    help="输入的单张对比图文件，与 --input-dir/--input-folders 互斥")
    ap.add_argument("--exts", nargs="*", default=[".png", ".jpg", ".jpeg", ".bmp"],
                    help="允许的扩展名（含优先级，仅用于--input-folders模式）")
    ap.add_argument("--roi", type=str, required=True,
                    help="ROI区域，格式: 'x,y,w,h' 或 'rx,ry,rw,rh' 或 'cx,cy,size'")
    ap.add_argument("--zoom", type=float, default=2.0,
                    help="放大倍数 (默认: 2.0)")
    ap.add_argument("--out-dir", type=str, default=None,
                    help="输出目录（单文件模式：默认同目录；目录模式：默认input_dir的父目录/crop_zoom）")
    ap.add_argument("--output-file", type=str, default=None,
                    help="单文件模式下的输出文件名（默认：原文件名_zoom.png）")
    ap.add_argument("--title-h", type=int, default=36,
                    help="对比图标题高度（默认: 36）")
    ap.add_argument("--pad", type=int, default=8,
                    help="子图内边距（默认: 8）")
    ap.add_argument("--sep", type=int, default=2,
                    help="子图间隔（默认: 2）")
    ap.add_argument("--cols", type=int, default=0,
                    help="输出布局列数（0=自动）")
    ap.add_argument("--font-scale", type=float, default=0.6,
                    help="标签字体大小（默认: 0.6）")
    ap.add_argument("--bg-gray", type=int, default=18,
                    help="背景灰度值（默认: 18）")
    ap.add_argument("--keep-aspect", action="store_true", default=True,
                    help="保持ROI原始宽高比，只统一高度（默认: True）")
    ap.add_argument("--stretch", action="store_true", default=False,
                    help="将所有ROI拉伸到最大尺寸（会改变宽高比）")
    args = ap.parse_args()
    
    # 检查输入参数
    input_modes = sum([bool(args.input_file), bool(args.input_dir), bool(args.input_folders)])
    if input_modes != 1:
        print("[Error] 必须且只能指定 --input-file、--input-dir 或 --input-folders 之一", file=sys.stderr)
        sys.exit(1)
    
    # 如果是多文件夹模式，检查参数
    if args.input_folders:
        if not args.filename:
            print("[Error] --input-folders 模式需要指定 --filename（文件名不含扩展名）", file=sys.stderr)
            sys.exit(1)
        if args.method_names and len(args.method_names) != len(args.input_folders):
            print(f"[Error] --method-names 数量 ({len(args.method_names)}) 与 --input-folders 数量 ({len(args.input_folders)}) 不一致", file=sys.stderr)
            sys.exit(1)
    
    # 解析ROI
    try:
        roi_params = parse_roi(args.roi)
    except ValueError as e:
        print(f"[Error] {e}", file=sys.stderr)
        sys.exit(1)
    
    # 多文件夹模式
    if args.input_folders:
        # 验证所有文件夹
        folder_paths = [Path(f) for f in args.input_folders]
        for fp in folder_paths:
            if not fp.is_dir():
                print(f"[Error] 文件夹不存在: {fp}", file=sys.stderr)
                sys.exit(1)
        
        # 确定方法名称
        if args.method_names:
            method_names = args.method_names
        else:
            method_names = [fp.name for fp in folder_paths]
        
        # 确定输出目录和文件名
        if args.out_dir:
            out_dir = Path(args.out_dir)
        else:
            # 使用第一个文件夹的父目录
            out_dir = folder_paths[0].parent / "crop_zoom"
        out_dir.mkdir(parents=True, exist_ok=True)
        
        # 从每个文件夹中查找指定的文件
        stem = args.filename
        allowed_exts = {e.lower() for e in args.exts}
        images = []
        labels = method_names
        found_files = []
        
        for folder_path in folder_paths:
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
            
            if img_path:
                img = imread_any(img_path)
                images.append(img)
                found_files.append(img_path.name)
            else:
                images.append(None)
                found_files.append(None)
                print(f"[Warn] 在 {folder_path} 中未找到文件: {stem}")
        
        if not any(img is not None for img in images):
            print(f"[Error] 在所有文件夹中都未找到文件: {stem}", file=sys.stderr)
            sys.exit(1)
        
        print(f"[Info] 处理文件: {stem}")
        print(f"[Info] 方法数量: {len(method_names)}")
        print(f"[Info] ROI: {roi_params}")
        print(f"[Info] 放大倍数: {args.zoom}x")
        
        # 提取ROI并放大
        zoomed_rois = []
        for idx, img in enumerate(images):
            if img is not None:
                h, w = img.shape[:2]
                roi_x, roi_y, roi_w, roi_h = get_roi_coords(roi_params, h, w)
                print(f"[Debug] 图像 {idx} ({method_names[idx]}): 原图尺寸={w}x{h}, ROI=({roi_x},{roi_y},{roi_w},{roi_h}), 放大后={int(roi_w*args.zoom)}x{int(roi_h*args.zoom)}")
            zoomed = extract_roi_from_image(img, roi_params, args.zoom)
            zoomed_rois.append(zoomed)
        
        # 拼接
        keep_aspect = not args.stretch  # 如果指定了--stretch，则不保持宽高比
        panel = tile_zoomed_rois(zoomed_rois, labels=labels, cols=args.cols,
                                pad=args.pad, bg=args.bg_gray, sep=args.sep,
                                font_scale=args.font_scale, keep_aspect=keep_aspect)
        
        if panel is None:
            print("[Error] 无法生成面板")
            sys.exit(1)
        
        # 添加标题
        title = np.full((args.title_h, panel.shape[1], 3), 24, np.uint8)
        cv2.putText(title, f"{stem} (Zoom {args.zoom}x)",
                   (10, args.title_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 255, 255), 1, cv2.LINE_AA)
        out_img = np.vstack([title, panel])
        
        # 确定输出文件名
        if args.output_file:
            out_path = Path(args.output_file)
        else:
            out_path = out_dir / f"{stem}_zoom.png"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存
        cv2.imencode(".png", out_img)[1].tofile(str(out_path))
        print(f"[OK] 已保存: {out_path}")
        return
    
    # 单文件模式
    if args.input_file:
        input_file = Path(args.input_file)
        if not input_file.is_file():
            print(f"[Error] 输入文件不存在: {input_file}", file=sys.stderr)
            sys.exit(1)
        
        panel_files = [input_file]
        is_single_file = True
        
        # 确定输出文件
        if args.output_file:
            output_file = Path(args.output_file)
        else:
            output_file = input_file.parent / f"{input_file.stem}_zoom.png"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        # 单文件模式不需要生成HTML，跳过输出目录设置
        out_dir = None
        panels_out = None
    else:
        # 目录模式
        input_dir = Path(args.input_dir)
        if not input_dir.is_dir():
            print(f"[Error] 输入目录不存在: {input_dir}", file=sys.stderr)
            sys.exit(1)
        
        panel_files = sorted([f for f in input_dir.glob("*.png") if f.name.endswith("_cmp.png")])
        if not panel_files:
            print(f"[Error] 在 {input_dir} 中未找到对比图（*_cmp.png）", file=sys.stderr)
            sys.exit(1)
        
        is_single_file = False
        
        # 确定输出目录
        if args.out_dir:
            out_dir = Path(args.out_dir)
        else:
            out_dir = input_dir.parent / "crop_zoom"
        out_dir.mkdir(parents=True, exist_ok=True)
        panels_out = out_dir / "panels"
        panels_out.mkdir(exist_ok=True)
    
    print(f"[Info] 找到 {len(panel_files)} 张对比图")
    print(f"[Info] ROI: {roi_params}")
    print(f"[Info] 放大倍数: {args.zoom}x")
    
    manifest = []
    for i, panel_file in enumerate(panel_files, 1):
        stem = panel_file.stem.replace("_cmp", "")
        panel_img = imread_any(panel_file)
        if panel_img is None:
            print(f"[Warn] 无法读取: {panel_file}")
            continue
        
        # 提取并放大ROI
        zoomed_rois = extract_and_zoom_roi(
            panel_img, roi_params, 
            title_h=args.title_h, pad=args.pad, sep=args.sep,
            zoom_factor=args.zoom
        )
        
        if not zoomed_rois:
            print(f"[Warn] 无法从 {panel_file} 提取ROI")
            continue
        
        # 拼接成新图
        new_panel = tile_zoomed_rois(
            zoomed_rois, cols=args.cols, pad=args.pad,
            bg=args.bg_gray, sep=args.sep, font_scale=args.font_scale
        )
        
        if new_panel is None:
            continue
        
        # 添加标题
        title = np.full((args.title_h, new_panel.shape[1], 3), 24, np.uint8)
        cv2.putText(title, f"{stem} (Zoom {args.zoom}x)  [{i}/{len(panel_files)}]",
                   (10, args.title_h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                   (255, 255, 255), 1, cv2.LINE_AA)
        out_img = np.vstack([title, new_panel])
        
        # 保存
        if is_single_file:
            out_path = output_file
            cv2.imencode(".png", out_img)[1].tofile(str(out_path))
            print(f"[OK] 已保存: {out_path}")
            return
        else:
            assert panels_out is not None, "目录模式下panels_out不应为None"
            out_path = panels_out / f"{stem}_zoom.png"
            cv2.imencode(".png", out_img)[1].tofile(str(out_path))
            manifest.append(out_path.name)
            print(f"[OK] {i}/{len(panel_files)}: {out_path.name}")
    
    if not manifest:
        print("[Warn] 没有生成任何图片")
        return
    
    # 生成index.html（仅目录模式，此时out_dir和panels_out一定不为None）
    assert out_dir is not None and panels_out is not None, "目录模式下out_dir和panels_out不应为None"
    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Crop Zoom Panels</title>
<style>
body {{ background:#111; color:#eee; font-family:system-ui,Arial; margin:0; }}
#wrap {{ display:flex; flex-direction:column; align-items:center; }}
#bar {{ width:100%; padding:10px 16px; box-sizing:border-box; background:#1a1a1a; position:sticky; top:0; }}
img {{ max-width:98vw; height:auto; margin:10px auto 40px; display:block; }}
button {{ margin-right:8px; }}
</style>
</head>
<body>
<div id="bar">
  <button onclick="prev()">← Prev</button>
  <button onclick="next()">Next →</button>
  <span id="info"></span>
</div>
<div id="wrap">
  <img id="panel" src="panels/{manifest[0]}">
</div>
<script>
const files = {repr(manifest)};
let idx = 0;
function update(){{
  const img = document.getElementById('panel');
  img.src = "panels/" + files[idx];
  document.getElementById('info').innerText = files[idx] + "  [" + (idx+1) + "/" + files.length + "]";
}}
function prev(){{ idx = (idx - 1 + files.length) % files.length; update(); }}
function next(){{ idx = (idx + 1) % files.length; update(); }}
document.addEventListener('keydown', e => {{
  if (e.key === 'ArrowLeft') prev();
  if (e.key === 'ArrowRight') next();
}});
update();
</script>
</body></html>"""
    (out_dir / "index.html").write_text(html, encoding="utf-8")
    print(f"\n[OK] 生成 {len(manifest)} 张放大对比图到 {panels_out}")
    print(f"[OPEN] {out_dir / 'index.html'}  （浏览器打开查看）")


if __name__ == "__main__":
    main()

"""
$ python plots/crop_zoom_viewer.py --input-folders /data/ykx/LLVIP/test/ir /data/ykx/LLVIP/test/vi /data/ykx/result/llvip/ori /data/ykx/result/llvip/grad /data/ykx/result/llvip/tcmoa /data/ykx/result/llvip/v1 --filename 190090 --roi '580,0,60,60' --zoom 10 --out-dir output
"""
