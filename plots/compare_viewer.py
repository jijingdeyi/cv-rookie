#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse, cv2, numpy as np, math, json, sys, os
from collections import defaultdict

def imread_any(p: Path):
    # 兼容中文路径与非 ASCII：用 tofile/ fromfile + imdecode
    arr = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

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

def tile_images(images, labels, cols=0, pad=8, bg=18, sep=2, font_scale=0.6):
    valid = [im for im in images if im is not None]
    if not valid: return None
    th, tw = valid[0].shape[:2]
    normed = []
    for im in images:
        if im is None:
            im = np.full((th, tw, 3), 64, np.uint8)
            im = draw_label(im, "MISSING", font_scale)
        else:
            if im.shape[:2] != (th, tw):
                im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
        normed.append(im)
    normed = [draw_label(im, lbl, font_scale, pad//2) for im, lbl in zip(normed, labels)]
    n = len(normed)
    if cols <= 0 or cols >= n: rows, cols = 1, n
    else: rows = math.ceil(n / cols)

    cell_h, cell_w = th + 2*pad, tw + 2*pad
    H = rows*cell_h + (rows-1)*sep
    W = cols*cell_w + (cols-1)*sep
    canvas = np.full((H, W, 3), bg, np.uint8)

    for idx, im in enumerate(normed):
        r, c = divmod(idx, cols)
        y0 = r*(cell_h + sep); x0 = c*(cell_w + sep)
        canvas[y0+pad:y0+pad+th, x0+pad:x0+pad+tw] = im
    for r in range(1, rows):
        y = r*cell_h + (r-1)*sep; canvas[y:y+sep,:,:] = 128
    for c in range(1, cols):
        x = c*cell_w + (c-1)*sep; canvas[:,x:x+sep,:] = 128
    return canvas

def parse_ref(arg: str):
    # 形如 "IR:/path/to/ir_dir"
    if ":" not in arg:
        print(f"[Error] --ref 需要 '标签:目录' 形式，如 IR:/data/IR", file=sys.stderr)
        sys.exit(1)
    name, path = arg.split(":", 1)
    name = name.strip(); path = path.strip()
    p = Path(path)
    if not p.is_dir():
        print(f"[Error] 参考目录不存在: {p}", file=sys.stderr)
        sys.exit(1)
    return name, p

def build_stem_index(dir_path: Path, exts_priority):
    """
    扫描目录下所有文件，建立：
    - stems: set of stems（不含扩展名）
    - best_path_by_stem: dict[stem] = Path（按 exts_priority 选择最优扩展名）
    支持大小写扩展名（.PNG 等）。
    """
    # 允许的扩展名集合（小写）
    allowed = {e.lower() for e in exts_priority}
    # 记录每个 stem 下不同扩展名的文件
    pool = defaultdict(dict)  # stem -> {ext_lower: Path}
    for p in dir_path.iterdir():
        if not p.is_file(): 
            continue
        ext = p.suffix.lower()
        if ext in allowed:
            stem = p.stem  # 保留原样（大小写/前导零不变）
            # 同一 ext 多次出现时，保留第一个即可
            if ext not in pool[stem]:
                pool[stem][ext] = p

    # 按优先级挑选最佳路径
    best = {}
    for stem, mapping in pool.items():
        for e in exts_priority:
            e_low = e.lower()
            if e_low in mapping:
                best[stem] = mapping[e_low]
                break
        # 如果都不在优先列表（理论不会发生），择任意
        if stem not in best and mapping:
            best[stem] = next(iter(mapping.values()))
    stems = set(best.keys())
    return stems, best

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True,
                    help="方法结果根目录（包含各方法子目录）")
    ap.add_argument("--methods", nargs="*", default=None,
                    help="方法子目录显示顺序；留空=自动发现并按字母序")
    ap.add_argument("--ref", action="append", default=[],
                    help="新增参考列：格式 '标签:目录'，可重复两次，例如 "
                         "--ref 'IR:/data/IR' --ref 'VIS:/data/VIS'")
    ap.add_argument("--exts", nargs="*", default=[".png",".jpg",".jpeg",".bmp"],
                    help="允许的扩展名（含优先级，前者优先）；大小写不敏感")
    ap.add_argument("--out-dir", type=str, default="export")
    ap.add_argument("--cols", type=int, default=0)
    ap.add_argument("--font-scale", type=float, default=0.6)
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--sep", type=int, default=2)
    ap.add_argument("--bg-gray", type=int, default=18)
    ap.add_argument("--strict-intersection", action="store_true",
                    help="开启后仅导出在所有列（参考+方法）都存在的 stem；关闭则按并集并对缺图用 MISSING")
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        print(f"[Error] root 目录不存在：{root}", file=sys.stderr); sys.exit(1)
    # 参考列（最左侧）
    ref_pairs = [parse_ref(r) for r in args.ref]  # [(label, Path)]
    # 方法列（参考列之后）
    if args.methods:
        methods = [m for m in args.methods if (root/m).is_dir()]
    else:
        methods = sorted([d.name for d in root.iterdir() if d.is_dir()])
    if not methods:
        print(f"[Error] {root} 下未找到方法子目录", file=sys.stderr); sys.exit(1)

    # 为每一列（参考+方法）建立 stem 索引
    col_labels = []
    col_stems = []           # list[set_of_stems]
    col_best_path = []       # list[dict stem->Path]

    # 参考列
    for lbl, d in ref_pairs:
        stems, best = build_stem_index(d, args.exts)
        col_labels.append(lbl)
        col_stems.append(stems)
        col_best_path.append(best)

    # 方法列
    for m in methods:
        d = root/m
        stems, best = build_stem_index(d, args.exts)
        col_labels.append(m)
        col_stems.append(stems)
        col_best_path.append(best)

    # 聚合所有 stem（交集或并集）
    if not col_stems:
        print("[Error] 没有可用列", file=sys.stderr); sys.exit(1)
    if args.strict_intersection:
        names = sorted(set.intersection(*col_stems))
    else:
        names = sorted(set.union(*col_stems))
    if not names:
        print("[Error] 没有匹配的文件主干（stem）。请检查命名。", file=sys.stderr); sys.exit(1)

    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    panels_dir = out / "panels"; panels_dir.mkdir(exist_ok=True)
    manifest = []

    # 逐个 stem 导出拼图
    for i, stem in enumerate(names, 1):
        ims, labels = [], []
        for lbl, best_map in zip(col_labels, col_best_path):
            p = best_map.get(stem, None)
            im = imread_any(p) if p and p.exists() else None
            ims.append(im); labels.append(lbl)

        panel = tile_images(ims, labels, cols=args.cols, pad=args.pad,
                            bg=args.bg_gray, sep=args.sep, font_scale=args.font_scale)
        if panel is None:
            # 所有列都缺就跳过
            continue

        title = np.full((36, panel.shape[1], 3), 24, np.uint8)
        cv2.putText(title, f"{stem}  [{i}/{len(names)}]", (10,24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,255), 1, cv2.LINE_AA)
        out_img = np.vstack([title, panel])

        out_path = panels_dir / f"{stem}_cmp.png"
        cv2.imencode(".png", out_img)[1].tofile(str(out_path))
        manifest.append(out_path.name)

    if not manifest:
        print("[Warn] 没有导出任何面板，请检查文件名主干是否对齐。")
        return

    # 生成 index.html
    html = f"""<!doctype html>
<html><head><meta charset="utf-8">
<title>Comparison Panels</title>
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
const files = {json.dumps(manifest)};
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
    (out/"index.html").write_text(html, encoding="utf-8")
    print(f"[OK] Exported {len(manifest)} panels to {panels_dir}")
    print(f"[OPEN] {out/'index.html'}  （浏览器打开，←/→ 切换；最左侧为参考列）")

if __name__ == "__main__":
    main()


"""

python plots/compare_viewer.py \
  --root /home/ykx/ReCoNet/result/msrs \
  --ref "IR:/home/ykx/data/MSRS/IR" \
  --ref "VIS:/home/ykx/data/MSRS/VIS" \
  --methods ori grad grad_aligned \
  --exts .png .jpg .jpeg .bmp \
  --out-dir cmp_export \
  --strict-intersection \
  --cols 0

python plots/compare_viewer.py --root /home/ykx/ReCoNet/result/msrs --ref IR:/data/ykx/MSRS/test/ir --ref VI:/data/ykx/MSRS/test/vi --methods ori grad 
tcmoa ours --out-dir temp --cols 0

"""