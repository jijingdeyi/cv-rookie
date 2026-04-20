#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from pathlib import Path
import argparse, cv2, numpy as np, math, json, sys, os
from collections import defaultdict
import socket, subprocess, time

# ================= 常用参数配置区（建议只改这里） =================
# 说明：
# 1) 这些值会作为 argparse 的默认值；命令行传参仍然会覆盖它们。
# 2) 你常改的参数都集中在这里，避免在 main() 里到处翻。
USER_COMMON_DEFAULTS = {
    "root": "/data/ykx/sota/OpIVF",
    "methods": [
        "didfuse", "rfnnest", "mfeif", "sdnet", "piafusion",
        "reconet", "swinfusion", "tardal", "cddfuse", "lrrnet",
        "metafusion", "segmif", "emma", "sage", "gifnet",
    ],
    "my_method": "ours:/home/ykx/ca-fusion-loss/results/OpIVF/ours",
    "ref": ["IR:/data/ykx/OpIVF/test/ir", "VIS:/data/ykx/OpIVF/test/vi"],
    "cols": 5,
    "only_stems": ['27'],  # None / 不传 = 导出全部；勿用 [""] 会误筛成空
    "zoom_pos": "br,tr",
    "zoom_factors": "1.8,2",  # 例如: "1.8,3.2"
    # 剖面线：单条 "x0,y0,x1,y1" 或多条用英文分号分隔；或 --pick-profile-line-web 写入 profile_line.json
    "profile_line": "",
    # 剖面纵轴：单段 "ymin,ymax"；多线与剖面线一一对应用英文分号，如 0,255;40,220；留空则默认约 (-5,260)
    "profile_ylim": "150, 256",
    # 亮度图 ROI：与剖面线相同坐标系 x,y,w,h（对齐后单格像素）；留空则不导出 brightness_maps/
    "brightness_map_roi": "",
    # 亮度图伪彩色值域 "vmin,vmax"（如 0,255）；留空则用 0,255，也可写在 brightness_map_roi.json
    "brightness_map_vlim": "150,255",
}
# ===============================================================

ZOOM_COLOR_PALETTE = [
    (0, 255, 255),   # yellow
    (0, 255, 0),     # green
    (255, 0, 255),   # magenta
    (255, 255, 0),   # cyan
    (0, 128, 255),   # orange-ish
    (255, 128, 0),   # blue-ish
]

def imread_any(p: Path):
    # 兼容中文路径与非 ASCII：用 tofile/ fromfile + imdecode
    arr = np.fromfile(str(p), dtype=np.uint8)
    return cv2.imdecode(arr, cv2.IMREAD_COLOR)

def draw_label(img, text, font_scale=0.6, pad=6, label_scale_mult=1.25, label_color=(0, 255, 255),
               ref_short_side=512.0):
    h, w = img.shape[:2]
    short = float(min(h, w))
    ref = max(1.0, float(ref_short_side))
    # 字号随图像短边成比例：在 ref_short_side 下等价于 font_scale * label_scale_mult
    base = float(font_scale) * float(label_scale_mult)
    label_scale = base * (short / ref)
    label_scale = max(0.15, min(label_scale, 16.0))
    overlay = img.copy()
    bar_h = int(max(22.0, 0.065 * short) + 2 * pad)
    bar_h = min(bar_h, h // 2)
    cv2.rectangle(overlay, (0,0), (w, bar_h), (0,0,0), -1)
    img = cv2.addWeighted(overlay, 0.35, img, 0.65, 0)
    thickness = max(1, int(2.2 * label_scale))
    thickness = min(thickness, max(1, bar_h // 4))
    x = pad
    y = min(h - 2, pad + int(0.75 * bar_h))
    # 黑色描边 + 黄色主文字，提高可读性
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                label_scale, (0, 0, 0), thickness + 2, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                label_scale, label_color, thickness, cv2.LINE_AA)
    return img

def parse_brightness_map_roi(arg: str):
    """单矩形 ROI：x,y,w,h（与剖面线/放大 ROI 相同：对齐后首图坐标系）。空则 None。"""
    s = str(arg).strip()
    if not s:
        return None
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 4:
        print(f"[Error] --brightness-map-roi 需要 'x,y,w,h' 形式，例如 120,80,96,96", file=sys.stderr)
        sys.exit(1)
    try:
        x, y, w, h = [int(v) for v in parts]
    except ValueError:
        print(f"[Error] --brightness-map-roi 必须为整数: {arg}", file=sys.stderr)
        sys.exit(1)
    if w <= 0 or h <= 0:
        print(f"[Error] --brightness-map-roi 中 w,h 必须 > 0: {arg}", file=sys.stderr)
        sys.exit(1)
    return x, y, w, h


def parse_brightness_map_vlim(arg: str):
    """亮度图 imshow 值域 min,max；空则 None（由配置文件或默认 0,255）。"""
    s = str(arg).strip()
    if not s:
        return None
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 2:
        print(f"[Error] --brightness-map-vlim 需要 vmin,vmax，例如 0,255 或 40,220", file=sys.stderr)
        sys.exit(1)
    try:
        a, b = float(parts[0]), float(parts[1])
    except ValueError:
        print(f"[Error] --brightness-map-vlim 必须为数字: {arg}", file=sys.stderr)
        sys.exit(1)
    if a >= b:
        print(f"[Error] --brightness-map-vlim 要求 vmin < vmax，得到 ({a}, {b})", file=sys.stderr)
        sys.exit(1)
    return a, b


def load_brightness_map_vlim_config(config_path: Path):
    """从 brightness_map_roi.json 读取 brightness_map_vlim: [vmin, vmax]。"""
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    raw = data.get("brightness_map_vlim")
    if not (isinstance(raw, list) and len(raw) == 2):
        return None
    try:
        a, b = float(raw[0]), float(raw[1])
    except (TypeError, ValueError):
        return None
    if a >= b:
        return None
    return a, b


def load_brightness_map_roi_config(config_path: Path):
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    roi = data.get("brightness_map_roi")
    if not (isinstance(roi, list) and len(roi) == 4):
        return None
    try:
        x, y, w, h = [int(v) for v in roi]
    except (TypeError, ValueError):
        return None
    if w <= 0 or h <= 0:
        return None
    return x, y, w, h


def parse_zoom_roi(arg: str):
    s = str(arg).strip()
    if not s:
        return None
    parts = [x.strip() for x in s.split(",")]
    if len(parts) != 4:
        print(f"[Error] --zoom-roi 需要 'x,y,w,h' 形式，例如 120,80,64,64", file=sys.stderr)
        sys.exit(1)
    try:
        x, y, w, h = [int(v) for v in parts]
    except ValueError:
        print(f"[Error] --zoom-roi 必须为整数: {arg}", file=sys.stderr)
        sys.exit(1)
    if w <= 0 or h <= 0:
        print(f"[Error] --zoom-roi 中 w,h 必须 > 0: {arg}", file=sys.stderr)
        sys.exit(1)
    return x, y, w, h

def parse_zoom_rois(arg: str):
    s = str(arg).strip()
    if not s:
        return []
    chunks = [c.strip() for c in s.split(";") if c.strip()]
    rois = [parse_zoom_roi(c) for c in chunks]
    return rois

def parse_zoom_factors(arg: str):
    s = str(arg).strip()
    if not s:
        return []
    out = []
    for p in [x.strip() for x in s.split(",") if x.strip()]:
        try:
            v = float(p)
        except ValueError:
            print(f"[Error] --zoom-factors 含非法浮点数: {p}", file=sys.stderr)
            sys.exit(1)
        if v <= 0:
            print(f"[Error] --zoom-factors 每项必须 > 0: {p}", file=sys.stderr)
            sys.exit(1)
        out.append(v)
    return out

def parse_zoom_positions(arg: str):
    s = str(arg).strip()
    if not s:
        return []
    valid = {"tl", "tr", "bl", "br"}
    out = []
    for p in [x.strip().lower() for x in s.split(",") if x.strip()]:
        if p not in valid:
            print(f"[Error] --zoom-poses 含非法位置: {p}，必须是 tl/tr/bl/br", file=sys.stderr)
            sys.exit(1)
        out.append(p)
    return out

def build_zoom_positions(base_pos: str, n: int):
    if n <= 0:
        return []
    orders = {
        "tl": ["tl", "tr", "bl", "br"],
        "tr": ["tr", "tl", "br", "bl"],
        "bl": ["bl", "br", "tl", "tr"],
        "br": ["br", "bl", "tr", "tl"],
    }
    seq = orders.get(base_pos, ["br", "bl", "tr", "tl"])
    return [seq[i % len(seq)] for i in range(n)]

def build_zoom_items(zoom_rois, base_pos="br", base_factor=2.0, factors=None, poses=None):
    if not zoom_rois:
        return []
    factors = factors or []
    poses = poses or []
    auto_pos = build_zoom_positions(base_pos, len(zoom_rois))
    items = []
    for i, roi in enumerate(zoom_rois):
        pos = poses[i] if i < len(poses) else auto_pos[i]
        factor = factors[i] if i < len(factors) else float(base_factor)
        items.append({"roi": roi, "pos": pos, "factor": factor})
    return items

def pick_zoom_color(idx: int, total: int, single_color):
    if total <= 1:
        return single_color
    return ZOOM_COLOR_PALETTE[idx % len(ZOOM_COLOR_PALETTE)]

def pick_zoom_roi_interactive(names, col_best_path, pick_stem=""):
    if not names:
        print("[Error] 没有可用于交互选框的 stem", file=sys.stderr)
        sys.exit(1)

    candidate_stems = []
    if pick_stem and pick_stem in names:
        candidate_stems.append(pick_stem)
    candidate_stems.extend([n for n in names if n != pick_stem])

    sample_img, sample_stem = None, None
    for stem in candidate_stems:
        for best_map in col_best_path:
            p = best_map.get(stem, None)
            if not p or (not p.exists()):
                continue
            im = imread_any(p)
            if im is not None:
                sample_img = im
                sample_stem = stem
                break
        if sample_img is not None:
            break

    if sample_img is None:
        print("[Error] 没有可用于交互选框的有效图片", file=sys.stderr)
        sys.exit(1)

    win = "Pick ROI (drag box, ENTER/SPACE confirm, C cancel)"
    preview = sample_img.copy()
    cv2.putText(preview, f"stem: {sample_stem}", (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2, cv2.LINE_AA)
    try:
        x, y, w, h = cv2.selectROI(win, preview, showCrosshair=True, fromCenter=False)
        cv2.destroyWindow(win)
    except cv2.error as e:
        print(f"[Error] 无法打开交互选框窗口（可能是无图形界面环境）: {e}", file=sys.stderr)
        sys.exit(1)

    x, y, w, h = int(x), int(y), int(w), int(h)
    if w <= 0 or h <= 0:
        print("[Error] 未选择有效 ROI（宽高为 0）", file=sys.stderr)
        sys.exit(1)
    print(f"[INFO] 交互选框完成：--zoom-roi {x},{y},{w},{h} （sample stem: {sample_stem}）")
    return (x, y, w, h)

def pick_sample_image(names, col_best_path, pick_stem=""):
    if not names:
        return None, None, None
    candidate_stems = []
    if pick_stem and pick_stem in names:
        candidate_stems.append(pick_stem)
    candidate_stems.extend([n for n in names if n != pick_stem])
    for stem in candidate_stems:
        for best_map in col_best_path:
            p = best_map.get(stem, None)
            if not p or (not p.exists()):
                continue
            im = imread_any(p)
            if im is not None:
                return stem, p, im
    return None, None, None

def write_web_roi_picker(sample_stem, sample_path: Path, sample_img, out_dir: Path, default_pos="br", default_factor=2.0, default_border=2):
    out_dir.mkdir(parents=True, exist_ok=True)
    picker_path = out_dir / "roi_picker.html"
    sample_web_name = "_roi_picker_sample.png"
    sample_web_path = out_dir / sample_web_name
    if sample_img is not None:
        ok, enc = cv2.imencode(".png", sample_img)
        if ok:
            enc.tofile(str(sample_web_path))
    html = """<!doctype html>
<html><head><meta charset="utf-8">
<title>ROI Picker</title>
<style>
body { background:#111; color:#eee; font-family:system-ui,Arial; margin:0; }
#wrap { max-width:1100px; margin:20px auto; padding:0 16px 24px; }
#hint { color:#bbb; margin:10px 0 14px; }
#imgBox { position:relative; display:inline-block; border:1px solid #444; }
#img { max-width:95vw; max-height:76vh; display:block; user-select:none; -webkit-user-drag:none; }
#sel { position:absolute; border:2px solid #00ffff; background:rgba(0,255,255,0.16); display:none; pointer-events:none; }
#bar { margin-top:14px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
input { width:260px; padding:6px 8px; background:#1d1d1d; border:1px solid #555; color:#fff; }
button { padding:6px 10px; cursor:pointer; }
</style>
</head><body>
<div id="wrap">
  <h3>ROI Picker (sample stem: __SAMPLE_STEM__)</h3>
  <div id="hint">鼠标拖拽可连续框选多个 ROI；每个 ROI 可单独设置位置与放大倍数。<br>
  关闭本标签页或点「完成并关闭服务」会结束本地 HTTP 进程并释放端口。</div>
  <div id="imgBox">
    <img id="img" src="__SAMPLE_SRC__">
    <div id="sel"></div>
  </div>
  <div id="bar">
    <button id="clearBtn">清空 ROI</button>
    <button id="copyBtn">复制</button>
    <button id="saveBtn">保存到配置</button>
    <button type="button" id="shutdownSrvBtn">完成并关闭服务</button>
    <span id="saveInfo" style="color:#9ad;"></span>
  </div>
  <div style="margin-top:10px;color:#bbb;font-size:13px;">拖框后会自动新增一行；可修改每行位置与倍率。</div>
  <table id="roiTable" style="width:100%;margin-top:8px;border-collapse:collapse;font-size:13px;">
    <thead><tr style="color:#aaa;">
      <th style="text-align:left;padding:4px 6px;border-bottom:1px solid #333;">#</th>
      <th style="text-align:left;padding:4px 6px;border-bottom:1px solid #333;">ROI (x,y,w,h)</th>
      <th style="text-align:left;padding:4px 6px;border-bottom:1px solid #333;">位置</th>
      <th style="text-align:left;padding:4px 6px;border-bottom:1px solid #333;">倍率</th>
      <th style="text-align:left;padding:4px 6px;border-bottom:1px solid #333;">操作</th>
    </tr></thead>
    <tbody id="roiBody"></tbody>
  </table>
  <div style="margin-top:14px;color:#bbb;font-size:13px;">小眼睛预览（实时，模拟最终布局）</div>
  <canvas id="previewCanvas" style="margin-top:8px;border:1px solid #333;max-width:95vw;"></canvas>
  <div id="previewInfo" style="margin-top:6px;color:#bbb;font-size:12px;"></div>
</div>
<script>
const img = document.getElementById('img');
const box = document.getElementById('imgBox');
const sel = document.getElementById('sel');
const roiBody = document.getElementById('roiBody');
const clearBtn = document.getElementById('clearBtn');
const copyBtn = document.getElementById('copyBtn');
const saveBtn = document.getElementById('saveBtn');
const saveInfo = document.getElementById('saveInfo');
const previewCanvas = document.getElementById('previewCanvas');
const previewInfo = document.getElementById('previewInfo');
let dragging = false;
let sx = 0, sy = 0, ex = 0, ey = 0;
let roiItems = [];
const POS = ['tl','tr','bl','br'];
const COLOR_PALETTE = ['#00ffff', '#00ff00', '#ff00ff', '#ffff00', '#ff8000', '#0080ff'];
const AUTO_POS = {
  tl: ['tl','tr','bl','br'],
  tr: ['tr','tl','br','bl'],
  bl: ['bl','br','tl','tr'],
  br: ['br','bl','tr','tl'],
};
const defaultPos = '__DEFAULT_POS__';
const defaultFactor = Number('__DEFAULT_FACTOR__') || 2.0;
const defaultBorder = Number('__DEFAULT_BORDER__') || 2;

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function boxPos(evt) {
  const r = box.getBoundingClientRect();
  return {
    x: clamp(evt.clientX - r.left, 0, img.clientWidth),
    y: clamp(evt.clientY - r.top, 0, img.clientHeight),
  };
}
function drawSel() {
  const x = Math.min(sx, ex), y = Math.min(sy, ey);
  const w = Math.abs(ex - sx), h = Math.abs(ey - sy);
  sel.style.display = (w >= 2 && h >= 2) ? 'block' : 'none';
  sel.style.left = x + 'px';
  sel.style.top = y + 'px';
  sel.style.width = w + 'px';
  sel.style.height = h + 'px';
}
function currentROIArray() {
  const x1 = Math.min(sx, ex), y1 = Math.min(sy, ey);
  const w1 = Math.abs(ex - sx), h1 = Math.abs(ey - sy);
  if (w1 < 2 || h1 < 2) { return null; }
  const rx = img.naturalWidth / img.clientWidth;
  const ry = img.naturalHeight / img.clientHeight;
  const X = Math.round(x1 * rx);
  const Y = Math.round(y1 * ry);
  const W = Math.round(w1 * rx);
  const H = Math.round(h1 * ry);
  if (W <= 0 || H <= 0) return null;
  return [X, Y, W, H];
}

function autoPos(idx) {
  const seq = AUTO_POS[defaultPos] || AUTO_POS.br;
  return seq[idx % seq.length];
}

function addROI(arr) {
  const idx = roiItems.length;
  roiItems.push({ roi: arr, pos: autoPos(idx), factor: defaultFactor });
  renderROIItems();
}

function removeROI(idx) {
  roiItems.splice(idx, 1);
  renderROIItems();
}

function renderROIItems() {
  roiBody.innerHTML = '';
  roiItems.forEach((it, i) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td style="padding:4px 6px;border-bottom:1px solid #222;">${i + 1}</td>
      <td style="padding:4px 6px;border-bottom:1px solid #222;">${it.roi.join(',')}</td>
      <td style="padding:4px 6px;border-bottom:1px solid #222;">
        <select data-i="${i}" data-k="pos" style="background:#1d1d1d;color:#fff;border:1px solid #555;">
          ${POS.map(p => `<option value="${p}" ${it.pos===p?'selected':''}>${p}</option>`).join('')}
        </select>
      </td>
      <td style="padding:4px 6px;border-bottom:1px solid #222;">
        <input data-i="${i}" data-k="factor" value="${it.factor}" style="width:80px;background:#1d1d1d;color:#fff;border:1px solid #555;">
      </td>
      <td style="padding:4px 6px;border-bottom:1px solid #222;">
        <button data-i="${i}" data-k="del">删除</button>
      </td>
    `;
    roiBody.appendChild(tr);
  });
  drawPreview();
}

function syncItemsFromInputs() {
  roiBody.querySelectorAll('select[data-k="pos"]').forEach(el => {
    const i = Number(el.dataset.i);
    if (roiItems[i]) roiItems[i].pos = el.value;
  });
  roiBody.querySelectorAll('input[data-k="factor"]').forEach(el => {
    const i = Number(el.dataset.i);
    const v = Number(el.value);
    if (roiItems[i] && Number.isFinite(v) && v > 0) roiItems[i].factor = v;
  });
}

function rectOverlap(a, b) {
  return !(a.x + a.w <= b.x || b.x + b.w <= a.x || a.y + a.h <= b.y || b.y + b.h <= a.y);
}

function computeInsetRect(item, W, H) {
  const [x0, y0, w0, h0] = item.roi;
  const x = Math.max(0, Math.min(x0, W - 1));
  const y = Math.max(0, Math.min(y0, H - 1));
  const rw = Math.max(1, Math.min(w0, W - x));
  const rh = Math.max(1, Math.min(h0, H - y));
  let iw = Math.max(24, Math.round(rw * Number(item.factor || defaultFactor)));
  let ih = Math.max(24, Math.round(rh * Number(item.factor || defaultFactor)));
  const margin = Math.max(6, defaultBorder + 3);
  const maxIw = Math.max(24, W - 2 * margin);
  const maxIh = Math.max(24, H - 2 * margin);
  if (iw > maxIw || ih > maxIh) {
    const s = Math.min(maxIw / Math.max(1, iw), maxIh / Math.max(1, ih));
    iw = Math.max(24, Math.round(iw * s));
    ih = Math.max(24, Math.round(ih * s));
  }
  let ix = margin, iy = margin;
  if (item.pos === 'tr') { ix = W - iw - margin; iy = margin; }
  else if (item.pos === 'bl') { ix = margin; iy = H - ih - margin; }
  else if (item.pos === 'br') { ix = W - iw - margin; iy = H - ih - margin; }
  ix = Math.max(0, Math.min(ix, W - iw));
  iy = Math.max(0, Math.min(iy, H - ih));
  return { roi: {x, y, w: rw, h: rh}, inset: {x: ix, y: iy, w: iw, h: ih} };
}

function drawPreview() {
  if (!img.complete || !img.naturalWidth || !img.naturalHeight) return;
  syncItemsFromInputs();
  const W = img.naturalWidth, H = img.naturalHeight;
  const maxW = 780, maxH = 420;
  const s = Math.min(maxW / W, maxH / H, 1.0);
  const cw = Math.max(1, Math.round(W * s));
  const ch = Math.max(1, Math.round(H * s));
  previewCanvas.width = cw;
  previewCanvas.height = ch;
  const ctx = previewCanvas.getContext('2d');
  ctx.clearRect(0, 0, cw, ch);
  ctx.drawImage(img, 0, 0, cw, ch);
  ctx.save();
  ctx.scale(s, s);
  ctx.lineWidth = Math.max(1, defaultBorder);
  const insetRects = [];
  roiItems.forEach((it, i) => {
    const strokeColor = roiItems.length <= 1 ? '#00ffff' : COLOR_PALETTE[i % COLOR_PALETTE.length];
    const g = computeInsetRect(it, W, H);
    insetRects.push(g.inset);
    ctx.strokeStyle = strokeColor;
    ctx.strokeRect(g.roi.x, g.roi.y, g.roi.w, g.roi.h);
    ctx.drawImage(img, g.roi.x, g.roi.y, g.roi.w, g.roi.h, g.inset.x, g.inset.y, g.inset.w, g.inset.h);
    ctx.strokeRect(g.inset.x, g.inset.y, g.inset.w, g.inset.h);
  });
  ctx.restore();
  if (!roiItems.length) {
    previewInfo.textContent = '尚未添加 ROI。';
    return;
  }
  let overlap = 0;
  for (let i = 0; i < insetRects.length; i++) {
    for (let j = i + 1; j < insetRects.length; j++) {
      if (rectOverlap(insetRects[i], insetRects[j])) overlap++;
    }
  }
  previewInfo.textContent = overlap > 0
    ? `注意：检测到 ${overlap} 组放大窗重叠，请调整位置或倍率。`
    : '当前放大窗无重叠。';
}

async function saveConfig() {
  syncItemsFromInputs();
  if (!roiItems.length) { saveInfo.textContent = '请先框选 ROI'; return; }
  try {
    const resp = await fetch('/save-roi', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ zoom_items: roiItems, sample_stem: '__SAMPLE_STEM__' })
    });
    if (!resp.ok) throw new Error('save failed');
    const data = await resp.json();
    saveInfo.textContent = `已保存: ${data.path}`;
  } catch (e) {
    saveInfo.textContent = '保存失败，请检查服务是否运行';
  }
}
box.addEventListener('mousedown', (e) => {
  if (e.button !== 0) return;
  const p = boxPos(e);
  dragging = true; sx = ex = p.x; sy = ey = p.y;
  drawSel();
});
window.addEventListener('mousemove', (e) => {
  if (!dragging) return;
  const p = boxPos(e);
  ex = p.x; ey = p.y;
  drawSel();
});
window.addEventListener('mouseup', () => {
  if (!dragging) return;
  dragging = false;
  drawSel();
  const arr = currentROIArray();
  if (arr) addROI(arr);
  sel.style.display = 'none';
});
copyBtn.addEventListener('click', async () => {
  if (!roiItems.length) return;
  const s = roiItems.map(it => it.roi.join(',')).join(';');
  try {
    await navigator.clipboard.writeText(s);
    copyBtn.textContent = '已复制';
    setTimeout(() => copyBtn.textContent = '复制', 900);
  } catch (_) {}
});
clearBtn.addEventListener('click', () => {
  roiItems = [];
  renderROIItems();
});
roiBody.addEventListener('change', (e) => {
  const t = e.target;
  if (!(t instanceof HTMLElement)) return;
  if (t.dataset.k === 'pos' || t.dataset.k === 'factor') drawPreview();
});
roiBody.addEventListener('input', (e) => {
  const t = e.target;
  if (!(t instanceof HTMLElement)) return;
  if (t.dataset.k === 'factor') drawPreview();
});
roiBody.addEventListener('click', (e) => {
  const t = e.target;
  if (!(t instanceof HTMLElement)) return;
  if (t.dataset.k === 'del') {
    removeROI(Number(t.dataset.i));
  }
});
saveBtn.addEventListener('click', saveConfig);
function requestLocalServerShutdown() {
  try {
    fetch('/shutdown', { method: 'GET', keepalive: true, cache: 'no-store' }).catch(() => {});
  } catch (e) {}
}
window.addEventListener('pagehide', requestLocalServerShutdown);
window.addEventListener('beforeunload', requestLocalServerShutdown);
const shutdownSrvBtn = document.getElementById('shutdownSrvBtn');
if (shutdownSrvBtn) {
  shutdownSrvBtn.addEventListener('click', () => {
    requestLocalServerShutdown();
    shutdownSrvBtn.disabled = true;
    shutdownSrvBtn.textContent = '已请求关闭…';
  });
}
img.addEventListener('load', drawPreview);
</script>
</body></html>"""
    html = html.replace("__SAMPLE_STEM__", str(sample_stem))
    html = html.replace("__DEFAULT_POS__", str(default_pos))
    html = html.replace("__DEFAULT_FACTOR__", str(float(default_factor)))
    html = html.replace("__DEFAULT_BORDER__", str(int(default_border)))
    if sample_web_path.exists():
        html = html.replace("__SAMPLE_SRC__", sample_web_name)
    else:
        html = html.replace("__SAMPLE_SRC__", sample_path.as_posix())
    picker_path.write_text(html, encoding="utf-8")
    return picker_path

def write_web_profile_line_picker(sample_stem, sample_path: Path, sample_img, out_dir: Path):
    """生成网页剖面线选取器：在样本图上拖拽画线，保存为 profile_line.json。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    picker_path = out_dir / "profile_line_picker.html"
    sample_web_name = "_profile_line_sample.png"
    sample_web_path = out_dir / sample_web_name
    if sample_img is not None:
        ok, enc = cv2.imencode(".png", sample_img)
        if ok:
            enc.tofile(str(sample_web_path))
    html = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Profile Line Picker</title>
<style>
body { background:#111; color:#eee; font-family:system-ui,Arial; margin:0; }
#wrap { max-width:1100px; margin:20px auto; padding:0 16px 24px; }
#hint { color:#bbb; margin:10px 0 14px; line-height:1.45; }
#imgBox { position:relative; display:inline-block; border:1px solid #444; cursor:crosshair; }
#img { max-width:95vw; max-height:76vh; display:block; user-select:none; -webkit-user-drag:none; }
#overlay { position:absolute; left:0; top:0; pointer-events:none; }
#bar { margin-top:14px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
button { padding:6px 10px; cursor:pointer; }
#coords { color:#9ad; font-size:13px; margin-top:8px; }
</style>
</head><body>
<div id="wrap">
  <h3>剖面线选取 (sample stem: __SAMPLE_STEM__)</h3>
  <div id="hint">拖拽画线段；可连续添加多条（颜色区分）。坐标按<strong>原图像素</strong>保存，与 <code>--profile-line</code> 分号语法一致。<br>
  「撤销上一条」删除最近一条；「清空」删除全部；保存写入 <code>profile_lines</code>。<br>
  关闭本标签页或点「完成并关闭服务」会结束本地 HTTP 并释放端口。</div>
  <div id="imgBox">
    <img id="img" src="__SAMPLE_SRC__" draggable="false">
    <canvas id="overlay"></canvas>
  </div>
  <div id="bar">
    <button id="undoBtn">撤销上一条</button>
    <button id="resetBtn">清空</button>
    <button id="copyBtn">复制全部(分号分隔)</button>
    <button id="saveBtn">保存到配置</button>
    <button type="button" id="shutdownSrvBtn">完成并关闭服务</button>
    <span id="saveInfo" style="color:#9ad;"></span>
  </div>
  <div id="coords"></div>
</div>
<script>
const img = document.getElementById('img');
const box = document.getElementById('imgBox');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const resetBtn = document.getElementById('resetBtn');
const copyBtn = document.getElementById('copyBtn');
const saveBtn = document.getElementById('saveBtn');
const saveInfo = document.getElementById('saveInfo');
const coordsEl = document.getElementById('coords');

const LINE_COLORS = ['#ffcc00', '#00ffff', '#ff66ff', '#66ff66', '#ff9966', '#99aaff'];
let dragging = false;
let sx = 0, sy = 0, ex = 0, ey = 0;
let lines = [];

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function boxPos(evt) {
  const r = box.getBoundingClientRect();
  return {
    x: clamp(evt.clientX - r.left, 0, img.clientWidth),
    y: clamp(evt.clientY - r.top, 0, img.clientHeight),
  };
}
function toNatural(px, py) {
  const rx = img.naturalWidth / img.clientWidth;
  const ry = img.naturalHeight / img.clientHeight;
  return { x: Math.round(px * rx), y: Math.round(py * ry) };
}
function syncOverlaySize() {
  overlay.width = img.clientWidth;
  overlay.height = img.clientHeight;
  drawLine();
}
function drawLine() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const rx = img.clientWidth / img.naturalWidth;
  const ry = img.clientHeight / img.naturalHeight;
  lines.forEach((ln, i) => {
    const x0 = ln[0] * rx, y0 = ln[1] * ry;
    const x1 = ln[2] * rx, y1 = ln[3] * ry;
    ctx.strokeStyle = LINE_COLORS[i % LINE_COLORS.length];
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(x0, y0);
    ctx.lineTo(x1, y1);
    ctx.stroke();
  });
  if (dragging || (sx !== ex || sy !== ey)) {
    const w = Math.hypot(ex - sx, ey - sy);
    if (w < 1) return;
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(sx, sy);
    ctx.lineTo(ex, ey);
    ctx.stroke();
  }
}
function updateCoordsText() {
  if (!lines.length) {
    coordsEl.textContent = '';
    return;
  }
  coordsEl.textContent = lines.length + ' 条: ' + lines.map(l => l.join(',')).join('  |  ');
}

const undoBtn = document.getElementById('undoBtn');
box.addEventListener('mousedown', (e) => {
  if (e.button !== 0) return;
  dragging = true;
  const p = boxPos(e);
  sx = ex = p.x; sy = ey = p.y;
  drawLine();
});
window.addEventListener('mousemove', (e) => {
  if (!dragging) return;
  const p = boxPos(e);
  ex = p.x; ey = p.y;
  drawLine();
});
window.addEventListener('mouseup', (e) => {
  if (!dragging) return;
  dragging = false;
  const p = boxPos(e);
  ex = p.x; ey = p.y;
  const a = toNatural(sx, sy);
  const b = toNatural(ex, ey);
  if (Math.abs(a.x - b.x) + Math.abs(a.y - b.y) < 1) {
    saveInfo.textContent = '线段太短';
    drawLine();
    updateCoordsText();
    return;
  }
  lines.push([a.x, a.y, b.x, b.y]);
  drawLine();
  saveInfo.textContent = '';
  updateCoordsText();
});

undoBtn.addEventListener('click', () => {
  lines.pop();
  dragging = false;
  drawLine();
  updateCoordsText();
  saveInfo.textContent = '';
});

resetBtn.addEventListener('click', () => {
  lines = [];
  dragging = false;
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  saveInfo.textContent = '';
  updateCoordsText();
});

copyBtn.addEventListener('click', async () => {
  if (!lines.length) { saveInfo.textContent = '请先画线'; return; }
  const s = lines.map(l => l.join(',')).join(';');
  try {
    await navigator.clipboard.writeText(s);
    copyBtn.textContent = '已复制';
    setTimeout(() => { copyBtn.textContent = '复制全部(分号分隔)'; }, 900);
  } catch (_) { saveInfo.textContent = '复制失败'; }
});

async function saveConfig() {
  if (!lines.length) { saveInfo.textContent = '请先画至少一条线'; return; }
  try {
    const resp = await fetch('/save-profile-line', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ profile_lines: lines, sample_stem: '__SAMPLE_STEM__' })
    });
    if (!resp.ok) throw new Error('save failed');
    const data = await resp.json();
    saveInfo.textContent = `已保存: ${data.path}`;
  } catch (e) {
    saveInfo.textContent = '保存失败，请检查服务是否运行';
  }
}
saveBtn.addEventListener('click', saveConfig);

function requestLocalServerShutdown() {
  try {
    fetch('/shutdown', { method: 'GET', keepalive: true, cache: 'no-store' }).catch(() => {});
  } catch (e) {}
}
window.addEventListener('pagehide', requestLocalServerShutdown);
window.addEventListener('beforeunload', requestLocalServerShutdown);
const shutdownSrvBtn = document.getElementById('shutdownSrvBtn');
if (shutdownSrvBtn) {
  shutdownSrvBtn.addEventListener('click', () => {
    requestLocalServerShutdown();
    shutdownSrvBtn.disabled = true;
    shutdownSrvBtn.textContent = '已请求关闭…';
  });
}

img.addEventListener('load', () => {
  syncOverlaySize();
});
window.addEventListener('resize', () => {
  if (img.complete) syncOverlaySize();
});
</script>
</body></html>"""
    html = html.replace("__SAMPLE_STEM__", str(sample_stem))
    if sample_web_path.exists():
        html = html.replace("__SAMPLE_SRC__", sample_web_name)
    else:
        html = html.replace("__SAMPLE_SRC__", sample_path.as_posix())
    picker_path.write_text(html, encoding="utf-8")
    return picker_path

def start_profile_line_picker_server(file_path: Path, config_path: Path, preferred_port=8000):
    base_dir = file_path.parent.parent if file_path.parent.name else file_path.parent
    rel = file_path.relative_to(base_dir).as_posix()
    port = _pick_available_port(preferred_port)
    host = "127.0.0.1"
    url = f"http://{host}:{port}/{rel}"

    server_code = """
import json, os, sys, time, threading
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

port = int(sys.argv[1])
base_dir = sys.argv[2]
config_path = Path(sys.argv[3]).resolve()
os.chdir(base_dir)

class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return
    def _reply_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        pth = self.path.split('?', 1)[0].rstrip('/')
        if pth == '/shutdown':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(bytes([111, 107, 10]))

            def _shutdown():
                try:
                    self.server.shutdown()
                except Exception:
                    pass
            threading.Thread(target=_shutdown, daemon=True).start()
            return
        return super().do_GET()
    def do_POST(self):
        if self.path != '/save-profile-line':
            self._reply_json(404, {'ok': False, 'error': 'not found'})
            return
        try:
            n = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            n = 0
        raw = self.rfile.read(max(0, n))
        try:
            payload = json.loads(raw.decode('utf-8') if raw else '{}')
        except Exception:
            self._reply_json(400, {'ok': False, 'error': 'invalid json'})
            return
        lines_out = []
        raw_lines = payload.get('profile_lines')
        if isinstance(raw_lines, list):
            for item in raw_lines:
                if not (isinstance(item, list) and len(item) == 4):
                    continue
                try:
                    x0, y0, x1, y1 = [int(v) for v in item]
                except Exception:
                    continue
                if abs(x1 - x0) + abs(y1 - y0) == 0:
                    continue
                lines_out.append([x0, y0, x1, y1])
        if not lines_out:
            line = payload.get('profile_line')
            if isinstance(line, list) and len(line) == 4:
                try:
                    x0, y0, x1, y1 = [int(v) for v in line]
                except Exception:
                    pass
                else:
                    if abs(x1 - x0) + abs(y1 - y0) > 0:
                        lines_out.append([x0, y0, x1, y1])
        if not lines_out:
            self._reply_json(400, {'ok': False, 'error': 'profile_lines or profile_line required'})
            return
        sample_stem = str(payload.get('sample_stem') or '')
        x0, y0, x1, y1 = lines_out[0]
        strs = [','.join(str(int(v)) for v in ln) for ln in lines_out]
        data = {
            'profile_lines': lines_out,
            'profile_line': [x0, y0, x1, y1],
            'profile_line_str': ';'.join(strs),
            'sample_stem': sample_stem,
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        self._reply_json(200, {'ok': True, 'path': str(config_path)})

httpd = ThreadingHTTPServer(('0.0.0.0', port), Handler)
httpd.serve_forever()
""".strip()
    cmd = [sys.executable, "-u", "-c", server_code, str(port), str(base_dir), str(config_path)]
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(0.25)
    if p.poll() is not None:
        return url, False, port, None
    return url, True, port, p.pid


def write_web_brightness_map_roi_picker(sample_stem, sample_path: Path, sample_img, out_dir: Path):
    """生成网页矩形 ROI 选取器：保存为 brightness_map_roi.json（与剖面线同一像素坐标系）。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    picker_path = out_dir / "brightness_map_picker.html"
    sample_web_name = "_brightness_map_sample.png"
    sample_web_path = out_dir / sample_web_name
    if sample_img is not None:
        ok, enc = cv2.imencode(".png", sample_img)
        if ok:
            enc.tofile(str(sample_web_path))
    html = """<!doctype html>
<html><head><meta charset="utf-8">
<title>Brightness Map ROI</title>
<style>
body { background:#111; color:#eee; font-family:system-ui,Arial; margin:0; }
#wrap { max-width:1100px; margin:20px auto; padding:0 16px 24px; }
#hint { color:#bbb; margin:10px 0 14px; line-height:1.45; }
#imgBox { position:relative; display:inline-block; border:1px solid #444; cursor:crosshair; }
#img { max-width:95vw; max-height:76vh; display:block; user-select:none; -webkit-user-drag:none; }
#overlay { position:absolute; left:0; top:0; pointer-events:none; }
#bar { margin-top:14px; display:flex; align-items:center; gap:8px; flex-wrap:wrap; }
button { padding:6px 10px; cursor:pointer; }
#coords { color:#9ad; font-size:13px; margin-top:8px; word-break:break-all; }
</style>
</head><body>
<div id="wrap">
  <h3>亮度图 ROI（矩形） sample stem: __SAMPLE_STEM__</h3>
  <div id="hint">拖拽框选区域；坐标为<strong>原图像素</strong> x,y,w,h，与 <code>--brightness-map-roi</code> 一致。<br>
  再次拖拽会<strong>替换</strong>当前 ROI。保存后运行 compare_viewer 将在 <code>brightness_maps/</code> 下导出各列灰度的二维伪彩图。<br>
  下方可填<strong>伪彩色值域</strong>（vmin/vmax），与导出图一致，会写入配置文件。<br>
  关闭本标签页或点「完成并关闭服务」会结束本地 HTTP 并释放端口。</div>
  <div id="imgBox">
    <img id="img" src="__SAMPLE_SRC__" draggable="false">
    <canvas id="overlay"></canvas>
  </div>
  <div id="bar">
    <button id="clearBtn">清除</button>
    <button id="copyBtn">复制 x,y,w,h</button>
    <button id="saveBtn">保存到配置</button>
    <button type="button" id="shutdownSrvBtn">完成并关闭服务</button>
    <span id="saveInfo" style="color:#9ad;"></span>
  </div>
  <div id="barVlim" style="margin-top:8px;display:flex;align-items:center;gap:8px;flex-wrap:wrap;color:#bbb;font-size:13px;">
    <span>伪彩色值域 vmin — vmax</span>
    <input id="bmin" type="number" step="any" value="__DEF_VMIN__" style="width:78px;padding:4px 6px;background:#1d1d1d;border:1px solid #555;color:#fff;">
    <input id="bmax" type="number" step="any" value="__DEF_VMAX__" style="width:78px;padding:4px 6px;background:#1d1d1d;border:1px solid #555;color:#fff;">
  </div>
  <div id="coords"></div>
</div>
<script>
const img = document.getElementById('img');
const box = document.getElementById('imgBox');
const overlay = document.getElementById('overlay');
const ctx = overlay.getContext('2d');
const saveInfo = document.getElementById('saveInfo');
const coordsEl = document.getElementById('coords');

let dragging = false;
let sx = 0, sy = 0, ex = 0, ey = 0;
let saved = null;

function clamp(v, a, b) { return Math.max(a, Math.min(b, v)); }
function boxPos(evt) {
  const r = box.getBoundingClientRect();
  return {
    x: clamp(evt.clientX - r.left, 0, img.clientWidth),
    y: clamp(evt.clientY - r.top, 0, img.clientHeight),
  };
}
function toNatural(px, py) {
  const rx = img.naturalWidth / img.clientWidth;
  const ry = img.naturalHeight / img.clientHeight;
  return { x: Math.round(px * rx), y: Math.round(py * ry) };
}
function syncOverlaySize() {
  overlay.width = img.clientWidth;
  overlay.height = img.clientHeight;
  drawAll();
}
function naturalRectFromClient() {
  const a = toNatural(sx, sy), b = toNatural(ex, ey);
  const x = Math.min(a.x, b.x), y = Math.min(a.y, b.y);
  const w = Math.abs(a.x - b.x), h = Math.abs(a.y - b.y);
  return [x, y, w, h];
}
function drawAll() {
  ctx.clearRect(0, 0, overlay.width, overlay.height);
  const rx = img.clientWidth / img.naturalWidth;
  const ry = img.clientHeight / img.naturalHeight;
  if (saved) {
    const [x, y, w, h] = saved;
    ctx.strokeStyle = '#ffaa33';
    ctx.lineWidth = 2;
    ctx.strokeRect(x * rx, y * ry, w * rx, h * ry);
  }
  if (dragging) {
    const x = Math.min(sx, ex), y = Math.min(sy, ey);
    const w = Math.abs(ex - sx), h = Math.abs(ey - sy);
    ctx.setLineDash([6, 4]);
    ctx.strokeStyle = '#ffffff';
    ctx.lineWidth = 2;
    ctx.strokeRect(x, y, w, h);
    ctx.setLineDash([]);
  }
}
function updateCoordsText() {
  if (!saved) { coordsEl.textContent = ''; return; }
  coordsEl.textContent = '当前 ROI: ' + saved.join(',');
}

box.addEventListener('mousedown', (e) => {
  if (e.button !== 0) return;
  dragging = true;
  const p = boxPos(e);
  sx = ex = p.x; sy = ey = p.y;
  drawAll();
});
window.addEventListener('mousemove', (e) => {
  if (!dragging) return;
  const p = boxPos(e);
  ex = p.x; ey = p.y;
  drawAll();
});
window.addEventListener('mouseup', (e) => {
  if (!dragging) return;
  dragging = false;
  const p = boxPos(e);
  ex = p.x; ey = p.y;
  const arr = naturalRectFromClient();
  if (arr[2] < 2 || arr[3] < 2) {
    saveInfo.textContent = '框太小，请拖大一点';
    drawAll();
    updateCoordsText();
    return;
  }
  saved = arr;
  saveInfo.textContent = '';
  drawAll();
  updateCoordsText();
});

document.getElementById('clearBtn').addEventListener('click', () => {
  saved = null;
  dragging = false;
  saveInfo.textContent = '';
  drawAll();
  updateCoordsText();
});
document.getElementById('copyBtn').addEventListener('click', async () => {
  if (!saved) { saveInfo.textContent = '请先框选'; return; }
  const s = saved.join(',');
  try {
    await navigator.clipboard.writeText(s);
    saveInfo.textContent = '已复制';
    setTimeout(() => { saveInfo.textContent = ''; }, 900);
  } catch (_) { saveInfo.textContent = '复制失败'; }
});

async function saveConfig() {
  if (!saved) { saveInfo.textContent = '请先框选 ROI'; return; }
  const vmin = Number(document.getElementById('bmin').value);
  const vmax = Number(document.getElementById('bmax').value);
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax) || vmin >= vmax) {
    saveInfo.textContent = '值域无效：需 vmin < vmax';
    return;
  }
  try {
    const resp = await fetch('/save-brightness-map-roi', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        brightness_map_roi: saved,
        brightness_map_vlim: [vmin, vmax],
        sample_stem: '__SAMPLE_STEM__'
      })
    });
    if (!resp.ok) throw new Error('save failed');
    const data = await resp.json();
    saveInfo.textContent = `已保存: ${data.path}`;
  } catch (e) {
    saveInfo.textContent = '保存失败，请检查服务是否运行';
  }
}
document.getElementById('saveBtn').addEventListener('click', saveConfig);

function requestLocalServerShutdown() {
  try {
    fetch('/shutdown', { method: 'GET', keepalive: true, cache: 'no-store' }).catch(() => {});
  } catch (e) {}
}
window.addEventListener('pagehide', requestLocalServerShutdown);
window.addEventListener('beforeunload', requestLocalServerShutdown);
const shutdownSrvBtn = document.getElementById('shutdownSrvBtn');
if (shutdownSrvBtn) {
  shutdownSrvBtn.addEventListener('click', () => {
    requestLocalServerShutdown();
    shutdownSrvBtn.disabled = true;
    shutdownSrvBtn.textContent = '已请求关闭…';
  });
}

img.addEventListener('load', () => { syncOverlaySize(); });
window.addEventListener('resize', () => { if (img.complete) syncOverlaySize(); });
</script>
</body></html>"""
    html = html.replace("__SAMPLE_STEM__", str(sample_stem))
    cfg_try = out_dir / "brightness_map_roi.json"
    def_vmin, def_vmax = "0", "255"
    vlim0 = load_brightness_map_vlim_config(cfg_try)
    if vlim0 is not None:
        def_vmin, def_vmax = repr(vlim0[0]), repr(vlim0[1])
    html = html.replace("__DEF_VMIN__", def_vmin).replace("__DEF_VMAX__", def_vmax)
    if sample_web_path.exists():
        html = html.replace("__SAMPLE_SRC__", sample_web_name)
    else:
        html = html.replace("__SAMPLE_SRC__", sample_path.as_posix())
    picker_path.write_text(html, encoding="utf-8")
    return picker_path


def start_brightness_map_roi_picker_server(file_path: Path, config_path: Path, preferred_port=8000):
    base_dir = file_path.parent.parent if file_path.parent.name else file_path.parent
    rel = file_path.relative_to(base_dir).as_posix()
    port = _pick_available_port(preferred_port)
    host = "127.0.0.1"
    url = f"http://{host}:{port}/{rel}"

    server_code = """
import json, os, sys, time, threading
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

port = int(sys.argv[1])
base_dir = sys.argv[2]
config_path = Path(sys.argv[3]).resolve()
os.chdir(base_dir)

class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return
    def _reply_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        pth = self.path.split('?', 1)[0].rstrip('/')
        if pth == '/shutdown':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(bytes([111, 107, 10]))

            def _shutdown():
                try:
                    self.server.shutdown()
                except Exception:
                    pass
            threading.Thread(target=_shutdown, daemon=True).start()
            return
        return super().do_GET()
    def do_POST(self):
        if self.path != '/save-brightness-map-roi':
            self._reply_json(404, {'ok': False, 'error': 'not found'})
            return
        try:
            n = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            n = 0
        raw = self.rfile.read(max(0, n))
        try:
            payload = json.loads(raw.decode('utf-8') if raw else '{}')
        except Exception:
            self._reply_json(400, {'ok': False, 'error': 'invalid json'})
            return
        roi = payload.get('brightness_map_roi')
        if not (isinstance(roi, list) and len(roi) == 4):
            self._reply_json(400, {'ok': False, 'error': 'brightness_map_roi [x,y,w,h] required'})
            return
        try:
            x, y, w, h = [int(v) for v in roi]
        except Exception:
            self._reply_json(400, {'ok': False, 'error': 'roi must be ints'})
            return
        if w <= 0 or h <= 0:
            self._reply_json(400, {'ok': False, 'error': 'w,h must be > 0'})
            return
        sample_stem = str(payload.get('sample_stem') or '')
        data = {
            'brightness_map_roi': [x, y, w, h],
            'brightness_map_roi_str': f'{x},{y},{w},{h}',
            'sample_stem': sample_stem,
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        vm = payload.get('brightness_map_vlim')
        if isinstance(vm, list) and len(vm) == 2:
            try:
                a, b = float(vm[0]), float(vm[1])
            except Exception:
                pass
            else:
                if a < b:
                    data['brightness_map_vlim'] = [a, b]
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        self._reply_json(200, {'ok': True, 'path': str(config_path)})

httpd = ThreadingHTTPServer(('0.0.0.0', port), Handler)
httpd.serve_forever()
""".strip()
    cmd = [sys.executable, "-u", "-c", server_code, str(port), str(base_dir), str(config_path)]
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(0.25)
    if p.poll() is not None:
        return url, False, port, None
    return url, True, port, p.pid


def load_profile_lines_config(config_path: Path):
    """读取 profile_line.json，返回多条线段 [(x0,y0,x1,y1), ...] 或 None。"""
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    out = []
    raw_lines = data.get("profile_lines")
    if isinstance(raw_lines, list):
        for item in raw_lines:
            if not (isinstance(item, list) and len(item) == 4):
                continue
            try:
                x0, y0, x1, y1 = [int(v) for v in item]
            except (TypeError, ValueError):
                continue
            if abs(x1 - x0) + abs(y1 - y0) == 0:
                continue
            out.append((x0, y0, x1, y1))
    if out:
        return out
    line = data.get("profile_line")
    if isinstance(line, list) and len(line) == 4:
        try:
            x0, y0, x1, y1 = [int(v) for v in line]
        except (TypeError, ValueError):
            return None
        if abs(x1 - x0) + abs(y1 - y0) == 0:
            return None
        return [(x0, y0, x1, y1)]
    s = data.get("profile_line_str")
    if isinstance(s, str) and s.strip():
        parsed = []
        for chunk in [c.strip() for c in s.split(";") if c.strip()]:
            parts = [p.strip() for p in chunk.split(",")]
            if len(parts) != 4:
                continue
            try:
                x0, y0, x1, y1 = [int(v) for v in parts]
            except ValueError:
                continue
            if abs(x1 - x0) + abs(y1 - y0) == 0:
                continue
            parsed.append((x0, y0, x1, y1))
        if parsed:
            return parsed
    return None

def load_profile_ylims_config(config_path: Path):
    """从 profile_line.json 读取 profile_ylims 或单条 profile_ylim；返回 [(ymin,ymax), ...] 或 None。"""
    if not config_path.exists():
        return None
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return None
    raw = data.get("profile_ylims")
    if isinstance(raw, list) and raw:
        out = []
        for item in raw:
            if not (isinstance(item, list) and len(item) == 2):
                continue
            try:
                a, b = float(item[0]), float(item[1])
            except (TypeError, ValueError):
                continue
            if min(a, b) >= max(a, b):
                continue
            out.append((min(a, b), max(a, b)))
        if out:
            return out
    y = data.get("profile_ylim")
    if isinstance(y, list) and len(y) == 2:
        try:
            a, b = float(y[0]), float(y[1])
        except (TypeError, ValueError):
            return None
        if min(a, b) >= max(a, b):
            return None
        return [(min(a, b), max(a, b))]
    return None

def load_zoom_roi_config(config_path: Path):
    if not config_path.exists():
        return []
    try:
        data = json.loads(config_path.read_text(encoding="utf-8"))
    except Exception:
        return []
    items = []
    if isinstance(data.get("zoom_items"), list):
        for item in data.get("zoom_items"):
            if not isinstance(item, dict):
                continue
            roi = item.get("roi")
            pos = str(item.get("pos", "br")).lower()
            fac = item.get("factor", 2.0)
            if not (isinstance(roi, list) and len(roi) == 4):
                continue
            try:
                x, y, w, h = [int(v) for v in roi]
                fac = float(fac)
            except (TypeError, ValueError):
                continue
            if w > 0 and h > 0 and fac > 0 and pos in {"tl", "tr", "bl", "br"}:
                items.append({"roi": (x, y, w, h), "pos": pos, "factor": fac})
    if items:
        return items

    rois = []
    if isinstance(data.get("zoom_rois"), list):
        for item in data.get("zoom_rois"):
            if isinstance(item, list) and len(item) == 4:
                try:
                    x, y, w, h = [int(v) for v in item]
                except (TypeError, ValueError):
                    continue
                if w > 0 and h > 0:
                    rois.append((x, y, w, h))
    if rois:
        return build_zoom_items(rois, base_pos="br", base_factor=2.0)
    roi = data.get("zoom_roi")
    if isinstance(roi, list) and len(roi) == 4:
        try:
            x, y, w, h = [int(v) for v in roi]
        except (TypeError, ValueError):
            return []
        if w > 0 and h > 0:
            return build_zoom_items([(x, y, w, h)], base_pos="br", base_factor=2.0)
    return []

def save_zoom_items_config(config_path: Path, zoom_items, sample_stem=""):
    norm_items = []
    for item in (zoom_items or []):
        roi = item.get("roi") if isinstance(item, dict) else None
        pos = str(item.get("pos", "br")).lower() if isinstance(item, dict) else "br"
        fac = item.get("factor", 2.0) if isinstance(item, dict) else 2.0
        if not (isinstance(roi, (list, tuple)) and len(roi) == 4):
            continue
        try:
            x, y, w, h = [int(v) for v in roi]
            fac = float(fac)
        except (TypeError, ValueError):
            continue
        if w <= 0 or h <= 0 or fac <= 0 or pos not in {"tl", "tr", "bl", "br"}:
            continue
        norm_items.append({"roi": [x, y, w, h], "pos": pos, "factor": fac})

    if not norm_items:
        return

    x, y, w, h = norm_items[0]["roi"]
    payload = {
        "zoom_roi": [x, y, w, h],
        "zoom_rois": [it["roi"] for it in norm_items],
        "zoom_items": norm_items,
        "zoom_roi_str": f"{x},{y},{w},{h}",
        "sample_stem": str(sample_stem or ""),
        "updated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

def _pick_available_port(preferred_port, host="127.0.0.1", max_tries=50):
    p = int(preferred_port)
    for _ in range(max_tries):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(0.2)
            if s.connect_ex((host, p)) != 0:
                return p
        p += 1
    return int(preferred_port)


def _print_http_picker_child_hint(server_pid, real_port, web_port_arg):
    """说明：子进程提供 HTTP；关闭选取页或点「完成并关闭服务」会请求 /shutdown 并退出子进程。"""
    if server_pid is None:
        return
    print(
        f"[INFO] HTTP 子进程 PID={server_pid}（端口 {real_port}）。"
        f"关闭浏览器标签页或点页面上的「完成并关闭服务」会自动结束该进程并释放端口；"
        f"若仍占用可手动 kill {server_pid}。"
    )


def start_roi_picker_server(file_path: Path, config_path: Path, preferred_port=8000):
    base_dir = file_path.parent.parent if file_path.parent.name else file_path.parent
    rel = file_path.relative_to(base_dir).as_posix()
    port = _pick_available_port(preferred_port)
    host = "127.0.0.1"
    url = f"http://{host}:{port}/{rel}"

    server_code = """
import json, os, sys, time, threading
from pathlib import Path
from http.server import ThreadingHTTPServer, SimpleHTTPRequestHandler

port = int(sys.argv[1])
base_dir = sys.argv[2]
config_path = Path(sys.argv[3]).resolve()
os.chdir(base_dir)

class Handler(SimpleHTTPRequestHandler):
    def log_message(self, fmt, *args):
        return
    def _reply_json(self, status, payload):
        data = json.dumps(payload, ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type', 'application/json; charset=utf-8')
        self.send_header('Content-Length', str(len(data)))
        self.end_headers()
        self.wfile.write(data)
    def do_GET(self):
        pth = self.path.split('?', 1)[0].rstrip('/')
        if pth == '/shutdown':
            self.send_response(200)
            self.send_header('Content-Type', 'text/plain; charset=utf-8')
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(bytes([111, 107, 10]))

            def _shutdown():
                try:
                    self.server.shutdown()
                except Exception:
                    pass
            threading.Thread(target=_shutdown, daemon=True).start()
            return
        return super().do_GET()
    def do_POST(self):
        if self.path != '/save-roi':
            self._reply_json(404, {'ok': False, 'error': 'not found'})
            return
        try:
            n = int(self.headers.get('Content-Length', '0'))
        except ValueError:
            n = 0
        raw = self.rfile.read(max(0, n))
        try:
            payload = json.loads(raw.decode('utf-8') if raw else '{}')
        except Exception:
            self._reply_json(400, {'ok': False, 'error': 'invalid json'})
            return
        items = []
        if isinstance(payload.get('zoom_items'), list):
            for it in payload.get('zoom_items'):
                if not isinstance(it, dict):
                    continue
                roi = it.get('roi')
                pos = str(it.get('pos', 'br')).lower()
                fac = it.get('factor', 2.0)
                if not (isinstance(roi, list) and len(roi) == 4):
                    continue
                try:
                    x, y, w, h = [int(v) for v in roi]
                    fac = float(fac)
                except Exception:
                    continue
                if w > 0 and h > 0 and fac > 0 and pos in {'tl','tr','bl','br'}:
                    items.append({'roi': [x, y, w, h], 'pos': pos, 'factor': fac})
        if not items:
            roi = payload.get('zoom_roi')
            if not (isinstance(roi, list) and len(roi) == 4):
                self._reply_json(400, {'ok': False, 'error': 'zoom_items or zoom_roi required'})
                return
            try:
                x, y, w, h = [int(v) for v in roi]
            except Exception:
                self._reply_json(400, {'ok': False, 'error': 'zoom_roi must be ints'})
                return
            if w <= 0 or h <= 0:
                self._reply_json(400, {'ok': False, 'error': 'w/h must be > 0'})
                return
            items = [{'roi': [x, y, w, h], 'pos': 'br', 'factor': 2.0}]
        sample_stem = str(payload.get('sample_stem') or '')
        x, y, w, h = items[0]['roi']
        data = {
            'zoom_roi': [x, y, w, h],
            'zoom_rois': [it['roi'] for it in items],
            'zoom_items': items,
            'zoom_roi_str': f'{x},{y},{w},{h}',
            'sample_stem': sample_stem,
            'updated_at': time.strftime('%Y-%m-%d %H:%M:%S'),
        }
        config_path.parent.mkdir(parents=True, exist_ok=True)
        config_path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
        self._reply_json(200, {'ok': True, 'path': str(config_path)})

httpd = ThreadingHTTPServer(('0.0.0.0', port), Handler)
httpd.serve_forever()
""".strip()
    cmd = [sys.executable, "-u", "-c", server_code, str(port), str(base_dir), str(config_path)]
    p = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    time.sleep(0.25)
    if p.poll() is not None:
        return url, False, port, None
    return url, True, port, p.pid

def apply_zoom_inset(img, roi, inset_pos="br", border=2, color=(0, 255, 255), zoom_factor=2.5):
    if roi is None or img is None:
        return img
    h, w = img.shape[:2]
    if h <= 1 or w <= 1:
        return img

    x, y, rw, rh = roi
    x = max(0, min(x, w - 1))
    y = max(0, min(y, h - 1))
    rw = max(1, min(rw, w - x))
    rh = max(1, min(rh, h - y))
    if rw <= 0 or rh <= 0:
        return img

    out = img.copy()
    cv2.rectangle(out, (x, y), (x + rw, y + rh), color, max(1, border), cv2.LINE_AA)

    crop = out[y:y + rh, x:x + rw]
    iw = max(24, int(round(rw * float(zoom_factor))))
    ih = max(24, int(round(rh * float(zoom_factor))))
    margin = max(6, border + 3)
    # 最终确保放大窗能放进图内
    max_iw = max(24, w - 2 * margin)
    max_ih = max(24, h - 2 * margin)
    if iw > max_iw or ih > max_ih:
        s = min(max_iw / float(max(1, iw)), max_ih / float(max(1, ih)))
        iw = max(24, int(round(iw * s)))
        ih = max(24, int(round(ih * s)))
    iw = min(iw, w)
    ih = min(ih, h)
    inset = cv2.resize(crop, (iw, ih), interpolation=cv2.INTER_CUBIC)

    if inset_pos == "tl":
        ix, iy = margin, margin
    elif inset_pos == "tr":
        ix, iy = w - iw - margin, margin
    elif inset_pos == "bl":
        ix, iy = margin, h - ih - margin
    else:  # br
        ix, iy = w - iw - margin, h - ih - margin

    ix = max(0, min(ix, w - iw))
    iy = max(0, min(iy, h - ih))

    out[iy:iy + ih, ix:ix + iw] = inset
    cv2.rectangle(out, (ix, iy), (ix + iw, iy + ih), (0, 0, 0), max(1, border + 1), cv2.LINE_AA)
    cv2.rectangle(out, (ix, iy), (ix + iw, iy + ih), color, max(1, border), cv2.LINE_AA)

    return out

def tile_images(images, labels, cols=0, pad=8, bg=18, sep=2, font_scale=0.6, label_scale_mult=1.25, label_color=(0, 255, 255),
                ref_short_side=512.0, zoom_items=None, zoom_border=2):
    valid = [im for im in images if im is not None]
    if not valid: return None
    th, tw = valid[0].shape[:2]
    normed = []
    for im in images:
        if im is None:
            im = np.full((th, tw, 3), 64, np.uint8)
            im = draw_label(im, "MISSING", font_scale, label_scale_mult=label_scale_mult, label_color=label_color,
                            ref_short_side=ref_short_side)
        else:
            if im.shape[:2] != (th, tw):
                im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
            if zoom_items:
                total_items = len(zoom_items)
                for i, it in enumerate(zoom_items):
                    roi = it.get("roi")
                    pos = it.get("pos", "br")
                    factor = it.get("factor", 2.0)
                    zoom_color = pick_zoom_color(i, total_items, label_color)
                    im = apply_zoom_inset(im, roi, inset_pos=pos,
                                          border=zoom_border, color=zoom_color, zoom_factor=factor)
        normed.append(im)
    normed = [
        draw_label(im, lbl, font_scale, pad//2, label_scale_mult=label_scale_mult, label_color=label_color,
                   ref_short_side=ref_short_side)
        for im, lbl in zip(normed, labels)
    ]
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

def parse_profile_lines(arg: str):
    """多条剖面线：分号分隔多段，每段 x0,y0,x1,y1（与单条规则相同）。"""
    s = str(arg).strip()
    if not s:
        return None
    chunks = [c.strip() for c in s.split(";") if c.strip()]
    out = []
    for ci, chunk in enumerate(chunks):
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 4:
            print(
                f"[Error] --profile-line 第 {ci + 1} 段需要四个整数 x0,y0,x1,y1；"
                f"多段用英文分号分隔，例如 120,200,400,200;120,280,400,280",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            x0, y0, x1, y1 = [int(v) for v in parts]
        except ValueError:
            print(f"[Error] --profile-line 第 {ci + 1} 段必须为整数: {chunk}", file=sys.stderr)
            sys.exit(1)
        if abs(x1 - x0) + abs(y1 - y0) == 0:
            print(f"[Error] --profile-line 第 {ci + 1} 段不能退化为点", file=sys.stderr)
            sys.exit(1)
        out.append((x0, y0, x1, y1))
    if not out:
        return None
    return out

def parse_profile_ylims(arg: str):
    """剖面纵轴：单段 'ymin,ymax'；多段与剖面线一一对应，用英文分号分隔，例如 0,255;40,220。
    仅一段时表示所有剖面线共用该范围。空则返回 None（导出时用默认约 (-5,260)）。"""
    s = str(arg).strip()
    if not s:
        return None
    chunks = [c.strip() for c in s.split(";") if c.strip()]
    out = []
    for ci, chunk in enumerate(chunks):
        parts = [p.strip() for p in chunk.split(",")]
        if len(parts) != 2:
            print(
                f"[Error] --profile-ylim 第 {ci + 1} 段需要 min,max；多段用英文分号分隔，例如 0,255;30,200",
                file=sys.stderr,
            )
            sys.exit(1)
        try:
            a, b = float(parts[0]), float(parts[1])
        except ValueError:
            print(f"[Error] --profile-ylim 第 {ci + 1} 段必须为数字: {chunk}", file=sys.stderr)
            sys.exit(1)
        if min(a, b) >= max(a, b):
            print(f"[Error] --profile-ylim 第 {ci + 1} 段要求 min < max: {chunk}", file=sys.stderr)
            sys.exit(1)
        out.append((min(a, b), max(a, b)))
    return out


def validate_profile_ylims_line_count(profile_ylims, n_lines: int):
    """profile_ylims 要么 1 段（共用），要么与剖面线条数相同。"""
    if not profile_ylims or n_lines <= 0:
        return
    if len(profile_ylims) == 1:
        return
    if len(profile_ylims) != n_lines:
        print(
            f"[Error] 剖面纵轴段数 {len(profile_ylims)} 与剖面线条数 {n_lines} 不一致；"
            f"请只给 1 段 min,max（全部共用）或给满 {n_lines} 段。",
            file=sys.stderr,
        )
        sys.exit(1)

def panel_grid_layout(n_cells, cols_arg):
    """与 tile_images 相同的行列数：返回 (rows, cols_effective)。"""
    n_cells = int(n_cells)
    cols_arg = int(cols_arg)
    if n_cells <= 0:
        return 1, 1
    cols = cols_arg
    if cols <= 0 or cols >= n_cells:
        return 1, n_cells
    cols_e = cols
    rows = int(math.ceil(n_cells / cols_e))
    return rows, cols_e

def align_images_for_profile(images):
    """与 tile_images 一致：以本 stem 首张非空图为 (th,tw)，其余列 resize 到同尺寸（不含标签与放大窗）。"""
    valid = [im for im in images if im is not None]
    if not valid:
        return None, None
    th, tw = valid[0].shape[:2]
    out = []
    for im in images:
        if im is None:
            out.append(None)
        else:
            if im.shape[:2] != (th, tw):
                im = cv2.resize(im, (tw, th), interpolation=cv2.INTER_AREA)
            out.append(im)
    return out, (th, tw)

def grayscale_line_profile(img_bgr, x0, y0, x1, y1):
    """沿线段等距采样灰度（双线性用最近邻网格近似：对齐到像素中心）。"""
    if img_bgr is None:
        return None
    h, w = img_bgr.shape[:2]
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    n = max(2, int(np.ceil(np.hypot(x1 - x0, y1 - y0))) + 1)
    xs = np.linspace(float(x0), float(x1), n)
    ys = np.linspace(float(y0), float(y1), n)
    xi = np.clip(np.round(xs).astype(np.int32), 0, w - 1)
    yi = np.clip(np.round(ys).astype(np.int32), 0, h - 1)
    return gray[yi, xi].astype(np.float32)

def profile_sample_axis_px(x0, y0, x1, y1, w, h):
    """与 grayscale_line_profile 相同的采样网格，返回横轴：沿线累积欧氏距离（像素）。"""
    n = max(2, int(np.ceil(np.hypot(x1 - x0, y1 - y0))) + 1)
    xs = np.linspace(float(x0), float(x1), n)
    ys = np.linspace(float(y0), float(y1), n)
    xi = np.clip(np.round(xs).astype(np.int32), 0, w - 1).astype(np.float32)
    yi = np.clip(np.round(ys).astype(np.int32), 0, h - 1).astype(np.float32)
    dx = np.diff(xi)
    dy = np.diff(yi)
    step = np.sqrt(dx * dx + dy * dy)
    return np.concatenate([np.zeros(1, dtype=np.float32), np.cumsum(step.astype(np.float32))])

def draw_profile_line_on_panel(panel, line_xy, n_images, cols_arg, pad, sep, th, tw,
                               color=(0, 0, 255), thickness=0):
    """在拼图 panel 的每个格子上叠加同一条剖面线（与 align 后单格像素坐标一致）。"""
    if panel is None or line_xy is None or n_images <= 0 or th <= 0 or tw <= 0:
        return
    lx0, ly0, lx1, ly1 = line_xy
    n = int(n_images)
    _rows, ccols = panel_grid_layout(n, cols_arg)
    cell_h, cell_w = th + 2 * pad, tw + 2 * pad
    t = max(2, int(round(0.0025 * float(min(th, tw)) + 1.5)))
    if thickness and thickness > 0:
        t = int(thickness)
    clip = cv2.clipLine((0, 0, tw, th), (int(lx0), int(ly0)), (int(lx1), int(ly1)))
    if not clip[0]:
        return
    _, p1, p2 = clip
    a0, a1 = int(p1[0]), int(p1[1])
    b0, b1 = int(p2[0]), int(p2[1])
    for idx in range(n):
        r, c = divmod(idx, ccols)
        yb = r * (cell_h + sep) + pad
        xb = c * (cell_w + sep) + pad
        cv2.line(
            panel,
            (xb + a0, yb + a1),
            (xb + b0, yb + b1),
            color,
            t,
            cv2.LINE_AA,
        )

def draw_profile_lines_on_panel(panel, lines_list, n_images, cols_arg, pad, sep, th, tw):
    """在拼图上叠加多条剖面线，颜色与放大窗调色板类似以便区分。"""
    if panel is None or not lines_list or n_images <= 0 or th <= 0 or tw <= 0:
        return
    nlines = len(lines_list)
    for li, line_xy in enumerate(lines_list):
        bgr = pick_zoom_color(li, nlines, (0, 0, 255))
        draw_profile_line_on_panel(panel, line_xy, n_images, cols_arg, pad, sep, th, tw, color=bgr, thickness=0)


def clip_roi_xywh(roi, img_w, img_h):
    """将 ROI 裁剪到图像内；无效输入返回 None。"""
    if roi is None or img_w <= 0 or img_h <= 0:
        return None
    x, y, w, h = roi
    x = max(0, min(int(x), img_w - 1))
    y = max(0, min(int(y), img_h - 1))
    w = max(1, min(int(w), img_w - x))
    h = max(1, min(int(h), img_h - y))
    return x, y, w, h


def draw_brightness_map_roi_on_panel(panel, roi_xywh, n_images, cols_arg, pad, sep, th, tw, color=None, thickness=0):
    """在拼图各格上叠加亮度图分析用的矩形 ROI（对齐后单格坐标）。"""
    if panel is None or roi_xywh is None or n_images <= 0 or th <= 0 or tw <= 0:
        return
    clip = clip_roi_xywh(roi_xywh, tw, th)
    if clip is None:
        return
    x0, y0, rw, rh = clip
    n = int(n_images)
    _rows, ccols = panel_grid_layout(n, int(cols_arg))
    cell_h, cell_w = th + 2 * pad, tw + 2 * pad
    t = max(2, int(round(0.0025 * float(min(th, tw)) + 1.5)))
    if thickness and thickness > 0:
        t = int(thickness)
    if color is None:
        color = (40, 200, 255)
    for idx in range(n):
        r, c = divmod(idx, ccols)
        yb = r * (cell_h + sep) + pad
        xb = c * (cell_w + sep) + pad
        cv2.rectangle(
            panel,
            (xb + x0, yb + y0),
            (xb + x0 + rw - 1, yb + y0 + rh - 1),
            color,
            t,
            cv2.LINE_AA,
        )


def _collect_series_for_line(labels, aligned_images, line_xy):
    x0, y0, x1, y1 = line_xy
    series = []
    for lbl, im in zip(labels, aligned_images):
        prof = grayscale_line_profile(im, x0, y0, x1, y1)
        if prof is None:
            continue
        series.append((str(lbl), prof))
    return series

def export_line_profile_png(
    out_path: Path,
    stem,
    labels,
    aligned_images,
    line_xy,
    dpi=140,
    stack="auto",
    panel_cols=5,
    ylim=None,
):
    """导出单条剖面线对应的一张图；多条线时由调用方循环并分别指定 out_path。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("[Error] 导出剖面图需要 matplotlib（例如: pip install matplotlib）", file=sys.stderr)
        sys.exit(1)
    if line_xy is None:
        return False
    ref_im = next((im for im in aligned_images if im is not None), None)
    if ref_im is None:
        return False
    hh, ww = ref_im.shape[:2]

    palette = (
        "#1f77b4", "#ff7f0e", "#2ca02d", "#d62728", "#9467bd", "#8c564b", "#e377c2",
        "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a", "#ff9896",
        "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
    )

    ymin, ymax = (-5.0, 260.0) if ylim is None else (float(ylim[0]), float(ylim[1]))
    if ymin >= ymax:
        print(f"[Error] --profile-ylim 要求 min < max，得到 ({ymin}, {ymax})", file=sys.stderr)
        sys.exit(1)

    series = _collect_series_for_line(labels, aligned_images, line_xy)
    if not series:
        return False
    x0, y0, x1, y1 = line_xy
    x_axis = profile_sample_axis_px(x0, y0, x1, y1, ww, hh)
    if len(x_axis) != len(series[0][1]):
        x_axis = np.arange(len(series[0][1]), dtype=np.float32)
    sm = str(stack or "auto").strip().lower()
    if sm == "always":
        use_stack = True
    elif sm == "never":
        use_stack = False
    else:
        use_stack = len(series) >= 5
    n = len(series)
    rows, cols_e = panel_grid_layout(n, int(panel_cols))
    if use_stack:
        fig_w = min(2.05 * cols_e + 2.0, 36.0)
        fig_h = min(2.0 * rows + 1.55, 42.0)
        fig, axes = plt.subplots(rows, cols_e, figsize=(fig_w, fig_h), sharex=True, sharey=True, dpi=dpi)
        ax_flat = np.atleast_1d(axes).ravel()
        total_slots = int(rows * cols_e)
        for j in range(total_slots):
            ax = ax_flat[j]
            if j >= n:
                ax.set_visible(False)
                continue
            lbl, y = series[j]
            col = palette[j % len(palette)]
            ax.plot(x_axis, y, color=col, linewidth=1.35)
            ax.set_ylim(ymin, ymax)
            ax.grid(True, alpha=0.35)
            ax.set_title(lbl, loc="left", fontsize=10, color=col, fontweight="bold", pad=4)
            ax.tick_params(axis="both", labelsize=6)
            rr, cc = divmod(j, cols_e)
            if rr < rows - 1:
                ax.tick_params(axis="x", labelbottom=False)
            if cc > 0:
                ax.tick_params(axis="y", labelleft=False)
        if hasattr(fig, "supylabel"):
            fig.supylabel("Gray (0–255)", fontsize=9, x=0.02)
        else:
            for j in range(n):
                if j % cols_e == 0:
                    ax_flat[j].set_ylabel("Gray", fontsize=7)
                    break
        fig.tight_layout(rect=(0.05, 0.03, 1.0, 0.93))
    else:
        fig, ax = plt.subplots(figsize=(10, 5), dpi=dpi)
        for i, (lbl, y) in enumerate(series):
            col = palette[i % len(palette)]
            ax.plot(x_axis, y, label=lbl, color=col, linewidth=1.2)
        ax.set_ylabel("Gray level (0–255)")
        ax.set_ylim(ymin, ymax)
        ax.grid(True, alpha=0.35)
        ax.legend(loc="best", fontsize=8, ncol=2 if len(series) > 8 else 1)
        fig.tight_layout()

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight")
    plt.close(fig)
    return True


def export_brightness_map_png(
    out_path: Path,
    stem,
    labels,
    aligned_images,
    roi_xywh,
    dpi=140,
    panel_cols=5,
    cmap="viridis",
    vmin=0.0,
    vmax=255.0,
):
    """导出框选区域内各列灰度的二维伪彩图；vmin/vmax 控制色标映射范围。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib import gridspec
    except ImportError:
        print("[Error] 导出亮度图需要 matplotlib（例如: pip install matplotlib）", file=sys.stderr)
        sys.exit(1)
    if roi_xywh is None:
        return False
    ref_im = next((im for im in aligned_images if im is not None), None)
    if ref_im is None:
        return False
    hh, ww = ref_im.shape[:2]
    clipped = clip_roi_xywh(roi_xywh, ww, hh)
    if clipped is None:
        return False
    rx, ry, rw, rh = clipped
    v0, v1 = float(vmin), float(vmax)
    if v0 >= v1:
        print(f"[Error] 亮度图值域要求 vmin < vmax，得到 ({v0}, {v1})", file=sys.stderr)
        sys.exit(1)

    cmap_name = str(cmap or "viridis").strip() or "viridis"
    try:
        plt.get_cmap(cmap_name)
    except Exception:
        cmap_name = "viridis"

    series = []
    for lbl, im in zip(labels, aligned_images):
        if im is None:
            continue
        if im.shape[:2] != (hh, ww):
            im = cv2.resize(im, (ww, hh), interpolation=cv2.INTER_AREA)
        crop = im[ry : ry + rh, rx : rx + rw]
        if crop.size == 0:
            continue
        g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        series.append((str(lbl), np.asarray(g)))
    if not series:
        return False

    n = len(series)
    use_stack = n >= 5
    rows, cols_e = panel_grid_layout(n, int(panel_cols))
    total_slots = int(rows * cols_e)

    def _draw_axes_and_cbar(fig, ax_flat, n_ax, cax=None):
        mappable = None
        for j in range(n_ax):
            ax = ax_flat[j]
            if j >= n:
                ax.set_visible(False)
                continue
            lbl, g = series[j]
            im_art = ax.imshow(
                g, cmap=cmap_name, vmin=v0, vmax=v1, aspect="equal", interpolation="nearest"
            )
            if mappable is None:
                mappable = im_art
            ax.set_title(lbl, fontsize=9, fontweight="bold", pad=2)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.tick_params(axis="both", which="both", length=0)
        if mappable is not None and cax is not None:
            cb = fig.colorbar(mappable, cax=cax)
            cb.ax.tick_params(labelsize=7)
            cb.set_label("Gray", fontsize=8, labelpad=2)

    if use_stack:
        fig_w = min(0.88 * cols_e + 0.42, 14.0)
        fig_h = min(0.72 * rows + 0.52, 12.0)
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        gs = gridspec.GridSpec(
            rows,
            cols_e + 1,
            figure=fig,
            width_ratios=[1.0] * cols_e + [0.028],
            wspace=0.02,
            hspace=0.06,
            left=0.02,
            right=0.995,
            top=0.94,
            bottom=0.05,
        )
        ax_flat = []
        for j in range(total_slots):
            r, c = divmod(j, cols_e)
            ax_flat.append(fig.add_subplot(gs[r, c]))
        cax = fig.add_subplot(gs[:, -1])
        _draw_axes_and_cbar(fig, ax_flat, total_slots, cax=cax)
    else:
        fig_w = min(0.92 * n + 0.38, 12.0)
        fig_h = min(max(2.05, 1.65 * rh / max(1.0, rw)), 5.8)
        fig = plt.figure(figsize=(fig_w, fig_h), dpi=dpi)
        gs = gridspec.GridSpec(
            1,
            n + 1,
            figure=fig,
            width_ratios=[1.0] * n + [0.028],
            wspace=0.02,
            left=0.02,
            right=0.995,
            top=0.92,
            bottom=0.06,
        )
        ax_flat = [fig.add_subplot(gs[0, j]) for j in range(n)]
        cax = fig.add_subplot(gs[0, -1])
        _draw_axes_and_cbar(fig, ax_flat, n, cax=cax)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(str(out_path), bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)
    return True


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

def parse_label_dir_arg(arg: str, arg_name: str):
    if ":" not in arg:
        print(f"[Error] {arg_name} 需要 '标签:目录' 形式，例如 ours:/path/to/dir", file=sys.stderr)
        sys.exit(1)
    name, path = arg.split(":", 1)
    name = name.strip(); path = path.strip()
    p = Path(path)
    if not p.is_dir():
        print(f"[Error] {arg_name} 目录不存在: {p}", file=sys.stderr)
        sys.exit(1)
    return name, p

def parse_label_color_arg(arg: str):
    presets = {
        "yellow": (0, 255, 255),
        "white": (255, 255, 255),
        "red": (0, 0, 255),
        "green": (0, 255, 0),
        "blue": (255, 0, 0),
        "cyan": (255, 255, 0),
        "magenta": (255, 0, 255),
        "black": (0, 0, 0),
    }
    s = str(arg).strip().lower()
    if s in presets:
        return presets[s]
    parts = [x.strip() for x in str(arg).split(",")]
    if len(parts) == 3:
        try:
            r, g, b = [int(x) for x in parts]
            if all(0 <= v <= 255 for v in (r, g, b)):
                return (b, g, r)  # OpenCV 使用 BGR
        except ValueError:
            pass
    print(f"[Error] --label-color 无效: {arg}。可用颜色名如 yellow，或 R,G,B（如 255,255,0）", file=sys.stderr)
    sys.exit(1)

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
    ap.add_argument("--root", type=str, default=USER_COMMON_DEFAULTS["root"],
                    help="方法结果根目录（包含各方法子目录）")
    ap.add_argument("--methods", nargs="*", default=USER_COMMON_DEFAULTS["methods"],
                    help="方法子目录显示顺序；留空=自动发现并按字母序")
    ap.add_argument("--my-method", type=str, default=USER_COMMON_DEFAULTS["my_method"],
                    help="追加到最后一列的方法，格式 '标签:目录'")
    ap.add_argument("--ref", action="append", default=USER_COMMON_DEFAULTS["ref"],
                    help="新增参考列：格式 '标签:目录'，可重复两次，例如 "
                         "--ref 'IR:/data/IR' --ref 'VIS:/data/VIS'")
    ap.add_argument("--exts", nargs="*", default=[".png",".jpg",".jpeg",".bmp"],
                    help="允许的扩展名（含优先级，前者优先）；大小写不敏感")
    ap.add_argument("--out-dir", type=str, default="output")
    ap.add_argument("--cols", type=int, default=USER_COMMON_DEFAULTS["cols"])
    ap.add_argument("--font-scale", type=float, default=0.6)
    ap.add_argument("--pad", type=int, default=8)
    ap.add_argument("--sep", type=int, default=2)
    ap.add_argument("--bg-gray", type=int, default=18)
    ap.add_argument("--label-scale", type=float, default=2.,
                    help="方法名标签字号倍率（与 --font-scale 相乘）；在图像短边等于 --label-ref-side 时达到该基准大小")
    ap.add_argument("--label-ref-side", type=float, default=512.0,
                    help="标签随图缩放时的参考短边（像素）：图越大字越大、图越小字越小；典型 256–768")
    ap.add_argument("--label-color", type=str, default="yellow",
                    help="方法名标签颜色：颜色名（yellow/white/red/green/blue/cyan/magenta/black）或 R,G,B")
    ap.add_argument("--stem-title-mult", type=float, default=1.0,
                    help="底部 stem 标题条（如 00123D [1/10]）高度与字号倍率，<1 更扁更小，例如 0.65")
    ap.add_argument("--only-stems", nargs="*", default=USER_COMMON_DEFAULTS["only_stems"],
                    help="仅导出指定 stem（不含扩展名），可多个；不传或全空则导出全部")
    ap.add_argument("--zoom-roi", type=str, default="",
                    help="局部放大 ROI：单个 x,y,w,h，或多个以 ';' 分隔（如 120,80,64,64;300,200,40,40）")
    ap.add_argument("--pick-zoom-roi", action="store_true",
                    help="交互式框选 ROI（鼠标拖拽），优先级高于 --zoom-roi")
    ap.add_argument("--pick-zoom-roi-web", action="store_true",
                    help="生成浏览器 ROI 选框页面（服务器/无 GUI 环境推荐）")
    ap.add_argument("--pick-stem", type=str, default="",
                    help="交互选框时优先使用的样本 stem（需存在于导出列表）")
    ap.add_argument("--web-port", type=int, default=8000,
                    help="网页选取器首选端口；若被占用则自动递增。注意 HTTP 在独立子进程中长期运行，用毕请 kill 子进程 PID（启动成功时会打印）以释放端口")
    ap.add_argument("--zoom-roi-config", type=str, default="",
                    help="ROI 配置文件路径（默认 out-dir/zoom_roi.json）")
    ap.add_argument("--ignore-zoom-roi-config", action="store_true",
                    help="忽略 ROI 配置文件，不自动读取")
    ap.add_argument("--zoom-pos", type=str, default=USER_COMMON_DEFAULTS["zoom_pos"], choices=["tl", "tr", "bl", "br"],
                    help="放大窗位置：tl/tr/bl/br")
    ap.add_argument("--zoom-factor", type=float, default=1.8,
                    help="固定放大倍数（>1 时放大更明显）")
    ap.add_argument("--zoom-poses", type=str, default="",
                    help="多 ROI 的位置列表（逗号分隔），如 br,tl；留空则按 --zoom-pos 自动分配")
    ap.add_argument("--zoom-factors", type=str, default=USER_COMMON_DEFAULTS["zoom_factors"],
                    help="多 ROI 的倍率列表（逗号分隔），如 2.0,3.0；留空则用 --zoom-factor")
    ap.add_argument("--zoom-border", type=int, default=2,
                    help="ROI 与放大窗边框粗细")
    ap.add_argument("--strict-intersection", action="store_true",
                    help="开启后仅导出在所有列（参考+方法）都存在的 stem；关闭则按并集并对缺图用 MISSING")
    ap.add_argument("--profile-line", type=str, default=USER_COMMON_DEFAULTS.get("profile_line", ""),
                    help="可选：剖面线段 x0,y0,x1,y1；多条用英文分号分隔。单条另存 profiles/<stem>_profile.png；"
                         "多条另存 profiles/<stem>_profile_01.png、_02.png …；与单格像素坐标一致")
    ap.add_argument("--profile-dpi", type=int, default=140,
                    help="剖面图 DPI（仅在使用 --profile-line 或 profile_line.json 时生效）")
    ap.add_argument("--profile-stack", type=str, default="auto", choices=["auto", "always", "never"],
                    help="剖面布局：auto 在曲线数≥5 时分格（与 --cols 同网格、行优先同 *_cmp）；always 总 "
                         "分格；never 总叠在一张图")
    ap.add_argument("--profile-ylim", type=str, default=USER_COMMON_DEFAULTS.get("profile_ylim", ""),
                    help="剖面纵轴：单段 min,max（如 0,255）；多条剖面线可各设一段，用英文分号分隔且段数与线数一致，"
                         "或只给 1 段表示全部共用。留空为默认 (-5,260)。非空时优先于 profile_line.json 中的 profile_ylims")
    ap.add_argument("--profile-line-config", type=str, default="",
                    help="剖面线 JSON 路径（默认 out-dir/profile_line.json；网页选线保存于此）")
    ap.add_argument("--ignore-profile-line-config", action="store_true",
                    help="忽略剖面线配置文件，不自动读取 profile_line.json")
    ap.add_argument("--pick-profile-line-web", action="store_true",
                    help="生成浏览器剖面线选取页（拖拽画线并保存配置，效果同 --profile-line）")
    ap.add_argument("--brightness-map-roi", type=str, default=USER_COMMON_DEFAULTS.get("brightness_map_roi", ""),
                    help="框选区域 x,y,w,h（与剖面线相同：对齐后单格像素）；导出各列灰度二维伪彩图到 brightness_maps/<stem>_brightness.png")
    ap.add_argument("--brightness-map-dpi", type=int, default=140, help="亮度图 matplotlib 导出 DPI")
    ap.add_argument("--brightness-map-cmap", type=str, default="viridis",
                    help="伪彩色图名（matplotlib），如 viridis、magma、inferno；非法名则回退 viridis")
    ap.add_argument("--brightness-map-vlim", type=str, default=USER_COMMON_DEFAULTS.get("brightness_map_vlim", ""),
                    help="伪彩色映射值域 vmin,vmax（如 0,255 或 30,220）；留空则用 brightness_map_roi.json 中的 brightness_map_vlim，否则默认 0,255")
    ap.add_argument("--brightness-map-roi-config", type=str, default="",
                    help="亮度图 ROI JSON 路径（默认 out-dir/brightness_map_roi.json；网页框选保存于此）")
    ap.add_argument("--ignore-brightness-map-roi-config", action="store_true",
                    help="忽略亮度图 ROI 配置文件，不自动读取 brightness_map_roi.json")
    ap.add_argument("--pick-brightness-map-roi-web", action="store_true",
                    help="生成浏览器矩形 ROI 选取页，保存 brightness_map_roi.json 后可直接导出亮度图")
    args = ap.parse_args()

    label_color = parse_label_color_arg(args.label_color)
    out = Path(args.out_dir)
    profile_cfg = Path(args.profile_line_config) if str(args.profile_line_config).strip() else (out / "profile_line.json")
    brightness_map_cfg = (
        Path(args.brightness_map_roi_config) if str(args.brightness_map_roi_config).strip() else (out / "brightness_map_roi.json")
    )
    profile_lines = parse_profile_lines(args.profile_line)
    if profile_lines is None and (not args.ignore_profile_line_config):
        profile_lines = load_profile_lines_config(profile_cfg)
    profile_ylims = parse_profile_ylims(args.profile_ylim)
    if profile_ylims is None and (not args.ignore_profile_line_config):
        profile_ylims = load_profile_ylims_config(profile_cfg)
    if profile_lines:
        validate_profile_ylims_line_count(profile_ylims, len(profile_lines))
        preview = "; ".join(f"{a},{b},{c},{d}" for a, b, c, d in profile_lines[:2])
        tail = f" …共{len(profile_lines)}条" if len(profile_lines) > 2 else ""
        print(f"[INFO] 已读取剖面线配置：{profile_cfg} -> {preview}{tail}")
        if profile_ylims:
            yprev = "; ".join(f"{a:.4g},{b:.4g}" for a, b in profile_ylims[:3])
            ytail = f" …共{len(profile_ylims)}段" if len(profile_ylims) > 3 else ""
            print(f"[INFO] 剖面纵轴范围：{yprev}{ytail}")
    brightness_roi = parse_brightness_map_roi(args.brightness_map_roi)
    if brightness_roi is None and (not args.ignore_brightness_map_roi_config):
        brightness_roi = load_brightness_map_roi_config(brightness_map_cfg)
    brightness_vlim = parse_brightness_map_vlim(args.brightness_map_vlim)
    if brightness_vlim is None and (not args.ignore_brightness_map_roi_config):
        brightness_vlim = load_brightness_map_vlim_config(brightness_map_cfg)
    if brightness_vlim is None:
        brightness_vlim = (0.0, 255.0)
    if brightness_roi:
        bx, by, bw, bh = brightness_roi
        bv0, bv1 = brightness_vlim
        print(
            f"[INFO] 亮度图 ROI：{bx},{by},{bw},{bh}；值域 [{bv0:g},{bv1:g}] → {out.resolve() / 'brightness_maps'}"
        )
    zoom_rois = parse_zoom_rois(args.zoom_roi)
    zoom_factors = parse_zoom_factors(args.zoom_factors)
    zoom_poses = parse_zoom_positions(args.zoom_poses)
    zoom_items = build_zoom_items(zoom_rois, base_pos=args.zoom_pos, base_factor=args.zoom_factor,
                                  factors=zoom_factors, poses=zoom_poses)
    zoom_cfg = Path(args.zoom_roi_config) if str(args.zoom_roi_config).strip() else (out / "zoom_roi.json")
    if (not zoom_items) and (not args.ignore_zoom_roi_config):
        cfg_items = load_zoom_roi_config(zoom_cfg)
        if cfg_items:
            zoom_items = cfg_items
            preview = "; ".join([
                f"{it['roi'][0]},{it['roi'][1]},{it['roi'][2]},{it['roi'][3]}@{it['pos']}x{it['factor']:.2f}"
                for it in cfg_items[:3]
            ])
            more = " ..." if len(cfg_items) > 3 else ""
            print(f"[INFO] 已读取 ROI 配置：{zoom_cfg} -> {preview}{more}")

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

    # 追加“我的方法”到最后一列（若提供）
    if args.my_method:
        my_label, my_dir = parse_label_dir_arg(args.my_method, "--my-method")
        stems, best = build_stem_index(my_dir, args.exts)
        col_labels.append(my_label)
        col_stems.append(stems)
        col_best_path.append(best)

    # 聚合所有 stem（交集或并集）
    if not col_stems:
        print("[Error] 没有可用列", file=sys.stderr); sys.exit(1)
    if args.strict_intersection:
        names = sorted(set.intersection(*col_stems))
    else:
        names = sorted(set.union(*col_stems))
    if args.only_stems:
        wanted = {s.strip() for s in args.only_stems if s.strip()}
        if wanted:
            names = [n for n in names if n in wanted]
    if not names:
        print("[Error] 没有匹配的文件主干（stem）。请检查命名。", file=sys.stderr); sys.exit(1)
    _web_pick_n = int(bool(args.pick_zoom_roi_web)) + int(bool(args.pick_profile_line_web)) + int(
        bool(args.pick_brightness_map_roi_web)
    )
    if _web_pick_n > 1:
        print(
            "[Error] 网页选取模式请三选一：--pick-zoom-roi-web、--pick-profile-line-web、--pick-brightness-map-roi-web（分次运行）。",
            file=sys.stderr,
        )
        sys.exit(1)

    if args.pick_profile_line_web:
        sample_stem, sample_path, sample_img = pick_sample_image(names, col_best_path, pick_stem=args.pick_stem)
        if sample_path is None:
            print("[Error] 没有可用于网页选剖面线的有效图片", file=sys.stderr); sys.exit(1)
        picker_path = write_web_profile_line_picker(sample_stem, sample_path, sample_img, out)
        url, started, real_port, server_pid = start_profile_line_picker_server(
            picker_path, profile_cfg, preferred_port=args.web_port
        )
        print(f"[OK] 已生成网页剖面线选取器：{picker_path}")
        print(f"[CFG] 剖面线将保存到：{profile_cfg}")
        if started:
            print(f"[OPEN] {url}")
            if real_port != args.web_port:
                print(f"[WARN] 端口 {args.web_port} 被占用，已自动改用 {real_port}")
            _print_http_picker_child_hint(server_pid, real_port, args.web_port)
        else:
            print(f"[OPEN] {url}  （服务启动失败，请检查端口或环境）")
        print("[NEXT] 可多次拖拽添加多条线，再点「保存到配置」；再直接运行本脚本即可导出 profiles/（勿加 --ignore-profile-line-config）")
        return

    if args.pick_brightness_map_roi_web:
        sample_stem, sample_path, sample_img = pick_sample_image(names, col_best_path, pick_stem=args.pick_stem)
        if sample_path is None:
            print("[Error] 没有可用于网页选亮度图 ROI 的有效图片", file=sys.stderr)
            sys.exit(1)
        picker_path = write_web_brightness_map_roi_picker(sample_stem, sample_path, sample_img, out)
        url, started, real_port, server_pid = start_brightness_map_roi_picker_server(
            picker_path, brightness_map_cfg, preferred_port=args.web_port
        )
        print(f"[OK] 已生成网页亮度图 ROI 选取器：{picker_path}")
        print(f"[CFG] ROI 将保存到：{brightness_map_cfg}")
        if started:
            print(f"[OPEN] {url}")
            if real_port != args.web_port:
                print(f"[WARN] 端口 {args.web_port} 被占用，已自动改用 {real_port}")
            _print_http_picker_child_hint(server_pid, real_port, args.web_port)
        else:
            print(f"[OPEN] {url}  （服务启动失败，请检查端口或环境）")
        print("[NEXT] 框选后点「保存到配置」，再直接运行本脚本即可导出 brightness_maps/（勿加 --ignore-brightness-map-roi-config）")
        return

    if args.pick_zoom_roi_web:
        sample_stem, sample_path, sample_img = pick_sample_image(names, col_best_path, pick_stem=args.pick_stem)
        if sample_path is None:
            print("[Error] 没有可用于网页选框的有效图片", file=sys.stderr); sys.exit(1)
        picker_path = write_web_roi_picker(sample_stem, sample_path, sample_img, out,
                                           default_pos=args.zoom_pos, default_factor=args.zoom_factor,
                                           default_border=args.zoom_border)
        url, started, real_port, server_pid = start_roi_picker_server(picker_path, zoom_cfg, preferred_port=args.web_port)
        print(f"[OK] 已生成网页选框器：{picker_path}")
        print(f"[CFG] ROI 将自动保存到：{zoom_cfg}")
        if started:
            print(f"[OPEN] {url}")
            if real_port != args.web_port:
                print(f"[WARN] 端口 {args.web_port} 被占用，已自动改用 {real_port}")
            _print_http_picker_child_hint(server_pid, real_port, args.web_port)
        else:
            print(f"[OPEN] {url}  （服务启动失败，请检查端口或环境）")
        print("[NEXT] 浏览器可连续拖框并逐条调位置/倍率，点“保存到配置”后，下次直接运行 compare_viewer.py 即可")
        return
    if args.pick_zoom_roi:
        try:
            zoom_roi = pick_zoom_roi_interactive(names, col_best_path, pick_stem=args.pick_stem)
            if zoom_roi is not None:
                zoom_items = build_zoom_items([zoom_roi], base_pos=args.zoom_pos, base_factor=args.zoom_factor)
                save_zoom_items_config(zoom_cfg, zoom_items, sample_stem=args.pick_stem)
                print(f"[INFO] 已写入 ROI 配置：{zoom_cfg}")
        except SystemExit:
            sample_stem, sample_path, sample_img = pick_sample_image(names, col_best_path, pick_stem=args.pick_stem)
            if sample_path is None:
                raise
            picker_path = write_web_roi_picker(sample_stem, sample_path, sample_img, out,
                                               default_pos=args.zoom_pos, default_factor=args.zoom_factor,
                                               default_border=args.zoom_border)
            url, started, real_port, server_pid = start_roi_picker_server(
                picker_path, zoom_cfg, preferred_port=args.web_port
            )
            print(f"[OK] 检测到无 GUI 环境，已自动生成网页选框器：{picker_path}")
            print(f"[CFG] ROI 将自动保存到：{zoom_cfg}")
            if started:
                print(f"[OPEN] {url}")
                if real_port != args.web_port:
                    print(f"[WARN] 端口 {args.web_port} 被占用，已自动改用 {real_port}")
                _print_http_picker_child_hint(server_pid, real_port, args.web_port)
            else:
                print(f"[OPEN] {url}  （服务启动失败，请检查端口或环境）")
            print("[NEXT] 浏览器可连续拖框并逐条调位置/倍率，点“保存到配置”后，下次直接运行 compare_viewer.py 即可")
            return

    out.mkdir(parents=True, exist_ok=True)
    panels_dir = out / "panels"; panels_dir.mkdir(exist_ok=True)
    profiles_dir = out / "profiles"
    brightness_maps_dir = out / "brightness_maps"
    manifest = []
    profile_exported = 0
    brightness_exported = 0
    profile_warned_oob = False

    # 逐个 stem 导出拼图
    for i, stem in enumerate(names, 1):
        ims, labels = [], []
        for lbl, best_map in zip(col_labels, col_best_path):
            p = best_map.get(stem, None)
            im = imread_any(p) if p and p.exists() else None
            ims.append(im); labels.append(lbl)

        panel = tile_images(ims, labels, cols=args.cols, pad=args.pad,
                            bg=args.bg_gray, sep=args.sep, font_scale=args.font_scale,
                            label_scale_mult=args.label_scale, label_color=label_color,
                            ref_short_side=args.label_ref_side, zoom_items=zoom_items,
                            zoom_border=args.zoom_border)
        if panel is None:
            # 所有列都缺就跳过
            continue

        aligned_prof = None
        prof_shape = None
        if profile_lines or brightness_roi:
            aligned_prof, prof_shape = align_images_for_profile(ims)
            if aligned_prof is not None and prof_shape is not None:
                thp, twp = prof_shape
                if profile_lines:
                    draw_profile_lines_on_panel(
                        panel, profile_lines, len(ims), args.cols, args.pad, args.sep, thp, twp
                    )
                if brightness_roi:
                    draw_brightness_map_roi_on_panel(
                        panel, brightness_roi, len(ims), args.cols, args.pad, args.sep, thp, twp
                    )

        ph, pw = panel.shape[:2]
        tshort = float(min(ph, pw))
        stm = float(args.stem_title_mult)
        title_text = f"{stem}  [{i}/{len(names)}]"
        mx = 10
        my_top, my_bot = 4, 4
        max_w = max(1, pw - 2 * mx)
        # 底部条：字号随拼图缩放；getTextSize 比实际抗锯齿/加粗略小，上下留白要保守
        tscale = 0.42 * (tshort / max(1.0, float(args.label_ref_side))) * stm
        tscale = max(0.2, min(tscale, 2.2))
        tw_thick = max(1, min(2, int(0.9 * tscale)))

        def stem_title_metrics(ts, tk):
            (tw, th), bl = cv2.getTextSize(title_text, cv2.FONT_HERSHEY_SIMPLEX, ts, tk)
            # 略大于 getTextSize，覆盖加粗 + LINE_AA 实际占位
            above = th + 4 * tk + 3
            below = bl + 3 * tk + 2
            need_h = my_top + above + below + my_bot + 2
            return tw, th, bl, above, below, need_h

        max_title_h = max(28, int(0.22 * ph))
        for _ in range(55):
            tw, th, bl, above, below, need_h = stem_title_metrics(tscale, tw_thick)
            if tw <= max_w and need_h <= max_title_h:
                break
            tscale *= 0.91
            tw_thick = max(1, min(2, int(0.9 * tscale)))
            if tscale < 0.12:
                break
        tw, th, bl, above, below, need_h = stem_title_metrics(tscale, tw_thick)
        # 宽度仍溢出则继续只压宽度（极端长 stem）
        while tw > max_w and tscale >= 0.12:
            tscale *= 0.91
            tw_thick = max(1, min(2, int(0.9 * tscale)))
            tw, th, bl, above, below, need_h = stem_title_metrics(tscale, tw_thick)

        title_h = max(18, int(0.018 * tshort * stm), int(need_h))
        title = np.full((title_h, pw, 3), 24, np.uint8)
        # 基线贴底对齐：在 need_h ≤ title_h 时，文字落在条内
        ty = title_h - my_bot - below
        cv2.putText(title, title_text, (mx, ty),
                    cv2.FONT_HERSHEY_SIMPLEX, tscale, (255, 255, 255), tw_thick, cv2.LINE_AA)
        out_img = np.vstack([title, panel])

        out_path = panels_dir / f"{stem}_cmp.png"
        cv2.imencode(".png", out_img)[1].tofile(str(out_path))
        manifest.append(out_path.name)

        if profile_lines and aligned_prof is not None:
            if not profile_warned_oob:
                ref_im = next((im for im in aligned_prof if im is not None), None)
                if ref_im is not None:
                    hh, ww = ref_im.shape[:2]
                    for x0, y0, x1, y1 in profile_lines:
                        if not all(0 <= v < ww for v in (x0, x1)) or not all(0 <= v < hh for v in (y0, y1)):
                            print(f"[WARN] 某条剖面线端点超出首张非空图范围 [0..{ww - 1}]×[0..{hh - 1}]，采样会自动裁剪；"
                                  f"若多 stem 尺寸不同请分别指定线或仅用同分辨率数据。")
                            profile_warned_oob = True
                            break
            n_pl = len(profile_lines)
            for li, line_xy in enumerate(profile_lines):
                if n_pl == 1:
                    prof_path = profiles_dir / f"{stem}_profile.png"
                else:
                    prof_path = profiles_dir / f"{stem}_profile_{li + 1:02d}.png"
                ylim_for = None
                if profile_ylims:
                    ylim_for = profile_ylims[0] if len(profile_ylims) == 1 else profile_ylims[li]
                if export_line_profile_png(
                    prof_path,
                    stem,
                    col_labels,
                    aligned_prof,
                    line_xy,
                    dpi=int(args.profile_dpi),
                    stack=str(args.profile_stack),
                    panel_cols=int(args.cols),
                    ylim=ylim_for,
                ):
                    profile_exported += 1

        if brightness_roi and aligned_prof is not None:
            bm_path = brightness_maps_dir / f"{stem}_brightness.png"
            if export_brightness_map_png(
                bm_path,
                stem,
                col_labels,
                aligned_prof,
                brightness_roi,
                dpi=int(args.brightness_map_dpi),
                panel_cols=int(args.cols),
                cmap=str(args.brightness_map_cmap),
                vmin=float(brightness_vlim[0]),
                vmax=float(brightness_vlim[1]),
            ):
                brightness_exported += 1

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
    if profile_lines:
        prof_abs = profiles_dir.resolve()
        print(f"[OK] Exported {profile_exported} profile figure(s) to {prof_abs}")
        if profile_exported == 0 and manifest:
            print("[WARN] 剖面图数量为 0：请检查线段坐标是否在图内、或该批 stem 是否都成功导出面板。", file=sys.stderr)
    if brightness_roi:
        bm_abs = brightness_maps_dir.resolve()
        print(f"[OK] Exported {brightness_exported} brightness map(s) to {bm_abs}")
        if brightness_exported == 0 and manifest:
            print("[WARN] 亮度图数量为 0：请检查 ROI 是否在图内、或各列是否有有效图像。", file=sys.stderr)
    print(f"[OPEN] {out/'index.html'}  （浏览器打开，←/→ 切换；最左侧为参考列）")

if __name__ == "__main__":
    main()

