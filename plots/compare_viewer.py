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
    "root": "/data/ykx/sota/MSRS",
    "methods": [
        "didfuse", "rfnnest", "mfeif", "sdnet", "piafusion",
        "reconet", "swinfusion", "tardal", "cddfuse", "lrrnet",
        "metafusion", "segmif", "emma", "sage", "gifnet",
    ],
    "my_method": "ours:/home/ykx/ca-fusion-loss/results/MSRS/ours",
    "ref": ["IR:/data/ykx/MSRS/test/ir", "VIS:/data/ykx/MSRS/test/vi"],
    "cols": 5,
    "only_stems": ["00931N"],
    "zoom_pos": "br,tr",
    "zoom_factors": "1.8,2",  # 例如: "1.8,3.2"
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
  <div id="hint">鼠标拖拽可连续框选多个 ROI；每个 ROI 可单独设置位置与放大倍数。</div>
  <div id="imgBox">
    <img id="img" src="__SAMPLE_SRC__">
    <div id="sel"></div>
  </div>
  <div id="bar">
    <button id="clearBtn">清空 ROI</button>
    <button id="copyBtn">复制</button>
    <button id="saveBtn">保存到配置</button>
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

def start_roi_picker_server(file_path: Path, config_path: Path, preferred_port=8000):
    base_dir = file_path.parent.parent if file_path.parent.name else file_path.parent
    rel = file_path.relative_to(base_dir).as_posix()
    port = _pick_available_port(preferred_port)
    host = "127.0.0.1"
    url = f"http://{host}:{port}/{rel}"

    server_code = """
import json, os, sys, time
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
        return url, False, port
    return url, True, port

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
                    help="仅导出指定 stem（不含扩展名）")
    ap.add_argument("--zoom-roi", type=str, default="",
                    help="局部放大 ROI：单个 x,y,w,h，或多个以 ';' 分隔（如 120,80,64,64;300,200,40,40）")
    ap.add_argument("--pick-zoom-roi", action="store_true",
                    help="交互式框选 ROI（鼠标拖拽），优先级高于 --zoom-roi")
    ap.add_argument("--pick-zoom-roi-web", action="store_true",
                    help="生成浏览器 ROI 选框页面（服务器/无 GUI 环境推荐）")
    ap.add_argument("--pick-stem", type=str, default="",
                    help="交互选框时优先使用的样本 stem（需存在于导出列表）")
    ap.add_argument("--web-port", type=int, default=8000,
                    help="网页选框器本地静态服务端口（用于 SSH 端口转发）")
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
    args = ap.parse_args()

    label_color = parse_label_color_arg(args.label_color)
    zoom_rois = parse_zoom_rois(args.zoom_roi)
    zoom_factors = parse_zoom_factors(args.zoom_factors)
    zoom_poses = parse_zoom_positions(args.zoom_poses)
    zoom_items = build_zoom_items(zoom_rois, base_pos=args.zoom_pos, base_factor=args.zoom_factor,
                                  factors=zoom_factors, poses=zoom_poses)
    out = Path(args.out_dir)
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
        names = [n for n in names if n in wanted]
    if not names:
        print("[Error] 没有匹配的文件主干（stem）。请检查命名。", file=sys.stderr); sys.exit(1)
    if args.pick_zoom_roi_web:
        sample_stem, sample_path, sample_img = pick_sample_image(names, col_best_path, pick_stem=args.pick_stem)
        if sample_path is None:
            print("[Error] 没有可用于网页选框的有效图片", file=sys.stderr); sys.exit(1)
        picker_path = write_web_roi_picker(sample_stem, sample_path, sample_img, out,
                                           default_pos=args.zoom_pos, default_factor=args.zoom_factor,
                                           default_border=args.zoom_border)
        url, started, real_port = start_roi_picker_server(picker_path, zoom_cfg, preferred_port=args.web_port)
        print(f"[OK] 已生成网页选框器：{picker_path}")
        print(f"[CFG] ROI 将自动保存到：{zoom_cfg}")
        if started:
            print(f"[OPEN] {url}")
            if real_port != args.web_port:
                print(f"[WARN] 端口 {args.web_port} 被占用，已自动改用 {real_port}")
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
            url, started, real_port = start_roi_picker_server(picker_path, zoom_cfg, preferred_port=args.web_port)
            print(f"[OK] 检测到无 GUI 环境，已自动生成网页选框器：{picker_path}")
            print(f"[CFG] ROI 将自动保存到：{zoom_cfg}")
            if started:
                print(f"[OPEN] {url}")
                if real_port != args.web_port:
                    print(f"[WARN] 端口 {args.web_port} 被占用，已自动改用 {real_port}")
            else:
                print(f"[OPEN] {url}  （服务启动失败，请检查端口或环境）")
            print("[NEXT] 浏览器可连续拖框并逐条调位置/倍率，点“保存到配置”后，下次直接运行 compare_viewer.py 即可")
            return

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
                            bg=args.bg_gray, sep=args.sep, font_scale=args.font_scale,
                            label_scale_mult=args.label_scale, label_color=label_color,
                            ref_short_side=args.label_ref_side, zoom_items=zoom_items,
                            zoom_border=args.zoom_border)
        if panel is None:
            # 所有列都缺就跳过
            continue

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

