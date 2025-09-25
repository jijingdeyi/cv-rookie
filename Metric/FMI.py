import numpy as np
from scipy.ndimage import generic_gradient_magnitude, sobel
from scipy.fft import dctn

def _ensure_float64(x):
    return np.asarray(x, dtype=np.float64)

def _feature_map(img, feature: str, feature_param=None):
    """把图像 img 映射到指定特征域."""
    img = _ensure_float64(img)

    if feature == 'none':          # 原始像素
        feat = img

    elif feature == 'gradient':    # 梯度强度（Sobel）
        # feature_param 可选：一个可调用算子；默认用 sobel
        op = sobel if feature_param is None else feature_param
        feat = generic_gradient_magnitude(img, op)

    elif feature == 'edge':        # 边缘（二值）
        # feature_param: 边缘阈值（比如 10 或 0.05*max）
        threshold = feature_param
        if threshold is None:
            # 自动阈值：用梯度强度的 90 分位
            g = generic_gradient_magnitude(img, sobel).astype(np.float64)
            threshold = np.percentile(g, 90.0)
        feat = (sobel(img) > threshold).astype(np.float64)

    elif feature == 'dct':         # DCT 系数
        feat = dctn(img, type=2, norm='ortho')

    elif feature == 'wavelet':
        raise NotImplementedError("wavelet not implemented yet")

    else:
        raise ValueError("feature must be 'none' | 'gradient' | 'edge' | 'dct'")

    return feat

def _quantize(x, bins=64, method='auto'):
    """
    把连续特征量化到 [0, bins-1] 的整数，便于统计直方图/互信息。
    method='auto' 会按分位裁剪后线性归一。
    """
    x = _ensure_float64(x)

    if method == 'auto':
        lo, hi = np.percentile(x, 1.0), np.percentile(x, 99.0)
        if hi <= lo:  # 退化情况
            lo, hi = x.min(), x.max()
        if hi <= lo:  # 全常量
            return np.zeros_like(x, dtype=np.int32)
        y = np.clip((x - lo) / (hi - lo), 0, 1)
        q = (y * (bins - 1) + 0.5).astype(np.int32)
        return q
    elif method == '01':
        y = np.clip(x, 0, 1)
        q = (y * (bins - 1) + 0.5).astype(np.int32)
        return q
    else:
        raise ValueError("unknown quantize method")

def _local_mi_map(X_q, Y_q, win=7, bins=64, eps=1e-12):
    """
    计算两张量化图的 局部互信息 MI map（每个 win×win 窗口一个 MI 值）。
    简洁版：用显式滑窗循环（稳但不极致快）。win 必须是奇数。
    """
    assert win % 2 == 1, "win 必须是奇数"
    X_q = np.asarray(X_q, dtype=np.int32)
    Y_q = np.asarray(Y_q, dtype=np.int32)
    H, W = X_q.shape
    r = win // 2

    out = np.empty((H - 2*r, W - 2*r), dtype=np.float64)

    # 预分配直方图数组，减少重复分配
    for i in range(r, H - r):
        for j in range(r, W - r):
            x_patch = X_q[i - r:i + r + 1, j - r:j + r + 1].ravel()
            y_patch = Y_q[i - r:i + r + 1, j - r:j + r + 1].ravel()

            # 联合直方图
            # 由于是离散整数 ∈ [0, bins-1]，可以用 np.bincount 快一些
            idx = x_patch * bins + y_patch
            hist2d = np.bincount(idx, minlength=bins*bins).reshape(bins, bins).astype(np.float64)

            pxy = hist2d / hist2d.sum()
            px = pxy.sum(axis=1, keepdims=True)  # [bins,1]
            py = pxy.sum(axis=0, keepdims=True)  # [1,bins]

            # MI = sum p(x,y) log ( p(x,y)/(p(x)p(y)) )
            # 只在 pxy>0 处累加，避免 log(0)
            mask = pxy > 0
            num = pxy[mask]
            den = (px @ py)[mask]
            mi = np.sum(num * (np.log(num + eps) - np.log(den + eps)))
            out[i - r, j - r] = mi

    return out

def fmi(ima, imb, imf, feature='gradient', feature_param=None, win=7, bins=64):
    """
    计算 Feature Mutual Information：
    - 把 A, B, F 映射到某特征域
    - 在该特征域中计算 F 与 A / B 的局部互信息 map
    - 取两者的平均 FMI 值（也可改成和/加权和）
    返回: fmi_value, fmi_map_FA, fmi_map_FB
    """
    A = _ensure_float64(ima)
    B = _ensure_float64(imb)
    F = _ensure_float64(imf)

    # 1) 特征映射
    Af = _feature_map(A, feature, feature_param)
    Bf = _feature_map(B, feature, feature_param)
    Ff = _feature_map(F, feature, feature_param)

    # 2) 量化到有限 bins
    Af_q = _quantize(Af, bins=bins, method='auto')
    Bf_q = _quantize(Bf, bins=bins, method='auto')
    Ff_q = _quantize(Ff, bins=bins, method='auto')

    # 3) 局部互信息 map（滑窗）
    mi_FA = _local_mi_map(Ff_q, Af_q, win=win, bins=bins)
    mi_FB = _local_mi_map(Ff_q, Bf_q, win=win, bins=bins)

    # 4) 聚合
    # 常见做法：两者求平均；也有人用求和或加权平均
    fmi_val = float(0.5 * (mi_FA.mean() + mi_FB.mean()))

    return fmi_val, mi_FA, mi_FB


if __name__ == "__main__":

    ima = np.random.rand(100, 100) * 255
    imb = np.random.rand(100, 100) * 255
    imf = np.random.rand(100, 100) * 255
    fmi_val, fmi_map_FA, fmi_map_FB = fmi(ima, imb, imf, feature='gradient', win=7, bins=64)
    print("FMI =", fmi_val)
