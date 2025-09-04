import torch
import torch.nn as nn
import torch.nn.functional as F


def smooth_BCE(eps: float = 0.1) -> tuple[float, float]:
    """
    Label smoothing for Binary Cross Entropy (BCE).

    Args:
        eps (float, optional): Smoothing factor in [0, 1]. Default is 0.1.

    Returns:
        tuple[float, float]: 
            - positive (float): Smoothed target for positive class (1 - eps/2).
            - negative (float): Smoothed target for negative class (eps/2).

    Reference:
        https://github.com/ultralytics/yolov3/issues/238#issuecomment-598028441
    """
    return 1.0 - 0.5 * eps, 0.5 * eps

