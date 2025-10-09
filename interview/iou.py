def iou_cal(point1, point2):
    x1, y1, x2, y2 = point1
    x3, y3, x4, y4 = point2

    # calculate the intersection
    xl = max(x1, x3)
    yl = max(y1, y3)
    xr = min(x2, x4)
    yr = min(y2, y4)

    height = max(0, yr-yl)
    width = max(0, xr-xl)
    intersection = height * width

    area1 = (x2 - x1) * (y2 - y1)
    area2 = (x4 - x3) * (y4 - y3)

    union = area1 + area2 - intersection
    iou = intersection / union
    return iou 