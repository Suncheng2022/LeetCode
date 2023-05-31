import numpy as np


def nms(boxes, scores, iou_threshold):
    """ 复习 """
    if len(boxes) == 0:
        return [], []

    boxes = np.asarray(boxes)
    scores = np.asarray(scores)

    areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    order = scores.argsort()
    picked_boxes, picked_scores = [], []
    while order:
        ind = order[-1]
        picked_boxes.append(boxes[ind])
        picked_scores.append(scores[ind])

        x1 = np.maximum(boxes[ind, 0], boxes[order[:-1], 0])
        y1 = np.maximum(boxes[ind, 1], boxes[order[:-1], 1])
        x2 = np.minimum(boxes[ind, 2], boxes[order[:-1], 2])
        y2 = np.minimum(boxes[ind, 3], boxes[order[:-1], 3])
        w = np.maximum(x2 - x1 + 1, 0)
        h = np.maximum(y2 - y1 + 1, 0)
        intersections = w * h

        ious = intersections / (areas[ind] + areas[order[:-1]] - intersections)
        left = np.nonzero(ious > iou_threshold)
        order = order[left]
    return picked_boxes, picked_scores


def cal_iou(bbox1, bbox2):
    """ 复习计算iou """
    # bbox1、bbox2的坐标
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2
    # bbox1、bbox的面积
    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
    # bbox1、bbox2相交区域的左上、右下坐标
    x1 = max(xmin1, xmin2)
    y1 = max(ymin1, ymin2)
    x2 = min(xmax1, xmax2)
    y2 = min(ymin1, ymin2)
    # bbox1、bbox2相交区域的宽、高
    w = max(0, x2 - x1 + 1)
    h = max(0, y2 - y1 + 1)
    inter = w * h

    iou = inter / (area1 + area2 - inter)
    return iou


def cal_iou02(bboxes1, bboxes2):
    """
    bboxes1.shape [A, 4]
    bboxes2.shape [B, 4]
    """
    A = bboxes1.shape[0]
    B = bboxes2.shape[0]

    # 计算相交区域坐标
    xy_min = np.maximum(bboxes1[:, np.newaxis, :2].repeat(B, axis=1),   # 将bboxes1升维 [A, 2]->[A, B, 2]
                        np.broadcast_to(bboxes2[:, :2], (A, B, 2)))     # 将bboxes2升维 [B, 2]->[A, B, 2]
    xy_max = np.minimum(bboxes1[:, np.newaxis, 2:].repeat(B, axis=1),
                        np.broadcast_to(bboxes2[:, 2:], (A, B, 2)))

    # 计算bboxes1、bboxes2的面积
    areas1 = ((bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1]))[:, np.newaxis, :].repeat(B, axis=1)      # [A, 1]->[A, B, 1]
    areas2 = ((bboxes2[:, 2] - bboxes2[:, 1]) * (bboxes2[:, 3] - bboxes2[:, 1]))[np.newaxis, :, :].repeat(A, axis=0)

    # 计算相交区域面积
    inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)     # [A，B，2]
    inter = inter[:, :, 0] * inter[:, :, 1]     # [A, B, 1]

    # 计算iou
    ious = inter / (areas1 + areas2 - inter)
    return ious
