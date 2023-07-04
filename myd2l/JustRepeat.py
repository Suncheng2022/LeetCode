import numpy as np


def nms(boxes, scores, iou_threshold):
    """ nms算法 """
    if len(boxes) == 0:
        return [], []

    boxes = np.asarray(boxes)
    scores = np.asarray(scores)

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])

    order = scores.argsort()
    picked_boxes = []
    picked_scores = []
    while order:
        ind = order[-1]     # 取score最大的预测框

        picked_boxes.append(boxes[ind])
        picked_scores.append(scores[ind])

        x1 = np.maximum(boxes[ind, 0], boxes[order[:-1], 0])
        y1 = np.maximum(boxes[ind, 1], boxes[order[:-1], 1])
        x2 = np.minimum(boxes[ind, 2], boxes[order[:-1], 2])
        y2 = np.minimum(boxes[ind, 3], boxes[order[:-1], 3])

        w = np.maximum((x2 - x1 + 1), 0)
        h = np.maximum((y2 - y1 + 1), 0)
        intersections = w * h

        ious = intersections / (areas[ind] + areas[order[:-1]] - intersections)

        left = np.nonzero(ious > iou_threshold)
        order = order[left]
    return picked_boxes, picked_scores


def nms01(boxes, scores, iou_threshold):
    """ 再写一遍 """
    if len(boxes) == 0:
        return [], []

    boxes = np.asarray(boxes)
    scores = np.asarray(scores)

    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
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
        w = np.maximum(0, (x2 - x1 + 1))
        h = np.maximum(0, (y2 - y1 + 1))
        intersections = w * h

