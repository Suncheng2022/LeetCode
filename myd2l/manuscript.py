import torch
import numpy as np


def IoU_batch(bboxes1, bboxes2):
    """ 批量计算IoU.
        bboxes1: tensor [N, 4]
        bboxes2: tensor [M, 4] """
    xy_min = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])        # [N, M, 2]
    xy_max = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])        # [M, N, 2]

    wh = (xy_max - xy_min).clamp(min=0)                             # [N, M, 2]
    inter_areas = wh[:, :, 0] * wh[:, :, 1]                         # [N, M]

    areas1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])      # [N,]
    areas2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])      # [M,]

    IoUs = inter_areas / (areas1[:, None] + areas2 - inter_areas)

    return IoUs


def IoU_batch(bboxes1, bboxes2):
    """ 再写一遍
        bboxes1: tensor [N, 4]
        bboxes2: tensor [M, 4] """
    xy_min = torch.max(bboxes1[:, None, :2], bboxes2[:, :2])        # [N, M, 2]
    xy_max = torch.min(bboxes1[:, None, 2:], bboxes2[:, 2:])        # [N, M, 2]
    wh = (xy_max - xy_min).clamp(min=0)                             # [N, M, 2]
    inter_areas = wh[:, :, 0] * wh[:, :, 1]                         # [N, M]

    areas1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])      # [N,]
    areas2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])      # [M,]

    IoUs = inter_areas / (areas1[:, None] + areas2 - inter_areas)
    return IoUs


def NMS(bboxes, confs, IOU_THRESHOLD):
    """ NMS算法
        bboxes: [N, 4]
        confs: [N, ] """
    bboxes = np.asarray(bboxes)
    confs = np.asarray(confs)

    areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)  # [N, ]

    orders = confs.argsort()
    res = []
    while orders:
        index = orders[-1]                                              # [N, ]
        res.append([bboxes[index], confs[index]])

        xmin = np.maximum(bboxes[index, 0], bboxes[orders[:-1], 0])     # [N-1, ]
        ymin = np.maximum(bboxes[index, 1], bboxes[orders[:-1], 1])
        xmax = np.minimum(bboxes[index, 2], bboxes[orders[:-1], 2])
        ymax = np.minimum(bboxes[index, 3], bboxes[orders[:-1], 3])

        w = np.maximum(0, xmax - xmin + 1)                              # [N-1, ]
        h = np.maximum(0, ymax - ymin + 1)
        inter_areas = w * h                                             # [N-1, ]

        IoUs = inter_areas / (areas[index] + areas[orders[:-1]] - inter_areas)              # [N-1, ]
        left = np.where(IoUs < IOU_THRESHOLD)
        orders = orders[left]
    return res


def NMS(bboxes, confs, IOU_THRESHOLD):
    """ 再写一遍 NMS算法
        bboxes: [N, 4]
        confs: [N, ] """
    bboxes = np.asarray(bboxes)
    confs = np.asarray(confs)

    areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)
    orders = confs.argsort()
    res = []

    while orders:
        index = orders[-1]
        res.append([bboxes[index], confs[index]])

        xmin = np.maximum(bboxes[index, 0], bboxes[orders[:-1], 0])             # [N-1, ]
        ymin = np.maximum(bboxes[index, 1], bboxes[orders[:-1], 1])
        xmax = np.minimum(bboxes[index, 2], bboxes[orders[:-1], 2])
        ymax = np.minimum(bboxes[index, 3], bboxes[orders[:-1], 3])

        w = np.maximum(0, xmax - xmin + 1)                                      # [N-1, ]
        h = np.maximum(0, ymax - ymin + 1)
        inter_areas = w * h

        IoUs = inter_areas / (areas[index] + areas[orders[:-1]] - inter_areas)  # [N-1, ]
        left = np.where(IoUs < IOU_THRESHOLD)
        orders = orders[left]
    return res

# 以上方法感觉实现还不错，保留着吧
