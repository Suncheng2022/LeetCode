""" 熟悉一下, 还会吗 """
import numpy as np

# 1.0 NMS
def NMS(bbox, conf, thresh):
    """
    :param bbox: array类型, n x 4
    :param conf: array类型, n x 1
    :param thresh: 阈值, 1
    """
    bbox = np.array(bbox)       # n x 4
    conf = np.array(conf)       # n x 1

    order = conf.argsort()[::-1]      # 降序
    area = (bbox[:, 2] - bbox[:, 0]) * (bbox[:, 3] - bbox[:, 1])        # n x 1

    pick_bbox = []
    pick_conf = []
    while order.size:
        ind = order[0]
        pick_bbox.append(bbox[ind])
        pick_conf.append(conf[ind])

        # 计算 选中的bbox与剩余bbox 的交集
        x1 = np.maximum(bbox[ind, 0], bbox[order[1:], 0])
        y1 = np.maximum(bbox[ind, 1], bbox[order[1:], 1])
        x2 = np.minimum(bbox[ind, 2], bbox[order[1:], 2])
        y2 = np.minimum(bbox[ind, 3], bbox[order[1:], 3])
        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)
        intersection = w * h
        iou = intersection / (area[ind] + area[order[1:]] - intersection)

        left = np.nonzero(iou < thresh)[0]
        order = order[left]
    return pick_bbox, pick_conf

# 1.1 NMS 再实现
def NMS(bboxes, confs, thresh):
    bboxes = np.asarray(bboxes)
    confs = np.asarray(confs)

    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    orders = confs.argsort()[::-1]      # 降序
    pick_bboxes = []
    pick_confs = []
    while orders.size:
        ind = orders[0]
        pick_bboxes.append(bboxes[ind])
        pick_confs.append(confs[ind])

        # 计算 挑选框 与 剩余框 的交集
        x1s = np.maximum(bboxes[ind, 0], bboxes[orders[1:], 0])
        y1s = np.maximum(bboxes[ind, 1], bboxes[orders[1:], 1])
        x2s = np.minimum(bboxes[ind, 2], bboxes[orders[1:], 2])
        y2s = np.minimum(bboxes[ind, 3], bboxes[orders[1:], 3])
        w = np.maximum(0, x2s - x1s)
        h = np.maximum(0, y2s - y1s)
        intersect_areas = w * h
        ious = intersect_areas / (areas[ind] + areas[orders[1:]] - intersect_areas)
        
        left = np.nonzero(ious < thresh)[0]
        orders = orders[left + 1]
    return pick_bboxes, pick_confs

# 1.2 NMS 熟悉了吧
def NMS(bboxes, confs, thresh):
    bboxes = np.asarray(bboxes)
    confs = np.asarray(confs)

    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    orders = confs.argsort()[::-1]      # 降序
    pick_bboxes = []
    pick_confs = []
    while orders.size:
        ind = orders[0]
        pick_bboxes.append(bboxes[ind])
        pick_confs.append(confs[ind])
        
        # 计算iou
        x1s = np.maximum(bboxes[ind, 0], bboxes[orders[1:], 0])
        y1s = np.maximum(bboxes[ind, 1], bboxes[orders[1:], 1])
        x2s = np.minimum(bboxes[ind, 2], bboxes[orders[1:], 2])
        y2s = np.minimum(bboxes[ind, 3], bboxes[orders[1:], 3])
        w = np.maximum(0, x2s - x1s)
        h = np.maximum(0, y2s - y1s)
        intersect_areas = w * h
        ious = intersect_areas / (areas[ind] + areas[orders[1:]] - intersect_areas)
        
        # 删除
        left = np.nonzero(ious < thresh)[0]
        orders = orders[left + 1]
    return np.asarray(pick_bboxes), np.asarray(pick_confs)

# 1.3 NMS 重复
def NMS(bboxes, confs, thresh):
    bboxes = np.array(bboxes)       # [n, 4]
    confs = np.array(confs)         # [n, ]

    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])       # [n, ]
    orders = confs.argsort()[::-1]  # [n, ]
    res_bboxes = []
    res_confs = []
    while orders.size:
        ind = orders[0]
        res_bboxes.append(bboxes[ind])
        res_confs.append(confs[ind])

        x1 = np.maximum(bboxes[ind, 0], bboxes[orders[1:], 0])        # [, ] 与 [n - 1, ] 计算 广播机制 >>> [n - 1, ]
        y1 = np.maximum(bboxes[ind, 1], bboxes[orders[1:], 1])
        x2 = np.minimum(bboxes[ind, 2], bboxes[orders[1:], 2])
        y2 = np.minimum(bboxes[ind, 3], bboxes[orders[1:], 3])
        w = np.maximum(0, x2 - x1)  # [n - 1, ]
        h = np.maximum(0, y2 - y1)
        intersect_areas = w * h     # [n - 1, ]
        ious = intersect_areas / np.maximum(1e-6, areas[ind] + areas[orders[1:]] - intersect_areas)     # [n - 1, ]
        left = np.nonzero(ious < thresh)[0]
        orders = orders[left + 1]
    return res_bboxes, res_confs

# 2.0 iou
def IOU(bbox1, bbox2):
    """ 只计算2个bbox的iou """
    bbox1 = np.asarray(bbox1)
    bbox2 = np.asarray(bbox2)

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    x1 = np.maximum(bbox1[0], bbox2[0])
    y1 = np.maximum(bbox1[1], bbox2[1])
    x2 = np.minimum(bbox1[2], bbox2[2])
    y2 = np.minimum(bbox1[3], bbox2[3])
    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)
    intersect_erea = w * h
    iou = intersect_erea / np.maximum(area1 + area2 - intersect_erea, 1e-6)
    
    return iou

# 2.1 iou
def IOU(bbox1, bbox2):
    """ 同2.0 """
    bbox1 = np.asarray(bbox1, dtype=np.float32)
    bbox2 = np.asarray(bbox2, dtype=np.float32)
    
    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    x1 = np.maximum(bbox1[0], bbox2[0])
    y1 = np.maximum(bbox1[1], bbox2[1])
    x2 = np.minimum(bbox1[2], bbox2[2])
    y2 = np.minimum(bbox1[3], bbox2[3])
    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)
    intersect_area = w * h
    iou = intersect_area / np.maximum(area1 + area2 - intersect_area, 1e-6)

    return iou

# 2.1 重复 一对一
def IOU(bbox1, bbox2):
    bbox1 = np.array(bbox1)     # [4, ]
    bbox2 = np.array(bbox2)     # [4, ]

    area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])

    xy1 = np.maximum(bbox1[:2], bbox2[:2])      # [2, ]
    xy2 = np.minimum(bbox1[2:], bbox2[2:])      # [2, ]
    wh = np.maximum(0, xy2 - xy1)       # [2, ]
    intersect_area = wh[0] * wh[1]      # []
    iou = intersect_area / (area1 + area2 - intersect_area)     # []
    return iou

# 2.2 iou 多个bbox
def IOU(bboxes1, bboxes2):
    """
    :param bboxes1: m x 4
    :param bboxes2: n x 4
    """
    bboxes1 = np.asarray(bboxes1)       # m x 4
    bboxes2 = np.asarray(bboxes2)       # n x 4

    areas1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])      # [m, ]
    areas2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])      # [n, ]

    # 计算交集
    x1 = np.maximum(bboxes1[:, None, 0], bboxes2[:, 0])     # [m, 1] 与 [n, ] 广播机制 ===> [m, n]. 因为要bboxes1的每个框都要和bboxes2的每个框 两两计算
    y1 = np.maximum(bboxes1[:, None, 1], bboxes2[:, 1])
    x2 = np.minimum(bboxes1[:, None, 2], bboxes2[:, 2])
    y2 = np.minimum(bboxes1[:, None, 3], bboxes2[:, 3])
    w = np.maximum(0, x2 - x1)
    h = np.maximum(0, y2 - y1)
    intersect_areas = w * h
    ious = intersect_areas / np.maximum(areas1[:, None] + areas2 - intersect_areas, 1e-6)
    return ious

# 2.3 iou 重复2.2
def IOU(bboxes1, bboxes2):
    bboxes1 = np.array(bboxes1)     # [m, 4]
    bboxes2 = np.array(bboxes2)     # [n, 4]

    areas1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])      # [m, ]
    areas2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])      # [n, ]

    x1 = np.maximum(bboxes1[:, None, 0], bboxes2[:, 0])     # [m, n] 广播机制, 还不是很明了
    y1 = np.maximum(bboxes1[:, None, 1], bboxes2[:, 1])
    x2 = np.minimum(bboxes1[:, None, 2], bboxes2[:, 2])
    y2 = np.minimum(bboxes1[:, None, 3], bboxes2[:, 3])
    w = np.maximum(0, x2 - x1)      # [m, n]
    h = np.maximum(0, y2 - y1)
    intersect_area = w * h          # [m, n]
    ious = intersect_area / np.maximum(areas1[:, None] + areas2 - intersect_area, 1e-6)
    
    return ious

# 2.4 iou 自己改进一下, 同时求xy
def IOU(bboxes1, bboxes2):
    bboxes1 = np.array(bboxes1)     # [m, 4]
    bboxes2 = np.array(bboxes2)     # [n, 4]

    areas1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])      # [m, ]
    areas2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])      # [n, ]

    xy1 = np.maximum(bboxes1[:, None, :2], bboxes2[:, :2])      # [m, n, 2]
    xy2 = np.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:])
    w = np.maximum(0, xy2[:, :, 0] - xy1[:, :, 0])              # [m, n]
    h = np.maximum(0, xy2[:, :, 1] - xy1[:, :, 1])
    intersect_areas = w * h                                     # [m, n]
    ious = intersect_areas / np.maximum(areas1[:, None] + areas2 - intersect_areas, 1e-6)       # [m, n]

    return ious

# 2.4 iou 再改进一下
def IOU(bboxes1, bboxes2):
    bboxes1 = np.array(bboxes1)     # [m, 4]
    bboxes2 = np.array(bboxes2)     # [n, 4]

    areas1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])      # [m, ]
    areas2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])      # [n, ]

    xy1 = np.maximum(bboxes1[:, None, :2], bboxes2[:, :2])      # [m, n, 2]
    xy2 = np.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:])
    wh = np.maximum(0, xy2 - xy1)       # [m, n, 2]
    intersect_areas = wh[..., 0] * wh[..., 1]       # [m, n]
    ious = intersect_areas / np.maximum(1e-6, areas1[:, None] + areas2 - intersect_areas)       # [m, n]
    return ious

# 2.4 iou 多对多
def IOU(bboxes1, bboxes2):
    bboxes1 = np.asarray(bboxes1)       # [m, 4]
    bboxes2 = np.asarray(bboxes2)       # [n, 4]

    areas1 = (bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1])      # [m, ]
    areas2 = (bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1])      # [n, ]

    xy1 = np.maximum(bboxes1[:, None, :2], bboxes2[:, :2])       # [m, 1, 2]与[n, 2] 广播机制 >>> [m, n, 2]
    xy2 = np.minimum(bboxes1[:, None, 2:], bboxes2[:, 2:])
    wh = np.maximum(0, xy2 - xy1)       # [m, n, 2]
    intersect_areas = wh[..., 0] * wh[..., 1]       # [m, n]
    ious = intersect_areas / np.maximum(1e-6, areas1[:, None] + areas2 - intersect_areas)       # [m, 1] 与 [n, ] 广播机制 >>> [m, n]
    
    return ious