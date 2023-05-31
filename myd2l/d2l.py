import numpy as np


# 1.nms
def num(dets, thresh):
    """
    此仅对一个类别nms，使用时要分别对各个类别nms
    :param dets: n个[x1,y1,x2,y2,score]
    :param thresh: 最大score的bbox与剩下所有bbox求IOU，大于此阈值的框删掉，因为这很可能是预测的同一目标
    :return:
    """
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    orders = scores.argsort()[::-1]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    pick = []
    while orders.size > 0:
        i = orders[0]
        pick.append(i)

        xx1 = np.maximum(x1[i], x1[orders[1:]])
        yy1 = np.maximum(y1[i], y1[orders[1:]])
        xx2 = np.minimum(x2[i], x2[orders[1:]])
        yy2 = np.minimum(y2[i], y2[orders[1:]])

        w = np.maximum(0., xx2 - xx1 + 1)
        h = np.maximum(0., yy2 - yy1 + 1)
        inter = w * h

        ious = inter / (areas[i] + areas[orders[1:]] - inter)
        inds = np.where(ious <= thresh)[0]  # 自动调用nonzero()-->返回的是好几个array[取决于维度大小]
        orders = orders[inds + 1]   # 因为ious数组长度比orders少一个——开头的0索引、即自己，所以这里所有的下标都加1
    return pick

# 1.1nms
def NMS(boxes, confs, iou_threshold):
    """
    https://zhuanlan.zhihu.com/p/37489043
    :param boxes:xyxy坐标
    :param confs:置信度
    :param iou_threshold:
    :return: Rest boxes after nms operation, [boxes], [confs]
    """
    # boxes为空，直接返回
    if len(boxes) == 0:
        return [], []

    # 转为numpy
    boxes = np.array(boxes)
    score = np.array(confs)

    # 保存NMS筛选的结果
    picked_boxes = []
    picked_score = []

    # 计算boxes面积
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    # 按置信度排序
    orders = score.argsort()

    # 开始迭代执行NMS算法
    while len(orders):
        # 最大置信度索引
        index = orders[-1]
        # 放入结果
        picked_boxes.append(boxes[index])
        picked_score.append(score[index])

        # 计算intersection区域坐标
        # np.max()计算序列的最值，np.maximum()2个序列逐位比较取较大值。参考：https://blog.csdn.net/lanchunhui/article/details/52700895
        x1 = np.maximum((boxes[index:, 0], boxes[orders[:-1], 0]))
        y1 = np.maximum((boxes[index:, 1], boxes[orders[:-1], 1]))
        x2 = np.minimum((boxes[index:, 2], boxes[orders[:-1], 2]))
        y2 = np.minimum((boxes[index:, 3], boxes[orders[:-1], 3]))
        # 计算intersection区域面积
        w = np.maximum((0, x2 - x1 + 1))
        h = np.maximum((0, y2 - y1 + 1))
        intersection = w * h

        # 计算iou
        iou = intersection / (areas[index] + areas[orders[:-1]] - intersection)

        # iou超过阈值的舍弃
        left = np.where(iou < iou_threshold)
        orders = orders[left]
    return picked_boxes, picked_score


# 1.2nms
def NMS(bboxes, confs, iou_threshold):
    bboxes = np.array(bboxes)
    confs = np.array(confs)

    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    orders = confs.argsort()

    res_bbox = []
    res_conf = []
    while len(orders):
        index = orders[-1]
        res_bbox.append(bboxes[index])
        res_conf.append(confs[index])

        x1 = np.maximum(bboxes[index, 0], bboxes[orders[:-1], 0])
        y1 = np.maximum(bboxes[index, 1], bboxes[orders[:-1], 1])
        x2 = np.minimum(bboxes[index, 2], bboxes[orders[:-1], 2])
        y2 = np.minimum(bboxes[index, 3], bboxes[orders[:-1], 3])

        # 万一iou为0呢
        intersection = np.maximum(0, x2 - x1 + 1) * np.maximum(0, y2 - y1 + 1)

        ious = intersection / (areas[index] + areas[orders[:-1]] - intersection)

        left = np.where(ious < iou_threshold)
        orders = orders[left]
    return res_bbox, res_conf


# 2.iou
def cal_iou(bbox1, bbox2):
    """
    计算bbox之间的iou  JustRepeat.py实现的更好些，增加了注释
    :param bbox1: [x1,y1,x2,y2]
    :param bbox2: [x1,y1,x2,y2]
    :return: 
    """
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox2

    s1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    s2 = (xmax2 - xmin2) * (ymax2 - ymin2)

    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    w = max(0., xmax - xmin)
    h = max(0., ymax - ymin)

    inter = w * h
    iou = inter / (s1 + s2 - inter)
    return iou


def cal_iou01(bbox1, bbox2):
    """ 计算iou：bbox1 bbox2各代表1个bbox """
    # 获取bbox1、bbox2坐标
    xmin1, ymin1, xmax1, ymax1 = bbox1
    xmin2, ymin2, xmax2, ymax2 = bbox1
    # 计算bbox1、bbox2面积
    area1 = (xmax1 - xmin1 + 1) * (ymax1 - ymin1 + 1)   # 这里计算的是像素的个数，所以+1
    area2 = (xmax2 - xmin2 + 1) * (ymax2 - ymin2 + 1)
    # 计算bbox1、bbox2相交区域坐标
    x1 = max(xmin1, xmin2)
    y1 = max(ymin1, ymin2)
    x2 = min(xmax1, xmax2)
    y2 = min(ymax1, ymax2)
    w = max(0, x2 - x1 + 1)     # 这里+1同样是计算像素数目
    h = max(0, y2 - y1 + 1)
    inter = w * h
    # 计算iou
    iou = inter / (area1 + area2 - inter)
    return iou


def cal_iou02(bboxes1, bboxes2):
    """ 计算iou，bboxes1、bboxes2代表多个boxes """
    A = bboxes1.shape[0]
    B = bboxes2.shape[0]

    areas1 = (bboxes1[:, 2] - bboxes1[:, 0] + 1) * (bboxes1[:, 3] - bboxes1[:, 1])
    areas2 = (bboxes2[:, 2] - bboxes2[:, 0] + 1) * (bboxes2[:, 3] - bboxes2[:, 1])

    xy_min = np.maximum(bboxes1[:, np.newaxis, :2].repeat(B, axis=1),
                        np.broadcast_to(B[:, :2], (A, B, 2)))
    xy_max = np.minimum(bboxes1[:, np.newaxis, 2:].repeat(B, axis=1),
                        np.broadcast_to(bboxes2[:, 2:], (A, B, 2)))

    inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)
    inter = inter[:, :, 0] * inter[:, :, 1]
    ious = inter / (areas1[:, np.newaxis].repeat(B, axis=1) +
                    areas2[:, np.newaxis].repeat(A, axis=1) -
                    inter)


def cal_iou02(bboxes1, bboxes2):
    """ 再写一遍 多对多 """
    A = bboxes1.shape[0]
    B = bboxes2.shape[0]

    # 计算交集坐标
    xy_min = np.maximum(bboxes1[:, np.newaxis, :2].repeat(B, axis=1),   # [A, 2]->[A, B, 2]
                        np.broadcast_to(bboxes2[:, :2], (A, B, 2)))     # [A, 2]->[A, B, 2]
    xy_max = np.minimum(bboxes1[:, np.newaxis, 2:].repeat(B, axis=1),
                        np.broadcast_to(bboxes2[:, 2:], (A, B, 2)))

    # 计算交集面积
    inter = np.clip(xy_max - xy_min, a_min=0, a_max=np.inf)     # [A, B, 1]
    inter = inter[:, :, 0] * inter[:, :, 1]

    # 计算bboxes1、bboxes2面积
    # [A, 1]->[A, B, 1]
    areas1 = ((bboxes1[:, 2] - bboxes1[:, 0]) * (bboxes1[:, 3] - bboxes1[:, 1]))[:, np.newaxis, :].repeat(B, axis=1)
    # [B, 1]->[A, B, 1]
    areas2 = ((bboxes2[:, 2] - bboxes2[:, 0]) * (bboxes2[:, 3] - bboxes2[:, 1]))[np.newaxis, :, :].repeat(A, axis=0)

    # 计算iou
    ious = inter / (areas1 + areas2 - inter)    # [A, B, 1]
    return ious