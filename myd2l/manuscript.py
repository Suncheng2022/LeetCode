import torch
import numpy as np

from torch import nn


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

# ------------------------------ 以上方法感觉实现还不错，保留着吧 ----------------------------------

# 以下重新写一遍 判断三角形是否重叠/相交
def orientation(p1, p2, p3):
    """ 判断p3在直线p1、p2的哪一侧，或共线
        p1: [x1,y1]
        p2: [x2,y2]
        p3: [x3,y3] """
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    # 平面上三点的面积量
    # S > 0 点在直线左侧
    # S = 0 点在直线上
    # S < 0 点在直线右侧
    S = (x1 - x3) * (y2 - y3) - (y1 - y3) * (x2 - x3)
    if S == 0:
        return 0                        # 0表示共线
    return 1 if S > 0 else 2            # 1表示p3在直线p1p2左边，2表示右边

def on_segement(p, r, q):
    """ 与orientation()结合使用-->判断r是否在 线段pq 上。
        orientation()只能判断点在哪侧或共线，若共线再结合on_segement()就能判断 点 是否在 线段 上 """
    if min(p[0], q[0]) <= r[0] <= max(p[0], q[0]) \
        and \
        min(p[1], q[1]) <= r[1] <= max(p[1], q[1]):
        return True         # 返回True，再结合orientation()就能判断 点 是否在 线段 上
    return False

def on_intersection(p1, p2, p3, p4):
    """ 判断2条线段 p1p2、p3p4 是否相交 """
    o1 = orientation(p1, p2, p3)
    o2 = orientation(p1, p2, p4)
    o3 = orientation(p3, p4, p1)
    o4 = orientation(p3, p4, p2)

    if o1 == 0 and on_segement(p1, p3, p2):     # p3在线段p1p2上，即相交，返回True
        return True
    elif o2 == 0 and on_segement(p1, p4, p2):   # 同上
        return True
    elif o3 == 0 and on_segement(p3, p1, p4):   # 同上
        return True
    elif o4 == 0 and on_segement(p3, p2, p4):   # 同上
        return True
    elif o1 != o2 and o3 != o4:                 # 注意，是and，or的话是不能判断出 线段 是否相交的
        return True
    return False                                # 以上均不符合，则p1p2、p3p4线段不相交


"""
p1, p2, p3 = triangle1
p4, p5, p6 = triangle2
if on_intersection(p1, p2, p4, p5) or on_intersection(p1, p2, p5, p6) or on_intersection(p1, p2, p4, p6):
    return True
"""


class MultiHeadAttention(nn.Module):
    """ 多头自注意力第三方实现，大概思路是没问题的 """
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = self.embed_dim // self.num_heads

        self.qkv_proj = nn.Linear(in_features=self.embed_dim, out_features=3 * self.embed_dim)
        self.out_proj = nn.Linear(in_features=self.embed_dim, out_features=self.embed_dim)

    def forward(self, x):
        """ 假设 x:[s,b,d] """
        seq_length, batch_size = x.shape[:2]
        qkv = self.qkv_proj(x)                          # [s,b,3*d]
        q, k, v = qkv.chunk(3, dim=-1)                  # [s,b,d]
        q = q.contiguous().view(seq_length, batch_size * self.num_heads, self.head_dim).permute(1, 0, 2)    # [b*h,s,h_d]
        k = k.contiguous().view(seq_length, batch_size * self.num_heads, self.head_dim).permute(1, 0, 2)    # [b*h,s,h_d]
        v = v.contiguous().view(seq_length, batch_size * self.num_heads, self.head_dim).permute(1, 0, 2)    # [b*h,s,h_d]

        attn = (q @ k.permute(0, 2, 1) / torch.sqrt(self.head_dim)).softmax(dim=-1)                         # [b*h,s,s]
        v = attn @ v                                                                                        # [b*h,s,h_d]
        v = v.permute(1, 0, 2).contiguous().view(seq_length, batch_size, -1)                                # [s,b,d]
        v = self.out_proj(v)
        return v

## 10.30    你好慢, 你好墨迹
def NMS(bboxes, confs, t):
    bboxes = np.asarray(bboxes)     # [n, 4]
    confs = np.asarray(confs)       # [n]

    pick_bboxes = []
    pick_confs = []

    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])   # [n]
    orders = confs.argsort()[::-1]  # [n]
    while len(orders):
        ind = orders[0]
        pick_bboxes.append(bboxes[ind])
        pick_confs.append(confs[ind])

        xmin = np.maximum(bboxes[ind, 0], bboxes[orders[1:], 0])    # [n - 1]
        ymin = np.maximum(bboxes[ind, 1], bboxes[orders[1:], 1])    # [n - 1]
        xmax = np.minimum(bboxes[ind, 2], bboxes[orders[1:], 2])    # [n - 1]
        ymax = np.minimum(bboxes[ind, 3], bboxes[orders[1:], 3])    # [n - 1]
        inter_areas = np.maximum(0, xmax - xmin) * np.maximum(0, ymax - ymin)   # [n - 1]

        ious = inter_areas / (areas[ind] + areas[orders[1:]] - inter_areas)     # [n - 1]
        valid_inds = np.where(ious < t)[0]      # [n - 1]
        orders = orders[1:][valid_inds]
    return pick_bboxes, pick_confs