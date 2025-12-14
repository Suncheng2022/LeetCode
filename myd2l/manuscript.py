from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

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

def BubbleSort(nums):
    """ 交换排序 - 冒泡排序, 稳定, 每次都会固定一个元素到最终位置 """
    # n = len(nums)
    # for i in range(n - 1):      # 冒泡次数
    #     flag = False
    #     for j in range(n - 1 - i):      # 冒泡范围
    #         if nums[j] > nums[j + 1]:
    #             flag = True
    #             nums[j], nums[j + 1] = nums[j + 1], nums[j]
    #     if not flag:
    #         break

    ## Again
    n = len(nums)
    for i in range(n - 1):          # 冒泡次数
        flag = False
        for j in range(n - 1 - i):  # 冒泡范围
            if nums[j] > nums[j + 1]:
                flag = True
                nums[j], nums[j + 1] = nums[j + 1], nums[j]
        if not flag:
            break

def SelectSort(nums):
    """ 简单选择排序 """
    n = len(nums)
    for i in range(n - 1):      # 选择次数
        minInd = i
        for j in range(i + 1, n):   # 选择范围 从i或i+1开始都行,没啥大影响
            if nums[j] < nums[minInd]:
                minInd = j
        if minInd != i:
            nums[minInd], nums[i] = nums[i], nums[minInd]

""" 堆排, 小顶堆 """
def HeapSort(nums):
    BuildHeap(nums)
    for i in range(len(nums) - 1, 0, -1):
        print(nums[1])      # 访问堆顶 最小值
        nums[1], nums[i] = nums[i], nums[1]
        AdjustDown(nums, 1, i - 1)

def BuildHeap(nums):
    i = len(nums) // 2
    while i >= 1:
        AdjustDown(nums, i, len(nums) - 1)
        i -= 1

def AdjustDown(nums, k, end):
    """
    Description:
        将根节点k下沉到合适的位置, 恢复堆的性质
    Args:
        nums:
        k:   待调整的根节点
        end: 待调整的最后节点
    """
    nums[0] = nums[k]       # 哨兵, 临时存放
    i = 2 * k
    while i <= end:
        if i < end and nums[i + 1] < nums[i]:   # i指向较小孩子
            i = i + 1
        if nums[i] < nums[0]:                   # i指向的元素 < 根节点, 往上浮
            nums[k] = nums[i]
            k = i
        i *= 2
    nums[k] = nums[0]       # 哨兵元素归位

# Definition for singly-linked list.
class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        """ 16.最接近的三数之和 """
        nums.sort()
        res = sum(nums[:3])
        for i in range(len(nums) - 2):
            l, r = i + 1, len(nums) - 1
            while l < r:
                _tmp = nums[i] + nums[l] + nums[r]
                res = res if abs(target - res) < abs(target - _tmp) else _tmp
                if _tmp == target:
                    return _tmp
                elif _tmp < target:
                    l += 1
                else:
                    r -= 1
        return res
    
    def sortList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        """ 148.排序链表 """
        ## 归并排序
        def mysplit(head, step):
            """ 将链表head划分为: step个节点 + 剩余部分 \n
                返回剩余部分第一个节点 """
            for _ in range(step - 1):
                if not head:
                    return
                head = head.next
            
            if head:
                res = head.next
                head.next = None
                return res
            else:
                return
        
        def mymerge(h1, h2):
            cur = dummyHead = ListNode()
            while h1 and h2:
                if h1.val < h2.val:
                    cur.next = h1
                    h1 = h1.next
                else:
                    cur.next = h2
                    h2 = h2.next
                cur = cur.next
            cur.next = h1 or h2
            return dummyHead.next

        # 边界
        if not head or not head.next:
            return head
        
        # 链表长度
        n = 0
        cur = head
        while cur:
            n += 1
            cur = cur.next
        
        dummyHead = ListNode()
        dummyHead.next = head
        
        step = 1
        while step < n:
            # 一次归排
            prev = dummyHead
            cur = prev.next

            while cur:
                left = cur
                right = mysplit(left, step)
                cur = mysplit(right, step)
                
                prev.next = mymerge(left, right)
                while prev.next:
                    prev = prev.next

            step *= 2
        return dummyHead.next

## Transformer +1. reference to myd2l/20251211_d2l.py
class PositionEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        _constant = torch.tensor(10000)

        pe = torch.zeros(max_len, d_model)          # [max_len, d_model]
        pos = torch.arange(max_len).unsqueeze(1)    # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(_constant) / d_model))       # [d_model / 2]
        pe[:, 0::2] = torch.sin(pos * div_term)     # [max_len, d_model / 2]
        pe[:, 1::2] = torch.cos(pos * div_term)     # [max_len, d_model / 2]
        pe.unsqueeze_(0)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]
    
class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, heads):
        super().__init__()
        assert d_model % heads == 0

        self.heads = heads
        self.d_k = d_model // heads

        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc_linear = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        """
        when q = k = v, it's self-attention
        when q comes from decoder, k and v comes from encoder, it's enc-dec attention
        mask: only for end-dec attention
        """
        batch = q.shape[0]
        _temp = torch.tensor(self.d_k).sqrt()

        Q = self.q_linear(q).view(batch, -1, self.heads, self.d_k).transpose(1, 2)      # [batch, heads, max_len, d_k]
        K = self.k_linear(k).view(batch, -1, self.heads, self.d_k).transpose(1, 2)      # [batch, heads, max_len, d_k]
        V = self.q_linear(v).view(batch, -1, self.heads, self.d_k).transpose(1, 2)      # [batch, heads, max_len, d_k]
        score = torch.matmul(Q, K.transpose(-2, -1)) / _temp                            # [batch, heads, max_len, max_len]
        if mask is not None:
            score.masked_fill_(mask==0, float('-inf'))
        score = score.softmax(dim=-1)
        out = torch.matmul(score, V).transpose(1, 2).reshape(batch, -1, self.heads * self.d_k)      # [batch, heads, d_model]
        out = self.fc_linear(out)
        return out

class FFN(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(.1)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, d_model, heads, d_ffn):
        super().__init__()

        self.MSA = MultiHeadAttention(d_model, heads)
        self.FFN = FFN(d_model, d_ffn)
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(.1)
    
    def forward(self, x):
        mas_out = self.MSA(x, x, x)
        x = self.ln1(x + self.dropout(mas_out))
        ffn_out = self.FFN(x)
        x = self.ln2(x + self.dropout(ffn_out))
        return x

class Encoder(nn.Module):
    def __init__(self, voc_size, max_len, d_model, heads, n):
        super().__init__()
        
        self.emb = nn.Embedding(voc_size, d_model)
        self.pos_emb = PositionEmbedding(max_len, d_model)
        self.encs = nn.ModuleList([MultiHeadAttention(d_model, heads) for _ in range(n)])
    
    def forward(self, x):
        x = self.emb(x)
        x = self.pos_emb(x)
        for enc in self.encs:
            x = enc(x, x, x)
        return x

class Transformer(nn.Module):
    def __init__(self, voc_size, max_len, d_model, heads, n):
        super().__init__()

        self.encoder = Encoder(voc_size, max_len, d_model, heads, n)
    
    def forward(self, x):
        return self.encoder(x)


if __name__ == '__main__':
    ## Transformer test
    voc_size = 6000
    batch, max_len, d_model = 1, 77, 512
    heads = 4
    d_ffn = 1024
    
    x = torch.zeros(batch, max_len, d_model)
    print(f'>>> x shape:{x.shape}')

    posEmb = PositionEmbedding(max_len, d_model)
    print(f'>>> posEmb output shape:{posEmb(x).shape}')

    MSA_block = MultiHeadAttention(d_model, heads)
    msa_out = MSA_block(q=x, k=x, v=x)        # self-attention, same q k v
    print(f'>>> mas_out shape:{msa_out.shape}')

    FFN_block = FFN(d_model, d_ffn)
    ffn_out = FFN_block(msa_out)
    print(f'>>> ffn_out shape:{ffn_out.shape}')

    encoder_layer = EncoderLayer(d_model, heads, d_ffn)
    enc_layer_out = encoder_layer(x)
    print(f'>>> enc_layer_out shape:{enc_layer_out.shape}')

    word_inds = torch.arange(max_len, dtype=torch.long)     # 输入映射为word id [之后再embedding成为token]
    encoder_tower = Encoder(voc_size, max_len, d_model, heads, n=8)
    enc_out = encoder_tower(word_inds)
    print(f'>>> enc_out shape:{enc_out.shape}')