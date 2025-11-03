"""
08.04   将近一月, 加油!
        参考 实现最正常的版本 吧
"""
import torch
import torch.nn.functional as F
import numpy as np

def NMS(bboxes, confs, threshold):
    """ 带batch维度的NMS """
    pick_bboxes = []
    pick_confs = []

    if len(bboxes) == 0:
        return pick_bboxes, pick_confs
    
    bboxes = np.array(bboxes)
    confs = np.array(confs)

    orders = confs.argsort()        # 返回升序排序的索引
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])
    while len(orders):
        ind = orders[-1]
        pick_bboxes.append(bboxes[ind])
        pick_confs.append(confs[ind])

        x1 = np.maximum((bboxes[ind, 0], bboxes[orders[:-1], 0]))
        y1 = np.maximum((bboxes[ind, 1], bboxes[orders[:-1], 1]))
        x2 = np.minimum((bboxes[ind, 2], bboxes[orders[:-1], 2]))
        y2 = np.minimum((bboxes[ind, 3], bboxes[orders[:-1], 3]))
        w = np.maximum((0, x2 - x1))
        h = np.maximum((0, y2 - y1))
        intersections = w * h
        
        ious = intersections / (areas[ind] + areas[orders[:-1]] - intersections)

        rest = np.where(ious < threshold)
        orders = orders[rest]
    
    return pick_bboxes, pick_confs

def NMS(bboxes, confs, threshold):
    """ 接近正常版本 """
    if len(bboxes) == 0:
        return [], []

    pick_bboxes, pick_confs = [], []

    bboxes = np.array(bboxes)
    confs = np.array(confs)

    areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)
    orders = confs.argsort()
    while len(orders):
        ind = orders[-1]
        pick_bboxes.append(bboxes[ind])
        pick_confs.append(confs[ind])

        x1 = np.maximum(bboxes[ind, 0], bboxes[orders[:-1], 0])
        y1 = np.maximum(bboxes[ind, 1], bboxes[orders[:-1], 1])
        x2 = np.minimum(bboxes[ind, 2], bboxes[orders[:-1], 2])
        y2 = np.minimum(bboxes[ind, 3], bboxes[orders[:-1], 3])
        w = np.maximum(0, x2 - x1 + 1)
        h = np.maximum(0, y2 - y1 + 1)
        intersections = w * h

        ious = intersections / (areas[ind] + areas[orders[:-1]] - intersections)

        rest = np.where(ious < threshold)[0]
        orders = orders[:-1][rest]
    
    return pick_bboxes, pick_confs

def NMS(bboxes, confs, threshold):
    """ 更接近正常的版本 """
    if len(bboxes) == 0:
        return [], []
    
    pick_bboxes, pick_confs = [], []

    bboxes = np.array(bboxes)
    confs = np.array(confs)

    areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)
    orders = confs.argsort()[::-1]      # 改为降序排序 后面方便更新
    while len(orders):
        ind = orders[0]
        pick_bboxes.append(bboxes[ind])
        pick_confs.append(confs[ind])

        x1 = np.maximum(bboxes[ind, 0], bboxes[orders[1:], 0])
        y1 = np.maximum(bboxes[ind, 1], bboxes[orders[1:], 1])
        x2 = np.minimum(bboxes[ind, 2], bboxes[orders[1:], 2])
        y2 = np.minimum(bboxes[ind, 3], bboxes[orders[1:], 3])
        w = np.maximum(0, x2 - x1 + 1)
        h = np.maximum(0, y2 - y1 + 1)
        intersections = w * h

        ious = intersections / (areas[ind] + areas[orders[1:]] - intersections)
        
        rest = np.where(ious < threshold)[0]
        orders = orders[rest + 1]
    return pick_bboxes, pick_confs

def NMS(bboxes, confs, threshold):
    # Again
    pick_bboxes = []
    pick_confs = []
    if len(bboxes) == 0:
        return pick_bboxes, pick_confs
    
    bboxes = np.array(bboxes)
    confs = np.array(confs)

    areas = (bboxes[:, 2] - bboxes[:, 0] + 1) * (bboxes[:, 3] - bboxes[:, 1] + 1)
    orders = confs.argsort()[::-1]
    while orders:
        ind = orders[0]
        pick_bboxes.append(bboxes[ind])
        pick_confs.append(confs[ind])

        x1 = np.maximum(bboxes[ind, 0], bboxes[orders[1:], 0])
        y1 = np.maximum(bboxes[ind, 1], bboxes[orders[1:], 1])
        x2 = np.minimum(bboxes[ind, 2], bboxes[orders[1:], 2])
        y2 = np.minimum(bboxes[ind, 3], bboxes[orders[1:], 3])
        w = np.maximum(0, x2 - x1 + 1)
        h = np.maximum(0, y2 - y1 + 1)
        intersections = w * h

        ious = intersections / (areas[ind] + areas[orders[1:]] - intersections)

        res = np.where(ious < threshold)[0]
        orders = orders[res + 1]
    return pick_bboxes, pick_confs

## ==== Transformer
import torch
import torch.nn as nn
import torch.nn.functional as F

class PositionEncoding(nn.Module):
    def __init__(self, max_len, dim):
        super().__init__()
        pe = torch.zeros(max_len, dim)                                              # [max_len, dim]
        position = torch.arange(0, max_len).unsqueeze(1)                            # [max_len, 1]
        div_term = torch.exp(torch.arange(0, dim, 2) * (-torch.log(10000) / dim))   # [dim // 2, ]  系数因子的分母部分
        pe[:, 0::2] = torch.sin(position * div_term)                                # [max_len, dim // 2]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)                                                        # [1, max_len, dim] 增加batch维度, 因为前向传入的x形状[batch, max_len, dim]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.shape[1]]

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc_linear = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v):
        """
        q: [batch, max_len, d_model]
        k:
        v:
        """
        batch = q.shape[0]
        Q = self.q_linear(q).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)  # [batch, num_heads, max_len, d_k]
        K = self.k_linear(k).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(v).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        score = torch.matmul(Q, K.transpose(-2, -1)) / self.d_k.sqrt()                  # [batch, num_heads, max_len, max_len]
        attn = score.softmax(dim=-1)
        out = torch.matmul(attn, V)                                                     # [batch, num_heads, max_len, d_k]
        out = out.transpose(1, 2).view(batch, -1, self.num_heads * self.d_k)            # [batch, max_len, d_model]
        out = self.fc_linear(out)
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, d_model):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_linear = nn.Linear(d_model, d_model)
        self.k_linear = nn.Linear(d_model, d_model)
        self.v_linear = nn.Linear(d_model, d_model)
        self.fc_linear = nn.Linear(d_model, d_model)
    
    def forward(self, q, k, v, mask=None):
        # BSD
        # mask: 适配解码器
        batch = q.shape[0]
        Q = self.q_linear(q).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)      # [batch, num_heads, max_len, d_k]
        K = self.k_linear(k).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        V = self.v_linear(v).view(batch, -1, self.num_heads, self.d_k).transpose(1, 2)
        score = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.d_k, dtype=torch.float32))  # [batch, num_heads, max_len, max_len]
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))
        attn = score.softmax(dim=-1)
        out = torch.matmul(attn, V)     # [batch, num_heads, max_len, max_len]
        out = out.transpose(1, 2).view(batch, -1, self.num_heads * self.d_k)
        out = self.fc_linear(out)
        return out

class FFN(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()
        self.linear_1 = nn.Linear(d_model, d_ffn)
        self.linear_2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(.1)
    
    def forward(self, x):
        return self.linear_2(self.dropout(F.relu(self.linear_1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, num_heads, d_model, d_ffn):
        super().__init__()
        self.mas = MultiHeadAttention(num_heads=num_heads, d_model=d_model)
        self.ffn = FFN(d_model=d_model, d_ffn=d_ffn)
        self.norm_1 = nn.LayerNorm(d_model)
        self.norm_2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(.1)
    
    def forward(self, x):
        msa_out = self.mas(x)
        x = self.norm_1(x + self.dropout(msa_out))
        ffn_out = self.ffn(x)
        x = self.norm_2(x + self.dropout(ffn_out))
        return x

class Encoder(nn.Module):
    def __init__(self, voc_size, dim, max_len, num_heads, d_ffn):
        super().__init__()
        self.emb = nn.Embedding(voc_size, dim)
        self.pos = PositionEncoding(max_len, dim)
        self.encs = nn.ModuleList([EncoderLayer(num_heads, dim, d_ffn) for _ in range(6)])
    
    def forward(self, x):
        x = self.emb(x)
        x = self.pos(x)
        for enc in self.encs:
            x = enc(x)
        return x

class Transformer(nn.Module):
    def __init__(self, voc_size, dim, max_len, num_heads, d_ffn):
        super().__init__()
        self.enc = Encoder(voc_size, dim, max_len, num_heads, d_ffn)
    
    def forward(self, x):
        return self.enc(x)
    
## ==== CLIP 损失
'''
# CLIP对比损失实现, 伪代码
def forward(self, batch):
    img_feats = self.ImgEnoder(batch['image'])     # [batch, d_model]
    text_feats = self.TextEncoder(batch['text'])   # [batch, d_model]
    img_emb = self.ImgEmbedding(img_feats)         # [batch, d_emb]
    text_emb = self.TextEmbedding(text_feats)      # [batch, d_emb]
    
    # == CLIP应该还对特征做一个归一化, 这样下面计算相似度时才准确, 不会受向量长度影响了
    img_emb = F.normalize(img_emb, p=2, dim=-1)    # [batch, d_emb]
    text_emb = F.normalize(text_emb, p=2, dim=-1)  # [batch, d_emb]
    
    logits_per_img = (img_emb @ text_emb.T) / temperature  # [batch, batch], 温度也可学习, 控制softmax的平滑程度. 对于每一行: 一张图像 vs 所有文本的相似度
    logits_per_text = logits_per_img.T             #                                                         对于每一行: 一个文本 vs 所有图像的相似度

    targets = torch.arrange(len(batch['image']))   # [batch, ]

    loss_img = F.cross_entropy(logits_per_img, targets)
    loss_text = F.cross_entropy(logits_per_text, targets)
    loss = (loss_img + loss_text) / 2
    return loss
'''

## ==== RGB KL loss, GPT生成
def differentiable_histogram(x, min=0, max=1, bins=64, sigma=1e-3):
    edges = torch.linspace(min, max, bins)                  # [bins]
    print(f'>>> edges.shape: {edges.shape}')

    x = x.unsqueeze(-1)                                     # [B,C,H,W,1]
    print(f'>>> x.shape: {x.shape}')

    dist = torch.exp(-0.5 * ((x - edges) / sigma) ** 2)     # [B,C,H,W,bins] 距离近 --> 1, 距离远 --> 0
    print(f'>>> dist.shape: {dist.shape}')

    hist = dist.sum(dim=(2, 3))                             # [B,C,bins] 将空间对bins的贡献统计出来
    print(f'>>> hist.shape: {hist.shape}')
    return hist / (hist.sum(-1, keepdim=True) + 1e-8)

def kl_loss_rgb(pred, tar):
    P = differentiable_histogram(pred)      # [B, C, bins]
    Q = differentiable_histogram(tar)       # [B, C, bins]

    loss = 0
    for c in range(3):
        loss += (P[:, c, :] * torch.log(P[:, c, :] / Q[:, c, :])).sum(dim=1).mean()
    return loss

## a demo of LN in NLP
def testLN():
    x = torch.tensor([
        # 样本1的3个token，每个token有4个特征
        [[1.0, 2.0, 3.0, 4.0],
        [5.0, 6.0, 7.0, 8.0],
        [9.0, 10.0, 11.0, 12.0]],
        # 样本2的3个token
        [[13.0, 14.0, 15.0, 16.0],
        [17.0, 18.0, 19.0, 20.0],
        [21.0, 22.0, 23.0, 24.0]]
    ])
    print(f'>>> x.shape:{x.shape}')             # [2, 3, 4]

    B, T, C = x.shape
    ln = nn.LayerNorm(normalized_shape=C, eps=1e-5, elementwise_affine=False)   # gamma/beta没学呢,初始化是1和0
    output = ln(x)
    print(f'>>> output:\n{output}')
    
    token = x[0, 0]                             # 取token
    print(f'>>> token.shape:{token.shape}')     # [4]
    mean = token.mean()
    var = token.var(unbiased=False)             # LN使用有偏方差 即除以C
    eps = ln.eps
    gamma, beta = ln.weight, ln.bias
    output_manual = (token - mean) / (var + eps).sqrt()
    print(f'>>> output_manual:\n{output_manual}')

if __name__ == '__main__':
    testLN()