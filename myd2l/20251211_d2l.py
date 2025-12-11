import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

## NMS
def NMS(bboxes, confs, t):
    """
    Args:
        bboxes: [n, 4]
        confs: [n, ]
        t: 常数
    """
    bboxes = np.array(bboxes)
    confs = np.array(confs)

    pick_bboxes = []
    pick_confs = []
    if len(bboxes) == 0:
        return pick_bboxes, pick_confs
    
    areas = (bboxes[:, 2] - bboxes[:, 0]) * (bboxes[:, 3] - bboxes[:, 1])   # [n, ]
    orders = confs.argsort()[::-1]                                          # [n, ] 降序
    while orders.size:
        ind = orders[0]

        pick_bboxes.append(bboxes[ind])
        pick_confs.append(confs[ind])

        x1 = np.maximum(bboxes[ind, 0], bboxes[orders[1:], 0])              # [n - 1, ]
        y1 = np.maximum(bboxes[ind, 1], bboxes[orders[1:], 1])              # [n - 1, ]
        x2 = np.minimum(bboxes[ind, 2], bboxes[orders[1:], 2])              # [n - 1, ]
        y2 = np.minimum(bboxes[ind, 3], bboxes[orders[1:], 3])              # [n - 1, ]
        w = np.maximum(0, x2 - x1)
        h = np.maximum(0, y2 - y1)
        inter_areas = w * h                                                 # [n - 1, ]
        ious = inter_areas / (areas[ind] + areas[orders[1:]] - inter_areas) # [n - 1, ]
        left = np.where(ious < t)[0]
        orders = orders[left + 1]
    return pick_bboxes, pick_confs

## Transformer
class PositionEmbedding(nn.Module):
    def __init__(self, max_len, d_model):
        super().__init__()
        _constant = torch.tensor(10000)

        pe = torch.zeros(size=(max_len, d_model))      # [max_len, d_model]
        pos = torch.arange(max_len).unsqueeze(1)       # [max_len, 1]
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-torch.log(_constant)) / d_model)  # [d_model / 2]
        pe[:, 0::2] = torch.sin(pos * div_term)      # [max_len, d_model / 2]
        pe[:, 1::2] = torch.cos(pos * div_term)      # [max_len, d_model / 2]
        pe.unsqueeze_(0)                               # [batch, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        """ x: [batch, max_len, d_model] """
        return x + self.pe[:, :x.shape[1]]

class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
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
        if self-attention, q = k = v. 
        if enc-dec attention, q from decoder, k and v from encoder
        """
        batch = x.shape[0]
        Q = self.q_linear(q).view(batch, -1, self.heads, self.d_k).transpose(1, 2)  # [batch, heads, max_len, d_k]
        K = self.k_linear(k).view(batch, -1, self.heads, self.d_k).transpose(1, 2)  # [batch, heads, max_len, d_k]
        V = self.v_linear(v).view(batch, -1, self.heads, self.d_k).transpose(1, 2)  # [batch, heads, max_len, d_k]
        score = torch.matmul(Q, K.transpose(-2, -1)) / torch.tensor(self.d_k).sqrt()     # [batch, heads, max_len, max_len]
        if mask is not None:
            score = score.masked_fill(mask == 0, float('-inf'))
        score = torch.softmax(score, dim=-1)
        out = torch.matmul(score, V).transpose(1, 2).reshape(batch, -1, self.heads * self.d_k)     # [batch, max_len, d_model]  希望你记住view/reshape的区别
        out = self.fc_linear(out)
        return out

class FFN(nn.Module):
    def __init__(self, d_model, d_ffn):
        super().__init__()

        self.linear1 = nn.Linear(d_model, d_ffn)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout = nn.Dropout(p=.1)
    
    def forward(self, x):
        return self.linear2(self.dropout(F.relu(self.linear1(x))))

class EncoderLayer(nn.Module):
    def __init__(self, heads, d_model, d_ffn):
        super().__init__()

        self.MSA = MultiHeadAttention(heads=heads, d_model=d_model)
        self.FFN = FFN(d_model=d_model, d_ffn=d_ffn)
        self.LN1 = nn.LayerNorm(d_model)
        self.LN2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(p=.1)
    
    def forward(self, x):
        mas_out = self.MSA(x, x, x)     # self-attention here
        x = self.LN1(x + self.dropout(mas_out))
        ffn_out = self.FFN(x)
        x = self.LN2(x + self.dropout(ffn_out))
        return x

class Encoder(nn.Module):
    def __init__(self, voc_size, max_len, d_model, d_ffn, heads, n):
        super().__init__()

        self.emb = nn.Embedding(voc_size, d_model)
        self.pos_emb = PositionEmbedding(max_len, d_model)
        self.encs = nn.ModuleList([EncoderLayer(heads, d_model, d_ffn) for _ in range(n)])
    
    def forward(self, x):
        x = self.emb(x)
        x = self.pos_emb(x)
        for enc in self.encs:
            x = enc(x)
        return x

class Transformer(nn.Module):
    def __init__(self, voc_size, max_len, d_model, d_ffn, heads, n):
        super().__init__()

        self.encoder = Encoder(voc_size, max_len, d_model, d_ffn, heads, n)
    
    def forward(self, x):
        return self.encoder(x)

if __name__ == '__main__':
    voc_size = 6000
    batch, max_len, d_model = 1, 77, 512
    heads = 4
    d_ffn = 1024
    
    x = torch.zeros(batch, max_len, d_model)
    print(f'>>> x shape:{x.shape}')

    posEmb = PositionEmbedding(max_len, d_model)
    print(f'>>> posEmb output shape:{posEmb(x).shape}')

    MSA_block = MultiHeadAttention(heads, d_model)
    msa_out = MSA_block(q=x, k=x, v=x)        # self-attention, same q k v
    print(f'>>> mas_out shape:{msa_out.shape}')

    FFN_block = FFN(d_model, d_ffn)
    ffn_out = FFN_block(msa_out)
    print(f'>>> ffn_out shape:{ffn_out.shape}')

    encoder_layer = EncoderLayer(heads, d_model, d_ffn)
    enc_layer_out = encoder_layer(x)
    print(f'>>> enc_layer_out shape:{enc_layer_out.shape}')

    word_inds = torch.arange(max_len, dtype=torch.long)     # 输入映射为word id [之后再embedding成为token]
    encoder_tower = Encoder(voc_size, max_len, d_model, d_ffn, heads, n=8)
    enc_out = encoder_tower(word_inds)
    print(f'>>> enc_out shape:{enc_out.shape}')