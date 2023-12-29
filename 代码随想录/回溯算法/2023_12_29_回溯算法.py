import torch
from torch import nn
import torch.nn.functional as F
import math

# Helper function to support different mask shapes.
# Output shape supports (batch_size, number of heads, seq length, seq length)
# If 2D: broadcasted over batch size and number of heads
# If 3D: broadcasted over number of heads
# If 4D: leave as is
def expand_mask(mask):
    assert mask.ndim > 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


def scaled_dot_product(q, k, v, mask=None):
    """ q,k,v: [b,h,s,h_d]
        return: 每个头的计算注意力结果、每个头的注意力分数 """
    h_d = q.size()[-1]                                              # 每个head分得的维度
    attn_logits = torch.matmul(q, k.transpose(-2, -1))              # [b,h,s,s]     为每个头 h维度 计算注意力分数
    attn_logits = attn_logits / math.sqrt(h_d)
    if mask is not None:
        attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
    attention = F.softmax(attn_logits, dim=-1)                      # [b,h,s,s]
    values = torch.matmul(attention, v)                             # [b,h,s,h_d]   为每个头 h维度 计算注意结果
    return values, attention                                        # [b,h,s,h_d], [b,h,s,s]


class MultiheadAttention(nn.Module):

    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be 0 modulo number of heads."

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        # Stack all weight matrices 1...h together for efficiency
        # Note that in many implementations you see "bias=False" which is optional
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        self.o_proj = nn.Linear(embed_dim, embed_dim)

        self._reset_parameters()

    def _reset_parameters(self):
        # Original Transformer initialization, see PyTorch documentation
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)

    def forward(self, x, mask=None, return_attention=False):
        """ x: [b,s,d] """
        batch_size, seq_length, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        qkv = self.qkv_proj(x)                                                              # [b,s,3*d]                 得到qkv的计算结果

        # Separate Q, K, V from linear output
        qkv = qkv.reshape(batch_size, seq_length, self.num_heads, 3*self.head_dim)          # [b,s,h,3*h_d]             把 d 拆分成 h个h_d
        qkv = qkv.permute(0, 2, 1, 3) # [Batch, Head, SeqLen, Dims]                           [b,h,s,3*h_d]
        q, k, v = qkv.chunk(3, dim=-1)                                                      # [b,h,s,h_d]               分离 q k v

        # Determine value outputs
        values, attention = scaled_dot_product(q, k, v, mask=mask)                          # [b,h,s,h_d], [b,h,s,s]
        values = values.permute(0, 2, 1, 3) # [Batch, SeqLen, Head, Dims]               [b,s,h,h_d]               与计算attention前形状相同了
        values = values.reshape(batch_size, seq_length, self.embed_dim)                     # [b,s,h*h_d]               与输入形状相同了
        o = self.o_proj(values)                                                             # [b,s,d]     h*h_d即d       综合多头注意力结果

        if return_attention:
            return o, attention
        else:
            return o