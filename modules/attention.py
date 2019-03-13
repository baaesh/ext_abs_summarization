import torch
import torch.nn as nn
from torch.nn import functional as F

from modules.mask import masked_softmax


class DotProductAttention(nn.Module):

    def __init__(self):
        super(DotProductAttention, self).__init__()

    def forward(self, q, k, v, rep_mask=None):
        """Calculates attention scores using the dot product.

                Args:
                    q: (batch_size, length1, dim)
                    k: (batch_size, length2, dim)
                    v: (batch_size, length2, dim)
                    rep_mask: (batch_size, length2)

                Returns:
                    out: (batch_size, length1, dim)
                    attn: (batch_size, length1, length2)
        """
        attn = torch.bmm(q, k.transpose(1, 2))
        rep_mask_ = rep_mask.unsqueeze(1)
        if rep_mask is None:
            attn = F.softmax(attn, dim=2)
        else:
            attn = masked_softmax(attn, rep_mask_, dim=2)

        out = torch.bmm(attn, v)
        return out, attn


class BilinearAttention(nn.Module):

    def __init__(self, opt, q_dim=None, k_dim=None):
        super(BilinearAttention, self).__init__()
        self.q_dim = q_dim or opt['lstm_hidden_units']
        self.k_dim = k_dim or opt['lstm_hidden_units']
        self.W = nn.Linear(self.q_dim, self.k_dim)
        self.attn = DotProductAttention()

    def forward(self, q, k, v=None, rep_mask=None):
        if v is None:
            v = k
        q_proj = self.W(q)
        return self.attn(q_proj, k, v, rep_mask)
