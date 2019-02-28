import torch
import torch.nn as nn
from torch.nn import functional as F


# Masked softmax
def masked_softmax(vec, mask, dim=1):
    masked_vec = vec * mask.float()
    max_vec = torch.max(masked_vec, dim=dim, keepdim=True)[0]
    exps = torch.exp(masked_vec - max_vec)
    masked_exps = exps * mask.float()
    masked_sums = masked_exps.sum(dim, keepdim=True)
    zeros = (masked_sums == 0)
    masked_sums += zeros.float()
    return masked_exps / (masked_sums + 1e-20)


# Representation mask for sentences of variable lengths
def get_rep_mask_tile(rep_mask):
    batch_size, sentence_len, _ = rep_mask.size()

    m1 = rep_mask.view(batch_size, sentence_len, 1)
    m2 = rep_mask.view(batch_size, 1, sentence_len)
    mask = torch.mul(m1, m2)

    return mask


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
        """
        attn = torch.bmm(q, k.transpose(1, 2))

        if rep_mask is None:
            attn = F.softmax(attn, dim=2)
        else:
            rep_mask_tile = get_rep_mask_tile(rep_mask)
            mask = rep_mask_tile

            attn = masked_softmax(attn, mask, dim=2)

        out = torch.bmm(attn, v)
        return out


class BilinearAttention(nn.Module):

    def __init__(self, opt, q_dim=None, k_dim=None):
        super(BilinearAttention, self).__init__()
        self.q_dim = q_dim or opt['lstm_hidden_units']
        self.k_dim = k_dim or opt['lstm_hidden_units']
        self.W = nn.Linear(self.q_dim, self.k_dim)
        self.attn = DotProductAttention()

    def forward(self, q, k, v, rep_mask):
        q_proj = self.W(q)
        return self.attn(q_proj, k, v, rep_mask)