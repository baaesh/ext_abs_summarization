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
        self.W = nn.Linear(self.q_dim, self.k_dim, bias=False)
        self.attn = DotProductAttention()

    def forward(self, q, k, v=None, rep_mask=None):
        if v is None:
            v = k
        q_proj = self.W(q)
        return self.attn(q_proj, k, v, rep_mask)


class AdditiveAttention(nn.Module):

    def __init__(self, opt, q_dim=None, k_dim=None, hidden_dim=None):
        super(AdditiveAttention, self).__init__()
        self.q_dim = q_dim or opt['lstm_hidden_units']
        self.k_dim = k_dim or opt['lstm_hidden_units']
        self.hidden_dim = hidden_dim or opt['lstm_hidden_units']

        self.w_q = nn.Linear(self.q_dim, self.hidden_dim, bias=False)
        self.w_k = nn.Linear(self.k_dim, self.hidden_dim, bias=False)

        self.v = nn.Linear(self.hidden_dim, 1, bias=False)

    def forward(self, q, k, v=None, rep_mask=None):
        # batch_size x num_target x 1 x hidden_dim
        q_proj = self.w_q(q).unsqueeze(2)
        # batch_size x 1 x sequence_length x hidden_dim
        k_proj = self.w_k(k).unsqueeze(1)

        # batch_size x num_target x sequence_length
        u = self.v(torch.tanh(k_proj + q_proj)).squeeze(-1)

        # attn: batch_size x num_target x sequence_length
        if rep_mask is None:
            attn = F.softmax(u, dim=2)
        else:
            attn = masked_softmax(u, rep_mask.unsqueeze(1), dim=2)

        out = None
        # attn: batch_size x num_target x sequence_length
        if v is not None:
            out = torch.bmm(attn, v)
        else:
            # k_proj: batch_size x sequence_length x hidden_unis
            # out: batch_size x num_target x hidden_units
            out = torch.bmm(attn, k_proj.squeeze(1))

        return out, attn
