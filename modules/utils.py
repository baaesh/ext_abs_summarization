import torch


# https://github.com/ChenRocks/fast_abs_rl/blob/master/model/util.py
def sequence_mean(sequence, seq_lens=None, dim=1, keepdim=False):
    if seq_lens is not None:
        sum_ = torch.sum(sequence, dim=dim, keepdim=keepdim)
        mean = torch.stack([s/l for s, l in zip(sum_, seq_lens)], dim=0)
    else:
        mean = torch.mean(sequence, dim=dim, keepdim=keepdim)
    return mean
