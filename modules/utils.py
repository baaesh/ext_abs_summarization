import torch
from torch.nn.utils.rnn import pad_sequence

# https://github.com/ChenRocks/fast_abs_rl/blob/master/model/util.py
def sequence_mean(sequence, seq_lens=None, dim=1, keepdim=False):
    if seq_lens is not None:
        sum_ = torch.sum(sequence, dim=dim, keepdim=keepdim)
        mean = torch.stack([s / l for s, l in zip(sum_, seq_lens)], dim=0)
    else:
        mean = torch.mean(sequence, dim=dim, keepdim=keepdim)
    return mean


def point2text(points, source, max_len, pad_id):
    # points: batch_size x max_ext
    batch_size, max_ext = points.size()
    points = points.cpu().numpy()
    batch_extracted = []
    batch_length = []
    for i in range(batch_size):
        point = points[i]
        extracted = []
        for j in range(max_ext):
            if source[i][point[j]] == pad_id:
                break
            extracted += source[i][point[j]]
        batch_extracted.append(torch.tensor(extracted))
        batch_length.appned(len(extracted))
    batch_extracted = pad_sequence(
        batch_extracted, batch_first=True, padding_value=pad_id)
    batch_length = torch.tensor(batch_length)
    return batch_extracted, batch_length


def one_hot_embedding(labels, batch_size, num_classes, device):
    zeros = torch.zeros(batch_size, num_classes).to(device)
    return zeros.scatter_(1, labels.unsqueeze(-1), 1)