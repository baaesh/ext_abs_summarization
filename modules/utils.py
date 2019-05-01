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


def remove_pad(tokens, pad_id):
    for i in range(len(tokens)):
        if tokens[i] == pad_id:
            return tokens[:i]
    return tokens

def point2result(points, origins):
    # points: batch_size x max_ext
    batch_size, max_ext = points.size()
    points = points.cpu().numpy()
    batch_extracted = []
    for i in range(batch_size):
        point = points[i]
        origin = origins[i]
        extracted = []
        for j in range(max_ext):
            extracted.append(origin[point[j]])
        batch_extracted.append(extracted)
    return batch_extracted


def point2text(points, source, source_length, pad_id, device='cuda:0'):
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
            extracted += remove_pad(source[i][point[j]], pad_id)

        # heuristic... to avoid error
        if len(extracted) == 0:
            extracted += [1]

        batch_extracted.append(torch.tensor(extracted))
        batch_length.append(len(extracted))
    batch_extracted = pad_sequence(
        batch_extracted, batch_first=True, padding_value=pad_id).to(device)
    batch_length = torch.tensor(batch_length).to(device)
    return batch_extracted, batch_length


def one_hot_embedding(labels, batch_size, num_classes, device):
    zeros = torch.zeros(batch_size, num_classes).to(device)
    return zeros.scatter_(1, labels.unsqueeze(-1), 1)


def idx2origin(idx_list, vocab, oov_tokens):
    tokens = []
    for idx in idx_list:
        if idx >= len(vocab):
            for token, oov_idx in oov_tokens.items():
                if oov_idx == idx:
                    tokens.append(token.strip())
        else:
            tokens.append(vocab.itos(idx).strip())
    return ' '.join(tokens)


def strip_positions(idx_list, max_len, pad_id):
    end_idx = max_len
    for i in range(len(idx_list)):
        if idx_list[i] == pad_id:
            end_idx = i
            break
    return idx_list[:end_idx]
    # return [idx for idx in idx_list if idx != pad_id]


def strip_sequence(words_list, max_len, bos_id, eos_id):
    start_idx = 0
    if words_list[0] == bos_id:
        start_idx = 1
    end_idx = max_len
    for i in range(len(words_list)):
        if words_list[i] == eos_id:
            end_idx = i
            break
    return words_list[start_idx:end_idx]
