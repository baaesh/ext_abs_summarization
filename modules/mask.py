import torch
import torch.nn.functional as F


def get_rep_mask(length, max_length=None):
    """Returns a length mask.

    Args:
        length (`torch.LongTensor`): A tensor of lengths; (batch_size,).
        max_length (int, optional): The maximum length.

    Returns:
        rep_mask (`torch.ByteTensor`): The binary mask tensor of size
            (batch_size, max_length).
    """
    device = length.device
    if max_length is None:
        max_length = length.max()
    # range_tensor: (1, max_length)
    range_tensor = torch.arange(max_length, device=device).unsqueeze(0)
    length = length.unsqueeze(1)  # (batch_size, 1)
    rep_mask = torch.lt(range_tensor, length)
    return rep_mask


def get_rep_mask_2d(lengths, max_length=None):
    device = lengths.device
    if max_length is None:
        max_length = lengths.max()
    batch_size = lengths.size()[0]
    range_tensor = torch.arange(max_length, device=device).expand(batch_size, -1).unsqueeze(1)
    lengths = lengths.unsqueeze(-1)
    rep_mask = torch.lt(range_tensor, lengths)
    return rep_mask


# Representation mask for sentences of variable lengths
def get_rep_mask_tile(rep_mask):
    batch_size, sentence_len, _ = rep_mask.size()

    m1 = rep_mask.view(batch_size, sentence_len, 1)
    m2 = rep_mask.view(batch_size, 1, sentence_len)
    mask = torch.mul(m1, m2)

    return mask


def masked_softmax(score, mask, dim=1):
    """ [(...), T]
    user should handle mask shape"""
    score = score.masked_fill(mask == 0, -1e18)
    norm_score = F.softmax(score, dim=dim)
    return norm_score


def get_target_mask(target, num_sequence):
    batch_size, num_target = target.size()

    mask = torch.zeros(batch_size, num_sequence).to(target.device)
    masks = [mask]

    for i in range(num_target - 1):
        mask = mask.scatter_(1, target[:, i].unsqueeze(-1), 1)
        masks.append(mask)

    target_mask = 1 - torch.stack(masks, dim=1)
    return target_mask.byte()
