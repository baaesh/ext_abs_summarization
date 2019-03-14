import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvSentenceEncoder(nn.Module):

    def __init__(self, opt):
        super(ConvSentenceEncoder, self).__init__()
        self.opt = opt

        for filter_size in opt['filter_sizes']:
            conv = nn.Conv1d(in_channels=1,
                             out_channels=opt['num_feature_maps'],
                             kernel_size=opt['word_dim'] * filter_size,
                             stride=opt['word_dim'])
            setattr(self, 'conv_' + str(filter_size), conv)

    def forward(self, x):
        # batch size x num_sentence x num_word x word dim
        batch_size, text_len, sentence_len, word_dim = x.size()
        conv_in = x.view(batch_size * text_len, 1, -1)

        out = []
        for filter_size in self.opt['filter_sizes']:
            conv_out = getattr(self, 'conv_' + str(filter_size))(conv_in)
            relu_out = F.relu(conv_out)
            pool_out = F.max_pool1d(relu_out, sentence_len - filter_size + 1)
            out.append(pool_out.view(batch_size, text_len, self.opt['num_feature_maps']))
        # batch_size x num_sentence x (num_filter * num_feature_maps)
        out = torch.cat(out, -1)
        return out
