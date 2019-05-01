import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvEncoder(nn.Module):

    def __init__(self, opt, input_dim=None, mode='s'):
        super(ConvEncoder, self).__init__()
        self.opt = opt
        self.mode = mode
        self.input_dim = input_dim or opt['word_dim']

        for filter_size in opt['filter_sizes']:
            conv = nn.Conv1d(in_channels=1,
                             out_channels=opt['num_feature_maps'],
                             kernel_size=self.input_dim * filter_size,
                             stride=self.input_dim)
            setattr(self, 'conv_' + str(filter_size), conv)

    def forward(self, x):
        if self.mode == 's':
            # batch size x num_sentence x num_word x word_dim
            batch_size, num_sentence, seq_len, _ = x.size()
            conv_in = x.view(batch_size * num_sentence, 1, -1)
        else:
            # batch_size x num_sentence x sentence_dim
            batch_size, seq_len, _ = x.size()
            conv_in = x.view(batch_size, 1, -1)

        out = []
        for filter_size in self.opt['filter_sizes']:
            conv_out = getattr(self, 'conv_' + str(filter_size))(conv_in)
            relu_out = F.relu(conv_out)
            pool_out = F.max_pool1d(relu_out, seq_len - filter_size + 1)
            if self.mode == 's':
                out.append(pool_out.view(batch_size, num_sentence, self.opt['num_feature_maps']))
            else:
                out.append(pool_out.view(batch_size, -1))
        # batch_size x num_sentence x (num_filter * num_feature_maps)
        # batch_size x (num_filter * num_feature_maps)
        out = torch.cat(out, -1)
        return out
