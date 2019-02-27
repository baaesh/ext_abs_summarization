import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


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
        # batch size x text len x sentence len x word dim
        batch_size, text_len, sentence_len, word_dim = x.size()
        conv_in = x.view(batch_size * text_len, 1, -1)

        out = []
        for filter_size in self.opt['filter_sizes']:
            conv_out = getattr(self, 'conv_' + str(filter_size))(conv_in)
            relu_out = F.relu(conv_out)
            pool_out = F.max_pool1d(relu_out, sentence_len - filter_size + 1)
            out.append(pool_out.view(batch_size, text_len, self.opt['num_feature_maps']))
        out = torch.cat(out, -1)
        return out


class LSTMEncoder(nn.Module):

    def __init__(self, opt, input_size=None, hidden_size=None, num_layers=None, bidirectional=None):
        super(LSTMEncoder, self).__init__()
        self.opt = opt
        self.input_size = input_size or opt['word_dim']
        self.hidden_size = hidden_size or opt['lstm_hidden_units']
        self.num_layers = num_layers or opt['lstm_num_layers']
        self.bidirectional = bidirectional or opt['lstm_bidirection']

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            bidirectional=self.bidirectional,
                            batch_first=True)

    def forward(self, x, lens):
        batch_size, _, _ = x.size()
        lens, indices = torch.sort(lens, 0, True)

        x_sorted = self.reorder_sequence(x, indices)

        lstm_in = pack(x_sorted, lens.tolist(), batch_first=True)
        lstm_out, lstm_states = self.lstm(lstm_in)
        lstm_out, _ = unpack(lstm_out, batch_first=True)

        _, reverse_indices = torch.sort(indices, 0)
        lstm_out = self.reorder_sequence(lstm_out, reverse_indices)
        h_n, c_n = self.reorder_lstm_states(lstm_states, reverse_indices)

        return lstm_out, (h_n, c_n)

    def reorder_sequence(self, x, reorder_idx):
        return x[reorder_idx]

    def reorder_lstm_states(self, x, reorder_idx):
        return (x[0].index_select(index=reorder_idx, dim=1),
                x[1].index_select(index=reorder_idx, dim=1))
