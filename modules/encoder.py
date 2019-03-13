import torch
import torch.nn as nn
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack


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
        lens, indices = torch.sort(lens, 0, True)

        x_sorted = self.reorder_sequence(x, indices)

        lstm_in = pack(x_sorted, lens.tolist(), batch_first=True)
        lstm_out, lstm_states = self.lstm(lstm_in)
        lstm_out, _ = unpack(lstm_out, batch_first=True)

        if self.bidirectional:
            lstm_states = self.cat_bidirectional_states(lstm_states)

        _, reverse_indices = torch.sort(indices, 0)
        lstm_out = self.reorder_sequence(lstm_out, reverse_indices)
        h_n, c_n = self.reorder_lstm_states(lstm_states, reverse_indices)

        return lstm_out, (h_n, c_n)

    def reorder_sequence(self, x, reorder_idx):
        return x[reorder_idx]

    def reorder_lstm_states(self, states, reorder_idx):
        return (states[0].index_select(index=reorder_idx, dim=1),
                states[1].index_select(index=reorder_idx, dim=1))

    def cat_bidirectional_states(self, states):
        return (torch.cat(states[0].chunk(2, dim=0), dim=2),
                torch.cat(states[1].chunk(2, dim=1), dim=2))
