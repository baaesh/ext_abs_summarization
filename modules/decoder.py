import torch
import torch.nn as nn

from attention import BilinearAttention


class AttnLSTMDecoder(nn.Module):

    def __init__(self, opt, input_size=None, hidden_size=None, num_layers=None):
        super(AttnLSTMDecoder, self).__init__()
        self.opt = opt
        self.input_size = input_size or opt['word_dim'] * 2
        self.hidden_size = hidden_size or opt['lstm_hidden_units']
        self.num_layers = num_layers or opt['lstm_num_layers']

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)

        self.attn = BilinearAttention(opt)

    def forward(self, input, prev_out, prev_hidden, encoder_outs, source_rep_mask=None):
        # input: batch_size x 1 x word_dim
        # prev_out: batch_size x 1 x word_dim
        # last_hidden: num_layers x batch size x hidden_dim
        # encoder_outs: batch_size x seq_len x hidden_dim
        lstm_in = torch.cat([input, prev_out], dim=2)
        lstm_out, hidden = self.lstm(lstm_in, prev_hidden)

        # lstm_out: batch_size x 1 x hidden_dim
        # context: batch_size x 1 x hidden_dim
        context = self.attn(lstm_out, encoder_outs, encoder_outs, source_rep_mask)

        return lstm_out, context, hidden