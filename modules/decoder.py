import torch.nn as nn


class LSTMDecoder(nn.Module):

    def __init__(self, opt, input_size=None, hidden_size=None, num_layers=None):
        super(LSTMDecoder, self).__init__()
        self.opt = opt
        self.input_size = input_size or opt['word_dim']
        self.hidden_size = hidden_size or opt['lstm_hidden_units']
        self.num_layers = num_layers or opt['lstm_num_layers']

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)

    def forward(self, input, encoder_outs, init_states):
        lstm_out, lstm_states = self.lstm(input, init_states)
