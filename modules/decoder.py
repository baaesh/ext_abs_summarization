import torch
import torch.nn as nn

from modules.attention import BilinearAttention
from modules.utils import sequence_mean


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

        self.out_proj = nn.Sequential(
            nn.Linear(2 * opt['lstm_hidden_units'], opt['lstm_hidden_units']),
            nn.Tanh(),
            nn.Linear(opt['lstm_hidden_units'], opt['word_dim'], bias=False))

    def forward(self, enc_outs, init_state, source_length, source_rep_mask=None, target=None):
        init_out = self.out_proj(torch.cat(
            [init_state[0][-1], sequence_mean(enc_outs, source_length, dim=1)], dim=1)).unsqueeze(1)

        max_len = target.size(1)
        outputs = []
        output = init_out
        state = init_state
        for i in range(max_len):
            input = target[:, i:i + 1]
            output, state = self._step(input, output, state, enc_outs, source_rep_mask)
            outputs.append(output)
        return torch.stack(outputs, dim=1)

    def _step(self, input, prev_out, prev_state, encoder_outs, source_rep_mask=None):
        lstm_in = torch.cat([input, prev_out], dim=2)
        lstm_out, state = self.lstm(lstm_in, prev_state)

        context = self.attn(lstm_out, encoder_outs, encoder_outs, source_rep_mask)

        output = self.out_proj(torch.cat([lstm_out, context], dim=2))

        return output, state