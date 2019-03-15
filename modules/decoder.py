import torch
import torch.nn as nn
from torch.nn import init
from torch.nn.utils.rnn import pad_packed_sequence as unpack
from torch.nn.utils.rnn import pack_padded_sequence as pack

from modules.attention import BilinearAttention, AdditiveAttention
from modules.mask import get_target_mask
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
        return torch.cat(outputs, dim=1)

    def _step(self, input, prev_out, prev_state, encoder_outs, source_rep_mask=None):
        lstm_in = torch.cat([input, prev_out], dim=2)
        lstm_out, state = self.lstm(lstm_in, prev_state)

        context, _ = self.attn(lstm_out, encoder_outs, encoder_outs, source_rep_mask)

        output = self.out_proj(torch.cat([lstm_out, context], dim=2))

        return output, state


class PointerGeneratorDecoder(nn.Module):

    def __init__(self, opt, input_size=None, hidden_size=None, num_layers=None):
        super(PointerGeneratorDecoder, self).__init__()
        self.opt = opt
        self.input_size = input_size or opt['word_dim'] * 2
        self.hidden_size = hidden_size or opt['lstm_hidden_units']
        self.num_layers = num_layers or opt['lstm_num_layers']

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)

        self.attn = BilinearAttention(opt)

        self.p_gen_linear = nn.Linear(opt['lstm_hidden_units'] * 2 + opt['word_dim'], 1)

    def forward(self, input, prev_out, prev_state, encoder_outs, source_rep_mask=None):
        lstm_in = torch.cat([input, prev_out], dim=2)
        lstm_out, state = self.lstm(lstm_in, prev_state)

        context, attn = self.attn(lstm_out, encoder_outs, encoder_outs, source_rep_mask)

        out = torch.cat([lstm_out, context], dim=2)

        p_gen_input = torch.cat((context, lstm_out, input), dim=2)
        p_gen = self.p_gen_linear(p_gen_input)
        p_gen = torch.sigmoid(p_gen).squeeze(1)

        return out, state, p_gen, attn


class PointerNetworkDecoder(nn.Module):

    def __init__(self, opt, input_size=None, hidden_size=None, num_layers=None):
        super(PointerNetworkDecoder, self).__init__()
        self.opt = opt
        self.input_size = input_size or opt['lstm_hidden_units']
        self.hidden_size = hidden_size or opt['lstm_hidden_units']
        self.num_layers = num_layers or opt['lstm_num_layers']

        # initial input and states of lstm
        self.init_h = nn.Parameter(torch.Tensor(self.num_layers, self.hidden_size))
        self.init_c = nn.Parameter(torch.Tensor(self.num_layers, self.hidden_size))
        self.init_in = nn.Parameter(torch.Tensor(self.input_size))
        init.uniform_(self.init_h, -1e-2, 1e-2)
        init.uniform_(self.init_c, -1e-2, 1e-2)
        init.uniform_(self.init_in, -0.1, 0.1)

        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            batch_first=True)

        self.glimpse_attn = AdditiveAttention(opt, k_dim=self.input_size)
        self.point_attn = AdditiveAttention(opt, k_dim=self.input_size)

    def forward(self, enc_outs, target, source_rep_mask=None, target_length=None):
        lstm_in, lstm_states = self._prepare(enc_outs, target)

        ### LSTM
        target_length, indices = torch.sort(target_length, 0, True)

        lstm_in_sorted = self.reorder_sequence(lstm_in, indices)
        lstm_in_packed = pack(lstm_in_sorted, target_length.tolist(), batch_first=True)
        # lstm_out: batch_size x num_target x hidden_units
        lstm_out_packed, _ = self.lstm(lstm_in_packed, lstm_states)
        lstm_out, _ = unpack(lstm_out_packed, batch_first=True)

        _, reverse_indices = torch.sort(indices, 0)
        lstm_out = self.reorder_sequence(lstm_out, reverse_indices)

        ### glimpse attention
        # glimpse: batch_size x num_target x hidden_units
        glimpse, _ = self.glimpse_attn(lstm_out, enc_outs, enc_outs, source_rep_mask)

        ### point attention
        # probs: batch_size x num_target x num_sentence
        _, probs = self.point_attn(glimpse, enc_outs, rep_mask=source_rep_mask)

        return (probs + 1e-20).log()

    def reorder_sequence(self, x, reorder_idx):
        return x[reorder_idx]

    def _prepare(self, enc_outs, target):
        batch_size, num_target = target.size()
        hidden_dim = enc_outs.size(2)

        init_in = self.init_in.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, hidden_dim)
        ptr_in = torch.gather(
            enc_outs, dim=1, index=target.unsqueeze(2).expand(batch_size, num_target, hidden_dim)
        )
        # lstm_in: batch_size x num_target x hidden_units
        lstm_in = torch.cat([init_in, ptr_in], dim=1)

        size = (self.num_layers, batch_size, self.hidden_size)
        lstm_states = (self.init_h.unsqueeze(1).expand(*size).contiguous(),
                       self.init_c.unsqueeze(1).expand(*size).contiguous())
        return lstm_in, lstm_states
