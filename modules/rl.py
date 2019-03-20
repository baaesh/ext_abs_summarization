import copy

import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.extractor import PointerNetwork
from modules.mask import get_rep_mask


# to be used as critic
class PtrScorer(nn.Module):

    def __init__(self, opt, ptr_decoder):
        super(PtrScorer, self).__init__()
        self.init_h = nn.Parameter(ptr_decoder.init_h.clone())
        self.init_c = nn.Parameter(ptr_decoder.init_c.clone())
        self.init_in = nn.Parameter(ptr_decoder.init_in.clone())
        self.lstm = copy.deepcopy(ptr_decoder.lstm)
        self.attn = copy.deepcopy(ptr_decoder.glimpse_attn)

        self.score_linear = nn.Linear(ptr_decoder.hidden_size, 1)

    def forward(self, enc_outs, num_sent, num_pred):
        rep_mask = get_rep_mask(num_sent)

        batch_size, _, input_dim = enc_outs.size()
        num_layers, hidden_dim = self.init_h.size()

        size = (num_layers, batch_size, hidden_dim)
        lstm_states = (self.init_h.unsqueeze(1).expand(*size).contiguous(),
                       self.init_c.unsqueeze(1).expand(*size).contiguous())

        lstm_in = self.init_in.unsqueeze(0).unsqueeze(1).expand(batch_size, 1, input_dim)

        scores = []
        for i in range(num_pred):
            lstm_out, _ = self.lstm(lstm_in, lstm_states)
            output, _ = self.attn(lstm_out, enc_outs, enc_outs, rep_mask)
            score = self.score_linear(output)
            scores.append(score)
            lstm_in = output
        return scores


class ActorCritic(nn.Module):

    def __init__(self, opt, pad_id, ext_state_dict):
        super(ActorCritic, self).__init__()

        ### Actor
        self._ext = PointerNetwork(opt, pad_id)
        self._ext.load_state_dict(ext_state_dict)

        ### Critic
        self.critic = PtrScorer(opt, self._ext.decoder)

    def forward(self, source, num_sentence, length):
        batch_size, _, max_length = source.size()

        # preds: batch_size x max_ext
        preds, enc_outs = self._ext(source, num_sentence)
        _, max_ext = source.size()

        scores = self.critic(enc_outs, num_sentence, max_ext)

        return preds, scores