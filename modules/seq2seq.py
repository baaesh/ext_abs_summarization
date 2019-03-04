import torch
import torch.nn as nn

from encoder import LSTMEncoder
from decoder import AttnLSTMDecoder
from utils import get_rep_mask


class Seq2SeqAttn(nn.Module):

    def __init__(self, opt, vectors=None):
        super(Seq2SeqAttn, self).__init__()
        self.opt = opt

        self.word_embedding = nn.Embedding(opt['vocab_size'], opt['word_dim'])
        if vectors is not None:
            self.word_embedding.weight.data.copy_(vectors)
        if opt['fix_embedding']:
            self.word_embedding.weight.requires_grad = False

        self.encoder = LSTMEncoder(opt)
        self.decoder = AttnLSTMDecoder(opt)

        self.enc_out_dim = opt['lstm_hidden_units']
        if opt['lstm_bidirection']:
            self.enc_out_dim *= 2

        self.enc_out_proj = nn.Linear(self.enc_out_dim, opt['lstm_hidden_units'])
        self.enc2dec_h = nn.Linear(self.enc_out_dim, opt['lstm_hidden_units'])
        self.enc2dec_c = nn.Linear(self.enc_out_dim, opt['lstm_hidden_units'])

    def forward(self, source, source_lens, target=None, target_lens=None):
        w_s = self.word_embedding(source)
        enc_outs, enc_states = self.encoder(w_s, source_lens)

        enc_outs = self.enc_out_proj(enc_outs)
        dec_h = torch.stack([self.enc2dec_h(h)
                             for h in enc_states[0]], dim=0)
        dec_c = torch.stack([self.enc2dec_c(c)
                             for c in enc_states[1]], dim=0)
        source_rep_mask = get_rep_mask(source_lens)

        if target is not None:
            logits = self.train_forward(enc_outs, (dec_h, dec_c), target, source_rep_mask)
            return logits
        else:
            self.predict_step(enc_outs, (dec_h, dec_c))

    def train_forward(self, enc_outs, dec_hiddens, target, source_rep_mask=None):
        max_len = target.size(1)
        logits = []
        prev_out = None
        for i in range(max_len):
            input = target[:, i:i+1]
            dec_out, dec_hiddens = self.decoder(input, prev_out, dec_hiddens, enc_outs, source_rep_mask)
            logit = torch.mm(dec_out, self.word_embedding.weight.t())
            logits.append(logit)
            prev_out = dec_out
        logits = torch.stack(logits, dim=1)
        return logits

    def predict_step(self, enc_outs, dec_hiddens):
        # TODO
        pass
