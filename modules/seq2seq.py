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

        source_rep_mask = get_rep_mask(source_lens)
        w_s = self.word_embedding(source)
        enc_outs, enc_states = self.encoder(w_s, source_lens)

        enc_outs = self.enc_out_proj(enc_outs)
        dec_h = torch.stack([self.enc2dec_h(h)
                             for h in enc_states[0]], dim=0)
        dec_c = torch.stack([self.enc2dec_c(c)
                             for c in enc_states[1]], dim=0)
