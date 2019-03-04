import torch
import torch.nn as nn

from encoder import LSTMEncoder
from decoder import AttnLSTMDecoder
from mask import get_rep_mask
from utils import sequence_mean

class Seq2SeqAttn(nn.Module):

    def __init__(self, opt, bos_id, vectors=None):
        super(Seq2SeqAttn, self).__init__()
        self.opt = opt
        self.bos_id = bos_id

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

        self.dec_out_proj = nn.Sequential(
            nn.Linear(2 * self.hidden_size, self.hidden_size),
            nn.Tanh(),
            nn.Linear(self.hidden_size, opt['word_dim'], bias=False))

    def forward(self, source, source_lens, target=None, target_lens=None):
        w_s = self.word_embedding(source)
        enc_outs, enc_states = self.encoder(w_s, source_lens)

        source_rep_mask = get_rep_mask(source_lens)

        # initialize decoder states
        enc_outs = self.enc_out_proj(enc_outs)
        dec_h = torch.stack([self.enc2dec_h(h)
                             for h in enc_states[0]], dim=0)
        dec_c = torch.stack([self.enc2dec_c(c)
                             for c in enc_states[1]], dim=0)
        init_dec_out = self.projection(torch.cat(
            [dec_h[-1], sequence_mean(enc_outs, source_lens, dim=1)], dim=1))

        # Training phase
        if target is not None:
            logits = self.train_forward(enc_outs, (dec_h, dec_c), init_dec_out, target, source_rep_mask)
            return logits
        # Prediction phase
        else:
            preds = self.predict_forward(enc_outs, (dec_h, dec_c), init_dec_out, source_rep_mask)
            return preds

    def decode_step(self, input, dec_out, dec_state, enc_outs, source_rep_mask=None):
        lstm_out, context, dec_state = self.decoder(input, dec_out, dec_state, enc_outs, source_rep_mask)
        dec_out = self.dec_out_proj(torch.cat([lstm_out, context], dim=2))
        logit = torch.mm(dec_out, self.word_embedding.weight.t())
        return logit

    def train_forward(self, enc_outs, dec_state, dec_out, target, source_rep_mask=None):
        max_len = target.size(1)
        logits = []
        for i in range(max_len):
            input = target[:, i:i+1]
            logit = self.decode_step(input, dec_out, dec_state, enc_outs, source_rep_mask)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)
        return logits

    def predict_forward(self, enc_outs, dec_state, dec_out, source_rep_mask=None):
        batch_size = enc_outs.size(0)
        preds = []
        pred = torch.tensor([self.bos_id] * batch_size).unsqueeze(-1).to(enc_outs.device)
        for i in range(self.opt['max_len']):
            input = self.word_embedding(pred)
            logit = self.decode_step(input, dec_out, dec_state, enc_outs, source_rep_mask)
            pred = torch.argmax(logit, dim=2)
            preds.append(pred)
        preds = torch.stack(preds, dim=1)
        return preds
