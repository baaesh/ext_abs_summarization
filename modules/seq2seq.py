import torch
import torch.nn as nn

from allennlp.nn.util import get_text_field_mask,\
    get_lengths_from_binary_sequence_mask

from modules.encoder import LSTMEncoder
from modules.decoder import AttnLSTMDecoder
from modules.utils import sequence_mean

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
            nn.Linear(2 * opt['lstm_hidden_units'], opt['lstm_hidden_units']),
            nn.Tanh(),
            nn.Linear(opt['lstm_hidden_units'], opt['word_dim'], bias=False))

    def forward(self, source, target=None):
        # Get mask and sequence lengths
        source_rep_mask = get_text_field_mask(source).to(self.opt['device'])
        source_lens = get_lengths_from_binary_sequence_mask(source_rep_mask).to(self.opt['device'])
        source = source['tokens'].to(self.opt['device'])

        w_s = self.word_embedding(source)

        ### Encode
        enc_outs, enc_states = self.encoder(w_s, source_lens)

        ### Initialize decoder states
        enc_outs = self.enc_out_proj(enc_outs)
        dec_h = torch.stack([self.enc2dec_h(h)
                             for h in enc_states[0]], dim=0)
        dec_c = torch.stack([self.enc2dec_c(c)
                             for c in enc_states[1]], dim=0)
        init_dec_out = self.dec_out_proj(torch.cat(
            [dec_h[-1], sequence_mean(enc_outs, source_lens, dim=1)], dim=1)).unsqueeze(1)

        ### Decode
        # Training phase
        if target is not None:
            target = self.word_embedding(target['tokens'].to(self.opt['device']))
            logits = self.train_forward(enc_outs, (dec_h, dec_c), init_dec_out, target, source_rep_mask)
            return logits
        # Prediction phase
        else:
            preds = self.predict_forward(enc_outs, (dec_h, dec_c), init_dec_out, source_rep_mask)
            return preds

    def decode_step(self, dec_in, dec_out, dec_state, enc_outs, source_rep_mask=None):
        lstm_out, context, dec_state = self.decoder(dec_in, dec_out, dec_state, enc_outs, source_rep_mask)
        dec_out = self.dec_out_proj(torch.cat([lstm_out, context], dim=2))
        logit = torch.mm(dec_out.squeeze(1), self.word_embedding.weight.t())
        return logit, dec_out, dec_state

    def train_forward(self, enc_outs, dec_state, dec_out, target, source_rep_mask=None):
        max_len = target.size(1)
        logits = []
        for i in range(max_len):
            dec_in = target[:, i:i+1]
            logit, dec_out, dec_state = self.decode_step(dec_in, dec_out, dec_state, enc_outs, source_rep_mask)
            logits.append(logit)
        logits = torch.stack(logits, dim=1)
        return logits

    def predict_forward(self, enc_outs, dec_state, dec_out, source_rep_mask=None):
        batch_size = enc_outs.size(0)
        pred = torch.tensor([self.bos_id] * batch_size).unsqueeze(-1).to(enc_outs.device)
        max_len = self.opt['max_len']
        preds = []
        for i in range(max_len):
            dec_in = self.word_embedding(pred)
            logit, dec_out, dec_state = self.decode_step(dec_in, dec_out, dec_state, enc_outs, source_rep_mask)
            pred = torch.argmax(logit, dim=2)
            preds.append(pred)
        preds = torch.stack(preds, dim=1)
        return preds
