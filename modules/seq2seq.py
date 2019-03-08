import torch
import torch.nn as nn

from modules.encoder import LSTMEncoder
from modules.decoder import AttnLSTMDecoder
from modules.utils import sequence_mean
from modules.mask import get_rep_mask


class Seq2Seq(nn.Module):

    def __init__(self, opt, pad_id, bos_id, vectors=None):
        super(Seq2Seq, self).__init__()
        self.opt = opt
        self.pad_id = pad_id
        self.bos_id = bos_id

        self.word_embedding = nn.Embedding(opt['vocab_size'], opt['word_dim'],
                                           padding_idx = pad_id)
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

    def forward(self, source, source_length, target=None, target_length=None):
        # Get mask
        source_rep_mask = get_rep_mask(source_length)

        w_s = self.word_embedding(source)

        ### Encode
        enc_outs, enc_states = self.encoder(w_s, source_length)

        ### Initialize decoder states
        enc_outs = self.enc_out_proj(enc_outs)
        dec_h = torch.stack([self.enc2dec_h(h)
                             for h in enc_states[0]], dim=0)
        dec_c = torch.stack([self.enc2dec_c(c)
                             for c in enc_states[1]], dim=0)

        ### Decode
        # Training phase
        if target is not None:
            target = self.word_embedding(target)
            dec_outs = self.decoder(enc_outs, (dec_h, dec_c), source_length, source_rep_mask, target)
            logits = torch.matmul(dec_outs, self.word_embedding.weight.t())
            return logits
        # Prediction phase
        else:
            batch_size = enc_outs.size(0)
            pred = torch.tensor([self.bos_id] * batch_size).unsqueeze(-1).to(enc_outs.device)
            dec_out = self.decoder.out_proj(torch.cat(
                [dec_h[-1], sequence_mean(enc_outs, source_length, dim=1)], dim=1)).unsqueeze(1)
            max_len = self.opt['max_len']
            preds = []
            for i in range(max_len):
                dec_in = self.word_embedding(pred)
                dec_out, (dec_h, dec_c) = self.decoder._step(dec_in,
                                                             dec_out,
                                                             (dec_h, dec_c),
                                                             enc_outs,
                                                             source_rep_mask)
                logit = torch.mm(dec_out, self.word_embedding.weight.t())
                pred = torch.argmax(logit, dim=-1)
                preds.append(pred)
            return pred
