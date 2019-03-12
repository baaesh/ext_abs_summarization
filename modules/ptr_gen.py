import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.encoder import LSTMEncoder
from modules.decoder import PointerGeneratorDecoder
from modules.utils import sequence_mean
from modules.mask import get_rep_mask


class PointerGenerator(nn.Module):

    def __init__(self, opt, pad_id, bos_id, vectors=None, criterion=None):
        super(PointerGenerator, self).__init__()
        self.opt = opt
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.criterion = criterion or nn.NLLLoss()

        self.word_embedding = nn.Embedding(opt['vocab_size'], opt['word_dim'],
                                           padding_idx = pad_id)
        if vectors is not None:
            self.word_embedding.weight.data.copy_(vectors)
        if opt['fix_embedding']:
            self.word_embedding.weight.requires_grad = False

        self.encoder = LSTMEncoder(opt)
        self.decoder = PointerGeneratorDecoder(opt)

        self.enc_out_dim = opt['lstm_hidden_units']
        if opt['lstm_bidirection']:
            self.enc_out_dim *= 2

        self.enc_out_proj = nn.Linear(self.enc_out_dim, opt['lstm_hidden_units'])
        self.enc2dec_h = nn.Linear(self.enc_out_dim, opt['lstm_hidden_units'])
        self.enc2dec_c = nn.Linear(self.enc_out_dim, opt['lstm_hidden_units'])

        self.out_proj = nn.Sequential(
            nn.Linear(2 * opt['lstm_hidden_units'], opt['lstm_hidden_units']),
            nn.Tanh(),
            nn.Linear(opt['lstm_hidden_units'], opt['word_dim'], bias=False))

    def forward(self, source, source_extended, source_length,
                target=None, target_extended=None, target_length=None):
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
        batch_size = source.size(0)
        max_oov_idx = source_extended.max()
        extra_zeros = torch.zeros((batch_size, max_oov_idx - self.opt['vocab_size'] + 1)).to(enc_outs.device)
        # Training phase
        if target is not None:
            w_t = self.word_embedding(target)
            dec_out = self.out_proj(torch.cat(
                [dec_h[-1], sequence_mean(enc_outs, source_length, dim=1)], dim=1)).unsqueeze(1)
            dec_state = (dec_h, dec_c)

            logits = []
            max_len = w_t.size(1)
            for i in range(max_len):
                dec_in = w_t[:, i:i+1]
                dec_out, dec_state, p_gen, point_dist = \
                    self.decoder.forward(dec_in,
                                         dec_out,
                                         dec_state,
                                         enc_outs,
                                         source_rep_mask)
                dec_out = self.out_proj(dec_out)
                logit = torch.matmul(dec_out, self.word_embedding.weight.t())

                vocab_dist = p_gen * F.softmax(logit, dim=2).squeeze(1)
                point_dist = (1 - p_gen) * point_dist.squeeze(1)

                vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=1)
                final_dist = vocab_dist.scatter_add(1, source_extended, point_dist) + 1e-20
                logits.append(final_dist.log())
            return logits
        # Prediction phase
        else:
            pred = torch.tensor([self.bos_id] * batch_size).unsqueeze(-1).to(enc_outs.device)
            dec_out = self.out_proj(torch.cat(
                [dec_h[-1], sequence_mean(enc_outs, source_length, dim=1)], dim=1)).unsqueeze(1)
            dec_state = (dec_h, dec_c)

            preds = []
            max_len = self.opt['max_len']
            for i in range(max_len):
                dec_in = self.word_embedding(pred)
                dec_out, dec_state, p_gen, point_dist = \
                    self.decoder.forward(dec_in,
                                         dec_out,
                                         dec_state,
                                         enc_outs,
                                         source_rep_mask)
                dec_out = self.out_proj(dec_out)
                logit = torch.matmul(dec_out, self.word_embedding.weight.t())

                vocab_dist = p_gen * F.softmax(logit, dim=2).squeeze(1)
                point_dist = (1 - p_gen) * point_dist.squeeze(1)

                vocab_dist = torch.cat([vocab_dist, extra_zeros], dim=1)
                final_dist = vocab_dist.scatter_add(1, source_extended, point_dist) + 1e-20
                pred = torch.argmax(final_dist, dim=-1)
                preds.append(pred)
            return pred