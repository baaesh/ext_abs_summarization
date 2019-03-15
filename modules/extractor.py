import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.convolution import ConvSentenceEncoder
from modules.encoder import LSTMEncoder
from modules.decoder import PointerNetworkDecoder
from modules.mask import get_rep_mask


class PointerNetwork(nn.Module):

    def __init__(self, opt, pad_id, vectors=None):
        super(PointerNetwork, self).__init__()
        self.opt = opt

        self.word_embedding = nn.Embedding(opt['vocab_size'], opt['word_dim'],
                                           padding_idx=pad_id)
        if vectors is not None:
            self.word_embedding.weight.data.copy_(vectors)
        if opt['fix_embedding']:
            self.word_embedding.weight.requires_grad = False

        self.cnn = ConvSentenceEncoder(opt)

        enc_input_size = opt['num_feature_maps'] * len(opt['filter_sizes'])
        self.encoder = LSTMEncoder(opt, input_size=enc_input_size, bidirectional=True)
        enc_out_dim = 2 * opt['lstm_hidden_units'] if opt['lstm_bidirection'] else opt['lstm_hidden_units']
        self.decoder = PointerNetworkDecoder(opt, input_size=enc_out_dim)

    def forward(self, source, source_length, target=None, target_length=None):

        ### Get mask
        source_rep_mask = get_rep_mask(source_length)

        ### Word Embedding
        # batch_size x num_sentences x num_words x word_dim
        w_s = self.word_embedding(source)

        ### Sentence Encode
        # batch_size x num_sentences x (num_feature_maps * num_filters)
        s_s = self.cnn(w_s)

        ### Encode
        enc_outs, enc_states = self.encoder(s_s, source_length)

        ### Decode
        # Training phase
        if target is not None:
            return self.decoder(enc_outs, target, source_rep_mask, target_length)
        # Prediction phase
        # TODO: not implemented yet
        else:
            return
