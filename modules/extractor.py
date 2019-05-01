import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence

from modules.convolution import ConvEncoder
from modules.encoder import LSTMEncoder
from modules.decoder import PointerNetworkDecoder, ConditionalPointerNetworkDecoder
from modules.mask import get_rep_mask, get_rep_mask_2d


# Chen and Bansal, 2018
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

        self.cnn = ConvEncoder(opt)

        enc_input_size = opt['num_feature_maps'] * len(opt['filter_sizes'])
        self.encoder = LSTMEncoder(opt, input_size=enc_input_size, bidirectional=True)
        enc_out_dim = 2 * opt['lstm_hidden_units']
        self.decoder = PointerNetworkDecoder(opt, input_size=enc_out_dim)

    def forward(self, source, num_sents, num_words=None, target=None, target_length=None):

        ### Get mask
        source_rep_mask = get_rep_mask(num_sents)

        ### Word Embedding
        # batch_size x num_sentences x num_words x word_dim
        w_s = self.word_embedding(source)

        ### Sentence Encode
        # batch_size x num_sentences x (num_feature_maps * num_filters)
        s_s = self.cnn(w_s)

        ### Encode
        enc_outs, _ = self.encoder(s_s, num_sents)

        ### Decode
        # Training phase
        if target is not None:
            return self.decoder(enc_outs, target, source_rep_mask, target_length)
        # Prediction phase
        else:
            return self.decoder.predict(enc_outs, source_rep_mask), enc_outs


# Xu and Durrett, 2019
class HierarchicalPointerNetwork(nn.Module):

    def __init__(self, opt, pad_id, vectors=None):
        super(HierarchicalPointerNetwork, self).__init__()
        self.opt = opt
        self.pad_id = pad_id

        self.word_embedding = nn.Embedding(opt['vocab_size'], opt['word_dim'],
                                           padding_idx=pad_id)
        if vectors is not None:
            self.word_embedding.weight.data.copy_(vectors)
        if opt['fix_embedding']:
            self.word_embedding.weight.requires_grad = False

        self.lstm_word = nn.LSTM(input_size=opt['word_dim'],
                                 hidden_size=opt['lstm_hidden_units'],
                                 num_layers=opt['lstm_num_layers'],
                                 bidirectional=True,
                                 batch_first=True)
        self.cnn_sent = ConvEncoder(opt, input_dim=opt['lstm_hidden_units'] * 2, mode='s')
        self.lstm_sent = LSTMEncoder(opt, input_size=opt['num_feature_maps'] * len(opt['filter_sizes']),
                                     bidirectional=True)
        self.cnn_doc = ConvEncoder(opt, input_dim=opt['lstm_hidden_units'] * 2, mode='d')
        enc_out_dim = opt['num_feature_maps'] * len(opt['filter_sizes'])
        self.decoder = ConditionalPointerNetworkDecoder(opt, input_size=enc_out_dim)

    def forward(self, source, num_sents, num_words, target=None, target_length=None):
        batch_size, num_sentences, _ = source.size()

        ### Get mask
        source_rep_mask = get_rep_mask(num_sents)
        source_context_mask = get_rep_mask_2d(num_words, max_length=self.opt['art_max_len'])

        ### Word Embedding
        # batch_size x num_sentences x num_words x word_dim
        w_s = self.word_embedding(source)

        ### Word Contextualize
        c_s = []
        for i in range(batch_size):
            c_i, _ = self.lstm_word(w_s[i])
            c_s.append(c_i)
        # batch_size x num_sentences x num_words x hidden_dim
        c_s = torch.stack(c_s, dim=0) * source_context_mask.unsqueeze(-1).float()

        ### Sentence Encode
        # batch_size x num_sentences x (num_feature_maps * num_filters)
        h_s = self.cnn_sent(c_s)

        ### Sentence Contextualize
        # batch_size x num_sentences x hidden_dim
        hc_s, _ = self.lstm_sent(h_s, num_sents)

        ### Document Encode
        # batch_size x (num_feature_maps * num_filters)
        d_s = self.cnn_doc(hc_s)

        ### Decode
        # Training phase
        if target is not None:
            return self.decoder(h_s, d_s, target, source_rep_mask, target_length)
        # Prediction phase
        else:
            return self.decoder.predict(h_s, d_s, source_rep_mask), h_s
