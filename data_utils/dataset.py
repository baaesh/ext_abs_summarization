import copy

import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class CnnDmDataset(Dataset):

    def __init__(self, opt, data, vocab, mode='a'):
        """A `torch.utils.data.Dataset` object for CNN/Daily Mail data.

        Args:
            data (list[dict]): A list of dicts loaded from
                the preprocessed file.
            vocab (utils.vocab.Vocab): Vocab objects loaded from
                the vocab file.
        """
        self.opt = opt
        self._data = data
        self._vocab = vocab
        self._mode = mode

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def collate(self, batch):
        if self._mode == 'e':
            return self.full_collate(batch)
        elif self._mode == 'a':
            return self.abs_collate_(batch)
        elif self._mode == 'r':
            return self.full_collate(batch)
        else:
            return self.full_collate(batch)

    def to_tensor(self, seq):
        return torch.tensor(seq)

    def process_article(self, ex):
        sents = []
        sents_unk = []
        lens = []

        oov_idx = len(self._vocab)
        oov_tokens = {}

        for sent in ex['article']:
            tokens = sent.split()
            token_ids = []
            token_unk_ids = []
            for token in tokens:
                idx = self._vocab.stoi(token)
                if not self._vocab.has_word(token):
                    if token not in oov_tokens:
                        oov_tokens[token] = oov_idx
                        oov_idx += 1
                    token_ids.append(oov_tokens[token])
                else:
                    token_ids.append(idx)
                token_unk_ids.append(idx)
            length = len(token_unk_ids)
            assert len(token_ids) == len(token_unk_ids)

            # Cut too long sentences
            if length > self.opt['art_max_len']:
                token_ids = token_ids[:self.opt['art_max_len']]
                token_unk_ids = token_unk_ids[:self.opt['art_max_len']]
                length = len(token_ids)
            # Padding
            while (len(token_ids) < self.opt['art_max_len']):
                token_ids += [self._vocab.pad_id]
                token_unk_ids += [self._vocab.pad_id]
            assert len(token_ids) == len(token_unk_ids)
            sents.append(token_ids)
            sents_unk.append(token_unk_ids)
            lens.append(length)
        return sents, sents_unk, lens, oov_tokens

    def process_extracted(self, ex):
        text = []
        text_unk = []

        oov_idx = len(self._vocab)
        oov_tokens = {}

        for pos in ex['extracted']:
            sent = ex['article'][pos]
            tokens = sent.split()
            for token in tokens:
                idx = self._vocab.stoi(token)
                if not self._vocab.has_word(token):
                    if token not in oov_tokens:
                        oov_tokens[token] = oov_idx
                        oov_idx += 1
                    text.append(oov_tokens[token])
                else:
                    text.append(idx)
                text_unk.append(idx)
        assert len(text) == len(text_unk)
        return text, text_unk, len(text), oov_tokens

    def process_abstract(self, ex, oov_tokens):
        text = []
        text_unk = []
        for sent in ex['abstract']:
            tokens = sent.split()
            for token in tokens:
                idx = self._vocab.stoi(token)
                if not self._vocab.has_word(token) \
                        and token in oov_tokens:
                    text.append(oov_tokens[token])
                else:
                    text.append(idx)
                text_unk.append(idx)
        text.append(self._vocab.eos_id)
        text_unk.append(self._vocab.eos_id)
        assert len(text) == len(text_unk)
        return text, text_unk, len(text)

    def full_collate(self, batch):
        d = {'id': [],
             'article': {'sents': [], 'sents_unk': [], 'lens': [], 'origin': []},
             'target': {'position': [], 'len': []},
             'abstract': {'text': [], 'text_unk': [], 'len': [], 'origin': []},
             'oov_tokens': []}

        for ex in batch:
            # Article
            art_sents, art_sents_unk, art_lens, oov_tokens = self.process_article(ex)
            if len(art_sents) < 1:
                continue

            # Target
            target = ex['extracted']
            if len(target) < 1:
                continue

            # Abstract
            abs_text, abs_text_unk, abs_len = self.process_abstract(ex, oov_tokens)
            if abs_len < 1:
                continue
            d['id'].append(ex['id'])
            d['article']['sents'].append(self.to_tensor(art_sents))
            d['article']['sents_unk'].append(art_sents_unk)
            d['article']['lens'].append(self.to_tensor(art_lens))
            d['article']['origin'].append(ex['article'])
            d['oov_tokens'].append(oov_tokens)
            d['abstract']['text'].append(self.to_tensor(abs_text))
            d['abstract']['text_unk'].append(self.to_tensor(abs_text_unk))
            d['abstract']['len'].append(abs_len)
            d['abstract']['origin'].append(ex['abstract'])
            d['target']['position'].append(self.to_tensor(target))
            d['target']['len'].append(len(target))

        d['article']['sents'] = pad_sequence(
            d['article']['sents'], batch_first=True, padding_value=self._vocab.pad_id)
        d['article']['sents_unk'] = pad_sequence(
            d['article']['sents_unk'], batch_first=True, padding_value=self._vocab.pad_id)
        d['article']['lens'] = pad_sequence(
            d['article']['lens'], batch_first=True, padding_value=0)
        d['abstract']['text'] = pad_sequence(
            d['abstract']['text'], batch_first=True, padding_value=self._vocab.pad_id)
        d['abstract']['text_unk'] = pad_sequence(
            d['abstract']['text_unk'], batch_first=True, padding_value=self._vocab.pad_id)
        d['abstract']['len'] = self.to_tensor(d['abstract']['len'])
        d['target']['position'] = pad_sequence(
            d['target']['position'], batch_first=True, padding_value=self._vocab.pad_id)
        d['target']['len'] = self.to_tensor(d['target']['len'])

        return d

    def abs_collate_(self, batch):
        d = {'id': [],
             'extracted': {'text': [], 'text_unk': [], 'len': []},
             'abstract': {'text': [], 'text_unk': [], 'len': [], 'origin': []},
             'oov_tokens': []}

        for ex in batch:
            # Extracted
            ext_text, ext_text_unk, ext_len, oov_tokens = self.process_extracted(ex)

            # Abstract
            abs_text, abs_text_unk, abs_len = self.process_abstract(ex, oov_tokens)

            if len(ext_text) >= 220 or len(abs_text) >= 125:
                continue
            if len(ext_text) == 0 or len(abs_text) == 0:
                continue

            d['id'].append(ex['id'])
            d['extracted']['text'].append(self.to_tensor(ext_text))
            d['extracted']['text_unk'].append(self.to_tensor(ext_text_unk))
            d['extracted']['len'].append(ext_len)
            d['oov_tokens'].append(oov_tokens)
            d['abstract']['text'].append(self.to_tensor(abs_text))
            d['abstract']['text_unk'].append(self.to_tensor(abs_text_unk))
            d['abstract']['length'].append(abs_len)

        d['extracted']['text'] = pad_sequence(
            d['extracted']['text'], batch_first=True, padding_value=self._vocab.pad_id)
        d['extracted']['text_unk'] = pad_sequence(
            d['extracted']['text_unk'], batch_first=True, padding_value=self._vocab.pad_id)
        d['extracted']['len'] = self.to_tensor(d['extracted']['len'])
        d['abstract']['text'] = pad_sequence(
            d['abstract']['text'], batch_first=True, padding_value=self._vocab.pad_id)
        d['abstract']['text_unk'] = pad_sequence(
            d['abstract']['text_unk'], batch_first=True, padding_value=self._vocab.pad_id)
        d['abstract']['len'] = self.to_tensor(d['abstract']['len'])

        return d
