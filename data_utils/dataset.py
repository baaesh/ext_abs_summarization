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
            return self.ext_collate(batch)
        elif self._mode == 'a':
            return self.abs_collate(batch)
        else:
            return self.rein_collate(batch)

    def to_tensor(self, seq):
        return torch.tensor(seq)

    def ext_collate(self, batch):
        d = {'id': [],
             'article': {'sentences': [], 'length': []},
             'target': {'positions': [], 'length': []}}

        for ex in batch:
            # Article
            article_sents = []
            article_sents.append([self._vocab.pad_id] * self.opt['art_max_len'])
            for art_sent in ex['article']:
                tokens = art_sent.split()
                token_ids = [self._vocab.stoi(token) for token in tokens]

                # Cut too long sentences
                if len(token_ids) > self.opt['art_max_len']:
                    token_ids = token_ids[:self.opt['art_max_len']]
                # Padding
                while (len(token_ids) < self.opt['art_max_len']):
                    token_ids += [self._vocab.pad_id]
                article_sents.append(token_ids)

            if len(article_sents) == 1 or len(ex['position']) == 0:
                continue

            d['id'].append(ex['id'])
            d['article']['sentences'].append(self.to_tensor(article_sents))
            d['article']['length'].append(len(article_sents))
            # position index starts from 1
            d['target']['positions'].append(self.to_tensor(ex['position']) + 1)
            d['target']['length'].append(len(ex['position']))

        d['article']['sentences'] = pad_sequence(
            d['article']['sentences'], batch_first=True, padding_value=self._vocab.pad_id)
        d['article']['length'] = self.to_tensor(d['article']['length'])
        d['target']['positions'] = pad_sequence(
            d['target']['positions'], batch_first=True, padding_value=self._vocab.pad_id)
        d['target']['length'] = self.to_tensor(d['target']['length'])

        return d

    def abs_collate(self, batch):
        d = {'id': [],
             'extracted': {'words': [], 'words_extended': [], 'length': []},
             'abstract': {'words': [], 'words_extended': [], 'length': []},
             'oov_tokens': []}

        for ex in batch:
            # Extracted
            oov_idx = len(self._vocab)
            oov_tokens = {}
            extracted = []
            extracted_extended = []  # extended for pointer generator
            for ext_sent in ex['extracted']:
                tokens = ext_sent.split()
                for token in tokens:
                    idx = self._vocab.stoi(token)
                    if not self._vocab.has_word(token):
                        if token not in oov_tokens:
                            oov_tokens[token] = oov_idx
                            oov_idx += 1
                        extracted_extended.append(oov_tokens[token])
                    else:
                        extracted_extended.append(idx)
                    extracted.append(idx)
            assert len(extracted) == len(extracted_extended)

            # Abstract
            abstract = [self._vocab.bos_id]
            abstract_extended = [self._vocab.bos_id]  # extended for pointer generator
            for abs_sent in ex['abstract']:
                tokens = abs_sent.split()
                for token in tokens:
                    idx = self._vocab.stoi(token)
                    if not self._vocab.has_word(token) \
                            and token in oov_tokens:
                        abstract_extended.append(oov_tokens[token])
                    else:
                        abstract_extended.append(idx)
                    abstract.append(idx)
            abstract.append(self._vocab.eos_id)
            abstract_extended.append(self._vocab.eos_id)
            assert len(abstract) == len(abstract_extended)

            if len(extracted) >= 220 or len(abstract) >= 125:
                continue
            if len(extracted) == 0 or len(abstract) == 0:
                continue

            d['id'].append(ex['id'])
            d['extracted']['words'].append(self.to_tensor(extracted))
            d['extracted']['words_extended'].append(self.to_tensor(extracted_extended))
            d['extracted']['length'].append(len(extracted))
            d['oov_tokens'].append(oov_tokens)
            d['abstract']['words'].append(self.to_tensor(abstract))
            d['abstract']['words_extended'].append(self.to_tensor(abstract_extended))
            d['abstract']['length'].append(len(abstract))

        d['extracted']['words'] = pad_sequence(
            d['extracted']['words'], batch_first=True, padding_value=self._vocab.pad_id)
        d['extracted']['words_extended'] = pad_sequence(
            d['extracted']['words_extended'], batch_first=True, padding_value=self._vocab.pad_id)
        d['extracted']['length'] = self.to_tensor(d['extracted']['length'])
        d['abstract']['words'] = pad_sequence(
            d['abstract']['words'], batch_first=True, padding_value=self._vocab.pad_id)
        d['abstract']['words_extended'] = pad_sequence(
            d['abstract']['words_extended'], batch_first=True, padding_value=self._vocab.pad_id)
        d['abstract']['length'] = self.to_tensor(d['abstract']['length'])

        return d

    def rein_collate(self, batch):
        d = {'id': [],
             'article': {'sentences': [], 'sentences_extended': [],
                         'num_sentence': [], 'length': []},
             'abstract': {'words': [], 'words_extended': [], 'length': []},
             'oov_tokens': []}

        for ex in batch:
            # Article
            oov_idx = len(self._vocab)
            oov_tokens = {}
            article_sents = []
            article_sents_extended = []
            lengths = []
            article_sents.append([self._vocab.pad_id] * self.opt['art_max_len'])
            for art_sent in ex['article']:
                tokens = art_sent.split()
                token_ids = []
                token_extended_ids = []
                for token in tokens:
                    idx = self._vocab.stoi(token)
                    if not self._vocab.has_word(token):
                        if token not in oov_tokens:
                            oov_tokens[token] = oov_idx
                            oov_idx += 1
                        token_extended_ids.append(oov_tokens[token])
                    else:
                        token_extended_ids.append(idx)
                    token_ids.append(idx)
                length = len(token_ids)
                assert len(token_ids) == len(token_extended_ids)

                # Cut too long sentences
                if length > self.opt['art_max_len']:
                    token_ids = token_ids[:self.opt['art_max_len']]
                    token_extended_ids = token_extended_ids[:self.opt['art_max_len']]
                    length = len(token_ids)
                # Padding
                while (len(token_ids) < self.opt['art_max_len']):
                    token_ids += [self._vocab.pad_id]
                    token_extended_ids += [self._vocab.pad_id]
                assert len(token_ids) == len(token_extended_ids)
                article_sents.append(token_ids)
                article_sents_extended.append(token_extended_ids)
                lengths.append(length)

            # Abstract
            abstract = [self._vocab.bos_id]
            abstract_extended = [self._vocab.bos_id]  # extended for pointer generator
            for abs_sent in ex['abstract']:
                tokens = abs_sent.split()
                for token in tokens:
                    idx = self._vocab.stoi(token)
                    if not self._vocab.has_word(token) \
                            and token in oov_tokens:
                        abstract_extended.append(oov_tokens[token])
                    else:
                        abstract_extended.append(idx)
                    abstract.append(idx)
            abstract.append(self._vocab.eos_id)
            abstract_extended.append(self._vocab.eos_id)
            assert len(abstract) == len(abstract_extended)

            if len(article_sents) == 1 or len(ex['position']) == 0:
                continue
            if len(abstract) == 0 or len(abstract) >= 125:
                continue

            d['id'].append(ex['id'])
            d['article']['sentences'].append(self.to_tensor(article_sents))
            d['article']['sentences_extended'].append(self.to_tensor(article_sents_extended))
            d['article']['num_sentence'].append(len(article_sents))
            d['article']['length'].append(self.to_tensor(lengths))
            d['oov_tokens'].append(oov_tokens)
            d['abstract']['words'].append(self.to_tensor(abstract))
            d['abstract']['words_extended'].append(self.to_tensor(abstract_extended))
            d['abstract']['length'].append(len(abstract))

        d['article']['sentences'] = pad_sequence(
            d['article']['sentences'], batch_first=True, padding_value=self._vocab.pad_id)
        d['article']['sentences_extended'] = pad_sequence(
            d['article']['sentences_extended'], batch_first=True, padding_value=self._vocab.pad_id)
        d['article']['num_sentence'] = self.to_tensor(d['article']['num_sentence'])
        d['abstract']['words'] = pad_sequence(
            d['abstract']['words'], batch_first=True, padding_value=self._vocab.pad_id)
        d['abstract']['words_extended'] = pad_sequence(
            d['abstract']['words_extended'], batch_first=True, padding_value=self._vocab.pad_id)
        d['abstract']['length'] = self.to_tensor(d['abstract']['length'])

        return d