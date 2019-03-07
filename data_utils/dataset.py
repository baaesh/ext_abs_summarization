import numpy as np
import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class CnnDmDataset(Dataset):

    def __init__(self, data, vocab, mode='a'):
        """A `torch.utils.data.Dataset` object for CNN/Daily Mail data.

        Args:
            data (list[dict]): A list of dicts loaded from
                the preprocessed file.
            vocab (utils.vocab.Vocab): Vocab objects loaded from
                the vocab file.
        """
        self._data = data
        self._vocab = vocab
        self._mode = mode

    def __getitem__(self, index):
        return self._data[index]

    def __len__(self):
        return len(self._data)

    def collate(self, batch):

        def to_tensor(seq):
            return torch.tensor(seq)

        d = {'id': [],
             'extracted': [],
             'abstract': [],
             'oov_tokens': []}

        for ex in batch:
            # ID
            d['id'].append(ex['id'])

            # Extracted
            oov_idx = self._vocab.__len__()
            oov_tokens = {}
            extracted = []
            for ext_sent in ex['extracted']:
                tokens = ext_sent.split()
                for token in tokens:
                    idx = self._vocab.stoi(token)
                    if not self._vocab.has_word(token):
                        if token not in oov_tokens:
                            oov_tokens[token] = oov_idx
                            oov_idx += 1
                        idx = oov_tokens[token]
                    extracted.append(idx)
            d['extracted'].append(to_tensor(extracted))
            d['oov_tokens'].append(oov_tokens)

            # Abstract
            abstract = [self._vocab.bos_id]
            for abs_sent in ex['abstract']:
                tokens = abs_sent.split()
                for token in tokens:
                    idx = self._vocab.stoi(token)
                    if not self._vocab.has_word(token)\
                            and token in oov_tokens:
                        idx = oov_tokens[token]
                    abstract.append(idx)
            abstract.append(self._vocab.eos_id)
            d['abstract'].append(to_tensor(abstract))

        d['extracted'] = pad_sequence(
            d['extracted'], batch_first=True, padding_value=self._vocab.pad_id)
        d['abstract'] = pad_sequence(
            d['abstract'], batch_first=True, padding_value=self._vocab.pad_id)

        return d
