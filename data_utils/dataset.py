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
             'extracted': {'words': [], 'words_extended': [], 'length': []},
             'abstract': {'words': [], 'words_extended': [], 'length': []},
             'oov_tokens': []}

        for ex in batch:
            # Extracted
            oov_idx = len(self._vocab)
            oov_tokens = {}
            extracted = []
            extracted_extended = []     # extended for pointer generator
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
            abstract_extended = [self._vocab.bos_id]    # extended for pointer generator
            for abs_sent in ex['abstract']:
                tokens = abs_sent.split()
                for token in tokens:
                    idx = self._vocab.stoi(token)
                    if not self._vocab.has_word(token)\
                            and token in oov_tokens:
                        abstract_extended.append(oov_tokens[token])
                    else:
                        abstract_extended.append(idx)
                    abstract.append(idx)
            abstract.append(self._vocab.eos_id)
            abstract_extended.append(self._vocab.eos_id)
            assert len(abstract) == len(abstract_extended)

            if len(extracted) >= 230 or len(abstract) >= 130:
                continue
            if len(extracted) == 0 or len(abstract) == 0:
                continue

            d['id'].append(ex['id'])
            d['extracted']['words'].append(to_tensor(extracted))
            d['extracted']['words_extended'].append(to_tensor(extracted_extended))
            d['extracted']['length'].append(len(extracted))
            d['oov_tokens'].append(oov_tokens)
            d['abstract']['words'].append(to_tensor(abstract))
            d['abstract']['words_extended'].append(to_tensor(abstract_extended))
            d['abstract']['length'].append(len(abstract))

        d['extracted']['words'] = pad_sequence(
            d['extracted']['words'], batch_first=True, padding_value=self._vocab.pad_id)
        d['extracted']['words_extended'] = pad_sequence(
            d['extracted']['words_extended'], batch_first=True, padding_value=self._vocab.pad_id)
        d['extracted']['length'] = to_tensor(d['extracted']['length'])
        d['abstract']['words'] = pad_sequence(
            d['abstract']['words'], batch_first=True, padding_value=self._vocab.pad_id)
        d['abstract']['words_extended'] = pad_sequence(
            d['abstract']['words_extended'], batch_first=True, padding_value=self._vocab.pad_id)
        d['abstract']['length'] = to_tensor(d['abstract']['length'])

        return d
