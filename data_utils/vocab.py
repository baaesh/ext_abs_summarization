class Vocab(object):

    def __init__(self, unk_token='<unk>', pad_token='<pad>',
                 bos_token='<bos>', eos_token='<eos>'):
        self._unk_token = unk_token
        self._pad_token = pad_token
        self._bos_token = bos_token
        self._eos_token = eos_token
        self._vocab_dict = {}
        if pad_token is not None:
            self._vocab_dict[pad_token] = len(self._vocab_dict)
        if unk_token is not None:
            self._vocab_dict[unk_token] = len(self._vocab_dict)
        if bos_token is not None:
            self._vocab_dict[bos_token] = len(self._vocab_dict)
        if eos_token is not None:
            self._vocab_dict[eos_token] = len(self._vocab_dict)
        self._reverse_vocab_dict = {i: s for s, i in self._vocab_dict.items()}
        assert len(self._vocab_dict) == len(self._reverse_vocab_dict)

    @classmethod
    def from_list(cls, tokens, unk_token='<unk>', pad_token='<pad>',
                  bos_token='<bos>', eos_token='<eos>'):
        vocab = cls(unk_token=unk_token, pad_token=pad_token,
                    bos_token=bos_token, eos_token=eos_token)
        for t in tokens:
            vocab.add(t)
        return vocab

    @property
    def unk_word(self):
        return self._unk_token

    @property
    def unk_id(self):
        return self.stoi(self.unk_word)

    @property
    def pad_word(self):
        return self._pad_token

    @property
    def pad_id(self):
        return self.stoi(self.pad_word)

    @property
    def bos_word(self):
        return self._bos_token

    @property
    def bos_id(self):
        return self.stoi(self.bos_word)

    @property
    def eos_word(self):
        return self._eos_token

    @property
    def eos_id(self):
        return self.stoi(self.eos_word)

    def __len__(self):
        return len(self._vocab_dict)

    def stoi(self, s):
        if s not in self._vocab_dict and self._unk_token is not None:
            return self._vocab_dict[self._unk_token]
        return self._vocab_dict[s]

    def itos(self, i):
        return self._reverse_vocab_dict[i]

    def add(self, s):
        if s in self._vocab_dict:
            return
        self._vocab_dict[s] = len(self._vocab_dict)
        self._reverse_vocab_dict[self._vocab_dict[s]] = s

    def has_word(self, s):
        return s in self._vocab_dict
