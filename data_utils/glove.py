import os

import numpy as np
import torch

from vocab import Vocab


def load_glove(path, vocab):
    word_vectors = dict()
    emb_dim = -1
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            word, *values = line.split()
            try:
                if vocab.has_word(word):
                    if word in word_vectors:
                        # Let's use the first occurrence only.
                        continue
                    word_vector = np.array([float(v) for v in values])
                    word_vectors[word] = word_vector
                    emb_dim = len(values)
            except ValueError:
                # 840D GloVe file has some encoding errors...
                # I think they can be ignored.
                continue
    glove_weight = np.zeros((len(vocab), emb_dim))
    for word in word_vectors:
        word_index = vocab.stoi(word)
        glove_weight[word_index, :] = word_vectors[word]
    glove_weight[vocab.unk_id] = 0
    glove_weight[vocab.pad_id] = 0
    return glove_weight


def dump_data(data, path):
    print(path)
    torch.save(data, path)


if __name__ == '__main__':
    vocab_path = '../data/cnn-dailymail/vocab/tokens.txt'
    with open(vocab_path, 'r', encoding='utf-8') as file:
        vocab_list = [line.strip() for line in file.readlines()]
    vocab = Vocab.from_list(vocab_list)

    glove_path = '../data/glove/glove.840B.300d.txt'
    out_path = '../data/glove/'
    print('Constructing GloVe embeddings...')
    glove_embeddings = load_glove(path=glove_path, vocab=vocab)
    glove_embeddings = torch.from_numpy(glove_embeddings).float()
    dump_data(data=glove_embeddings, path=os.path.join(out_path, 'glove.pth'))
