from os.path import basename

import gensim
import torch
from torch import nn
from tqdm import tqdm


def make_embedding(vocab, w2v_file, initializer=None):
    attrs = basename(w2v_file).split('.')  #word2vec.{dim}d.{vsize}k.bin
    w2v = gensim.models.Word2Vec.load(w2v_file).wv
    vocab_size = len(vocab)
    emb_dim = int(attrs[-3][:-1])
    embedding = nn.Embedding(vocab_size, emb_dim).weight
    if initializer is not None:
        initializer(embedding)

    oovs = []
    with torch.no_grad():
        for i in tqdm(range(len(vocab))):
            # NOTE: id2word can be list or dict
            if i == vocab.bos_id:
                embedding[i, :] = torch.Tensor(w2v['<s>'])
            elif i == vocab.eos_id:
                embedding[i, :] = torch.Tensor(w2v[r'<\s>'])
            elif vocab.itos(i) in w2v:
                embedding[i, :] = torch.Tensor(w2v[vocab.itos(i)])
            else:
                oovs.append(i)
    return embedding, oovs