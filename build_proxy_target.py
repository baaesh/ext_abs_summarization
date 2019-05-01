import glob
import json

import numpy as np
from nltk import word_tokenize
from tqdm import tqdm

from metric import rouge_L, rouge_n


def getProxyLabels(art, abs):
    scores = []
    positions = []
    indexes = [i for i in range(len(art))]
    for abs_sent in abs:
        if len(art) == 0:
            break
        rougeLs = np.array([rouge_L(art_sent, abs_sent, 'r') for art_sent in art])
        index = np.argmax(rougeLs)
        scores.append(rougeLs[index])
        positions.append(indexes[index])
        del art[index]
        del indexes[index]
    return positions, scores


def getExtOracle(art, abs):
    labels = []
    scores = []
    positions = []
    indexes = [i for i in range(len(art))]
    for abs_sent in abs:
        if len(art) == 0:
            break
        rouge1s = np.array([rouge_n(art_sent, abs_sent, n=1) for art_sent in art]).reshape(-1, 1)
        rouge2s = np.array([rouge_n(art_sent, abs_sent, n=2) for art_sent in art]).reshape(-1, 1)
        rougeLs = np.array([rouge_L(art_sent, abs_sent) for art_sent in art]).reshape(-1, 1)
        averages = np.concatenate((rouge1s, rouge2s, rougeLs), axis=1).mean(axis=1)
        index = np.argmax(averages)
        labels.append(art[index])
        scores.append(averages[index])
        positions.append(indexes[index])
        del art[index]
        del indexes[index]
    return labels, positions, scores


def buildProxyTargets():
    # test data doesn't need extracted labels
    dirs = ['train', 'valid', 'test']

    for dir in dirs:
        file_path_list = glob.glob('data/cnn-dailymail/' + dir + '/*.json')
        for file_path in tqdm(file_path_list):
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.loads(file.read())
            except Exception as e:
                print(file_path)
                print(e)
                return
            tokenized_art = [word_tokenize(art_sent) for art_sent in data['article']]
            tokenized_abs = [word_tokenize(abs_sent) for abs_sent in data['abstract']]
            positions, scores = getProxyLabels(tokenized_art, tokenized_abs)

            data['extracted'] = positions
            data['score'] = scores
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)


if __name__ == '__main__':
    buildProxyTargets()
