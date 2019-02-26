import glob
import json

import numpy as np
from nltk import word_tokenize

from metric import rouge_L


def getProxyLabels(art, abs):
    labels = []
    scores = []
    for abs_sent in abs:
        if len(art) == 0:
            break
        rouges = np.array([rouge_L(art_sent, abs_sent, mode='r') for art_sent in art])
        index = np.argmax(rouges)
        labels.append(art[index])
        scores.append(rouges[index])
        del art[index]
    return labels, scores


def buildProxyTargets():
    # test data doesn't need extracted labels
    dirs = ['train', 'valid']

    for dir in dirs:
        file_path_list = glob.glob('data/cnn-dailymail/' + dir + '/*.json')
        for file_path in file_path_list:
            print(file_path)
            try:
                with open(file_path, 'r', encoding='utf-8') as file:
                    data = json.loads(file.read())
            except:
                print(file_path)
                return
            tokenized_art = [word_tokenize(art_sent) for art_sent in data['article']]
            tokenized_abs = [word_tokenize(abs_sent) for abs_sent in data['abstract']]
            labels, scores = getProxyLabels(tokenized_art, tokenized_abs)

            data['extracted'] = [' '.join(label_sen) for label_sen in labels]
            data['score'] = scores
            with open(file_path, 'w', encoding='utf-8') as file:
                json.dump(data, file, indent=4)


if __name__ == '__main__':
    buildProxyTargets()