import glob
import json
from tqdm import tqdm

from torch.utils.data import DataLoader

from data_utils.dataset import CnnDmDataset
from data_utils.vocab import Vocab
from data_utils.word2vec import make_embedding


class CnnDm():

    def __init__(self, opt):
        super(CnnDm, self).__init__()
        self.opt = opt

        with open(opt['vocab_path'], 'r', encoding='utf-8') as file:
            vocab_list = [line.strip() for line in file.readlines()]
        self.vocab = Vocab.from_list(vocab_list)

        train_examples = glob.glob(opt['train_path'] + '/*.json')
        valid_examples = glob.glob(opt['valid_path'] + '/*.json')
        test_examples = glob.glob(opt['test_path'] + '/*.json')
        train_data = []
        valid_data = []
        test_data = []

        print("Loading Training Data")
        for example in tqdm(train_examples):
            with open(example, 'r', encoding='utf-8') as file:
                train_data.append(json.load(file))
        print("Loading Validation Data")
        for example in tqdm(valid_examples):
            with open(example, 'r', encoding='utf-8') as file:
                valid_data.append(json.load(file))
        print("Loading Test Data")
        for example in tqdm(test_examples):
            with open(example, 'r', encoding='utf-8') as file:
                test_data.append(json.load(file))

        print('Loading Word2Vec pretrained vectors')
        self.vectors, _ = make_embedding(self.vocab, 'data/word2vec/word2vec.128d.226k.bin')

        self.train_dataset = CnnDmDataset(opt, train_data, self.vocab, opt['mode'])
        self.valid_dataset = CnnDmDataset(opt, valid_data, self.vocab, opt['mode'])
        self.test_dataset = CnnDmDataset(opt, test_data, self.vocab, 't')

        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=opt['batch_size'],
                                       shuffle=True,
                                       collate_fn=self.train_dataset.collate)
        self.valid_loader = DataLoader(dataset=self.valid_dataset,
                                       batch_size=opt['batch_size'],
                                       shuffle=True,
                                       collate_fn=self.valid_dataset.collate)
        self.test_loader = DataLoader(dataset=self.test_dataset,
                                      batch_size=opt['batch_size'],
                                      shuffle=False,
                                      collate_fn=self.test_dataset.collate)
