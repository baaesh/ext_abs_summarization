import glob
import json
from tqdm import tqdm

from torch.utils.data import DataLoader

from data_utils.dataset import CnnDmDataset
from data_utils.vocab import Vocab


class CnnDm():

    def __init__(self, opt):
        super(CnnDm, self).__init__()
        self.opt = opt

        with open(opt['vocab_path'], 'r', encoding='utf-8') as file:
            vocab_list = [line.strip() for line in file.readlines()]
        self.vocab = Vocab.from_list(vocab_list)

        train_examples = glob.glob(opt['train_path'] + '/*.json')
        valid_examples = glob.glob(opt['valid_path'] + '/*.json')
        train_data = []
        valid_data = []

        print("Loading Training Data")
        for example in tqdm(train_examples):
            with open(example, 'r', encoding='utf-8') as file:
                train_data.append(json.load(file))
        print("Loading Validation Data")
        for example in tqdm(valid_examples):
            with open(example, 'r', encoding='utf-8') as file:
                valid_data.append(json.load(file))

        self.train_dataset = CnnDmDataset(train_data, self.vocab)
        self.valid_dataset = CnnDmDataset(valid_data, self.vocab)

        self.train_loader = DataLoader(dataset=self.train_dataset,
                                       batch_size=opt['batch_size'],
                                       shuffle=True,
                                       collate_fn=self.train_dataset.collate)
        self.valid_loader = DataLoader(dataset=self.valid_dataset,
                                       batch_size=opt['batch_size'],
                                       shuffle=False,
                                       collate_fn=self.valid_dataset.collate)
