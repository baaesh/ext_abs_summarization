import os
import math
import glob
import json

from typing import Dict, List

from overrides import overrides

from allennlp.common.tqdm import Tqdm
from allennlp.data import Vocabulary
from allennlp.data.fields import TextField, ListField, IndexField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.tokenizer import Tokenizer
from allennlp.data.tokenizers.word_tokenizer import WordTokenizer
from allennlp.data.iterators.bucket_iterator import BucketIterator


class CnnDm():

    def __init__(self, opt):
        super(CnnDm, self).__init__()
        self.opt = opt
        wordTokenizer = WordTokenizer()
        token_indexers = {"tokens": SingleIdTokenIndexer(start_tokens=['@@BOS@@'],
                                                         end_tokens=['@@EOS@@'])}
        train_reader = CnnDmReader(tokenizer=wordTokenizer,
                                   token_indexers=token_indexers,
                                   lazy=True, mode=opt['mode'],
                                   type='train')
        valid_reader = CnnDmReader(tokenizer=wordTokenizer,
                                   token_indexers=token_indexers,
                                   lazy=True, mode=opt['mode'],
                                   type='valid', tqdm=False)
        self.train_instances = train_reader.read(opt['train_path'])
        self.valid_instances = valid_reader.read(opt['valid_path'])

        # Load or Build Vocab
        if os.path.isdir(opt['vocab_dir']) and os.listdir(opt['vocab_dir']):
            print("Loading Vocabulary")
            train_reader.set_total_instances(opt['train_path'])
            valid_reader.set_total_instances(opt['valid_path'])
            self.vocab = Vocabulary.from_files(opt['vocab_dir'])
            print("Finished")
        else:
            print("Building Vocabulary")
            self.vocab = Vocabulary.from_instances(self.train_instances)
            self.vocab.save_to_files(opt['vocab_dir'])
            print("Finished")

        # Iterator
        if opt['mode'] == 'a':
            sorting_keys = [('extracted', 'num_tokens')]
        else:
            sorting_keys = [('article', 'num_fields')]

        self.train_iterator = BucketIterator(sorting_keys=sorting_keys,
                                             batch_size=opt['batch_size'],
                                             track_epoch=True,
                                             max_instances_in_memory=math.ceil(
                                                 train_reader.total_instances * opt['lazy_ratio']))
        self.valid_iterator = BucketIterator(sorting_keys=sorting_keys,
                                             batch_size=opt['batch_size'],
                                             track_epoch=True)
        self.train_iterator.vocab = self.vocab
        self.valid_iterator.vocab = self.vocab


@DatasetReader.register("cnn-dailymail")
class CnnDmReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 tqdm: bool = True,
                 mode: str = 'r',
                 type: str = 'train'):
        super(CnnDmReader, self).__init__(lazy)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or \
                               {"tokens": SingleIdTokenIndexer()}
        self._tqdm = tqdm
        self._mode = mode   # extractor, abstracter, reinforcement
        self._type = type   # train, valid, test
        if type == 'test':
            assert mode == 'r'

    def set_total_instances(self, dir_path):
        file_path_list = glob.glob(dir_path + '/*.json')
        self.total_instances = len(file_path_list)

    def _read(self, dir_path):
        file_path_list = glob.glob(dir_path + '/*.json')
        self.total_instances = len(file_path_list)
        if self._tqdm:
            file_path_list = Tqdm.tqdm(file_path_list)
        for example_num, file_path in enumerate(file_path_list):
            with open(file_path, 'r', encoding="utf-8") as file:
                data = json.loads(file.read())
            if len(data['article']) <= 0 or len(data['abstract']) <= 0:    # emtpy file
                pass
            else:
                if type != 'test':      # test data doesn't have extracted sentences.
                    yield self.text_to_instance(data['article'], data['abstract'],
                                                data['extracted'], data['position'])
                else:
                    yield self.text_to_instance(data['article'], data['abstract'])

    def text_to_instance(self, article, abstract, extracted=None, position=None):
        if self._mode == 'e':
            article_field = self.process_article(article)
            position_field = self.process_position(position, article_field)
            fields = {'article': article_field, 'position': position_field}
            return Instance(fields)

        elif self._mode == 'a':
            extracted_field = self.process_extracted(extracted)
            abstract_field = self.process_abstract(abstract)
            fields = {'extracted': extracted_field, 'abstract': abstract_field}
            return Instance(fields)
        else:
            article_field = self.process_article(article)
            abstract_field = self.process_abstract(abstract)
            fields = {'article': article_field, 'abstract': abstract_field}
            return Instance(fields)

    def process_article(self, article):
        tokenized_article = []
        for art_sent in article:
            tokenized_art_sent = self._tokenizer.tokenize(art_sent)
            art_sent_field = TextField(tokenized_art_sent, self._token_indexers)
            tokenized_article.append(art_sent_field)
        return ListField(tokenized_article)

    def process_position(self, position, article_field):
        pos_field_list = [IndexField(pos, article_field) for pos in position]
        return ListField(pos_field_list)

    def process_extracted(self, extracted):
        tokenized_extracted = []
        for ext_sent in extracted:
            tokenized_ext_sent = self._tokenizer.tokenize(ext_sent)
            tokenized_extracted += tokenized_ext_sent
        return TextField(tokenized_extracted, self._token_indexers)

    def process_abstract(self, abstract):
        tokenized_abstract = []
        for abs_sent in abstract:
            tokenized_abs_sent = self._tokenizer.tokenize(abs_sent)
            tokenized_abstract += tokenized_abs_sent
        return TextField(tokenized_abstract, self._token_indexers)
