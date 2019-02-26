import glob
import json

from typing import Dict, List

from overrides import overrides

from allennlp.common.tqdm import Tqdm
from allennlp.data import Token
from allennlp.data.fields import TextField, ListField, IndexField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data import Instance
from allennlp.data.dataset_readers.dataset_reader import DatasetReader
from allennlp.data.tokenizers.tokenizer import Tokenizer


@DatasetReader.register("cnn-dailymail")
class CnnDmReader(DatasetReader):

    def __init__(self,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None,
                 lazy: bool = False,
                 max_len: int = 100,
                 tqdm: bool = True,
                 mode: str = 'r'):
        super(CnnDmReader, self).__init__(lazy)
        self._tokenizer = tokenizer
        self._token_indexers = token_indexers or \
                               {"tokens": SingleIdTokenIndexer()}

        self._max_len = max_len
        self._tqdm = tqdm
        self._mode = mode   # e, a, r

    def _read(self, dir_path):
        file_path_list = glob.glob(dir_path + '/*.json')
        self.total_instances = len(file_path_list)
        if self._tqdm:
            file_path_list = Tqdm.tqdm(file_path_list)
        for example_num, file_path in enumerate(file_path_list):
            with open(file_path, 'r', encoding="utf-8") as file:
                data = json.loads(file.read())
            article = data['article']
            abstract = data['abstract']
            extracted = data['extracted']
            position = data['position']
            yield self.text_to_instance(article, abstract, extracted, position)

    def text_to_instance(self, article, abstract, extracted, position):
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
