import json
from tqdm import tqdm
from trainer import util
from collections import OrderedDict
from typing import List
from transformers import BertTokenizer
from trainer.entities import Dataset,Entity,EntityType,Sentiment,sentimentType
import numpy as np
class JsonInputReader():
    def __init__(self,types_path: str,tokenizer: BertTokenizer, neg_entity_count: int = None, neg_senti_count: int = None, max_span_size: int = None):

        types = json.load(open(types_path), object_pairs_hook=OrderedDict)  # entity + sentiments types
        self._entity_types = OrderedDict()
        self._idx2entity_type = OrderedDict()
        self._sentiment_types = OrderedDict()
        self._idx2sentiment_type = OrderedDict()
        self._datasets = dict()
        self._neg_entity_count = neg_entity_count
        self._neg_senti_count = neg_senti_count
        self._max_span_size = max_span_size
        self._tokenizer = tokenizer

        # entities
        # add 'None' entity type
        none_entity_type = EntityType('None', 0, 'None', 'No Entity')
        self._entity_types['None'] = none_entity_type
        self._idx2entity_type[0] = none_entity_type

        # specified entity types
        for i, (key, v) in enumerate(types['entities'].items()):
            entity_type = EntityType(key, i + 1, v['short'], v['verbose'])
            self._entity_types[key] = entity_type
            self._idx2entity_type[i + 1] = entity_type
        """
        self._entity_types:
                entity_types = {OrderedDict: 3} OrderedDict([('None', <trainer.entities.EntityType object at 0x000001D94AA3EE30>), ('target', <trainer.entities.EntityType object at 0x000001D94AA3CC70>), ('opinion', <trainer.entities.EntityType object at 0x000001D94AA3D1E0>)])
                 'None' = {EntityType} <trainer.entities.EntityType object at 0x000001D94AA3EE30>
                 'target' = {EntityType} <trainer.entities.EntityType object at 0x000001D94AA3CC70>
                 'opinion' = {EntityType} <trainer.entities.EntityType object at 0x000001D94AA3D1E0>
                  __len__ = {int} 3
        """

        # sentiments
        # add 'None' sentiment type
        none_sentiment_type = sentimentType('None', 0, 'None', 'No Sentiment')
        self._sentiment_types['None'] = none_sentiment_type
        self._idx2sentiment_type[0] = none_sentiment_type

        # specified sentiment types
        for i, (key, v) in enumerate(types['sentiment'].items()):
            sentiment_type = sentimentType(key, i + 1, v['short'], v['verbose'], v['symmetric'])
            self._sentiment_types[key] = sentiment_type
            self._idx2sentiment_type[i + 1] = sentiment_type

    def read(self,dataset_paths):
        for dataset_label,dataset_path in dataset_paths.items():
            # print(dataset_label,dataset_path)
            dataset = Dataset(dataset_label, self._sentiment_types, self._entity_types, self._neg_entity_count,
                              self._neg_senti_count, self._max_span_size)
            self._parse_dataset(dataset_path, dataset)
            self._datasets[dataset_label] = dataset
            # print(dataset.__getitem__(0))

    def _parse_dataset(self,dataset_path, dataset):
        sentences = json.load(open(dataset_path))
        if len(sentences) % 16 != 0:
            num_to_remove = len(sentences) % 16
            sentences = sentences[:-num_to_remove]
        for sentence in tqdm(sentences,desc="Parse dataset '%s'"% dataset.label):
            # print(document)
            self._parse_sentence(sentence,dataset)

    def _parse_sentence(self, sen, dataset):
        jdependency = sen['dependency']
        jpos = sen['pos']
        # print("jdependency:",jdependency)
        jtokens = sen['tokens']
        jsentiments = sen['sentiments']
        jentities = sen['entities']
        sen_id = sen['orig_id']
        sen_tokens, sen_encoding, adj = self._parse_tokens(jtokens,jdependency, dataset)

        entities = self._parse_entities(jentities, sen_tokens, dataset)

        sentiments = self._parse_sentiments(jsentiments, entities, dataset,sen_id)

        sentence = dataset.create_sentence(sen_tokens, entities, sentiments, sen_encoding,adj)
        # print("sen_tokens:",sen_tokens)
        # return sentence

    def tree_to_adj(self,dependency,len_sen, self_loop=True, directed=False):
        ret = np.zeros((len_sen, len_sen), dtype=np.float32)
        for k in range(1, len(dependency)):
            ral = dependency[k]
            r, i, j = ral[0], ral[1], ral[2]
            # print(r, i, j)
            ret[i - 1, j - 1] = 1
        if not directed:
            ret = ret + ret.T
            # print(ret.T)
        if self_loop:
            ret += np.eye(ret.shape[0])
        return ret
    def change_dep(self,dependency,idx,l):
        for k in range(1, len(dependency)):
            ral = dependency[k]
            r, i, j = ral[0], ral[1], ral[2]
            if i > idx:
                ral[1]+=l
            if j > idx:
                ral[2]+=l
        return dependency

    def _parse_tokens(self, jtokens,jdependency, dataset):
        # print()
        # print(jdependency)
        sen_tokens = []
        # Full sentence encoding including special tokens ([CLS] and [SEP]) and byte pair encoding of the original token
        sen_encoding = [self._tokenizer.convert_tokens_to_ids('[CLS]')]
        sen = ""
        # parse tokens
        for i, token_phrase in enumerate(jtokens):
            sen += token_phrase+" "

            token_encoding = self._tokenizer.encode(token_phrase, add_special_tokens=False)
            span_start, span_end = (len(sen_encoding), len(sen_encoding) + len(token_encoding))

            token = dataset.create_token(i, span_start, span_end, token_phrase)

            sen_tokens.append(token)
            sen_encoding += token_encoding
            # print(i, token_phrase,token_encoding)
            l=len(token_encoding)
            if l > 1:
                idx = i+1
                jdependency = self.change_dep(dependency=jdependency,idx=idx,l=l-1)
        # print()
        # print(">>>:",jdependency)

        sen_encoding += [self._tokenizer.convert_tokens_to_ids('[SEP]')]
        adj = self.tree_to_adj(jdependency, len(sen_encoding))

        return sen_tokens, sen_encoding, adj

    def _parse_entities(self, jentities, doc_tokens, dataset) -> List[Entity]:
        entities = []

        for entity_idx, jentity in enumerate(jentities):
            # print(entity_idx,jentity)
            entity_type = self._entity_types[jentity['type']]
            start, end = jentity['start'], jentity['end']

            # create entity mention
            tokens = doc_tokens[start:end]
            phrase = " ".join([t.phrase for t in tokens])
            entity = dataset.create_entity(entity_type, tokens, phrase)
            entities.append(entity)

        return entities

    def _parse_sentiments(self, jsentiments, entities, dataset,sen_id) -> List[Sentiment]:
        sentiments = []

        for jsentiment in jsentiments:
            sentiment_type = self._sentiment_types[jsentiment['type']]

            head_idx = jsentiment['head']
            tail_idx = jsentiment['tail']

            head = entities[head_idx]
            tail = entities[tail_idx]
            try:
                reverse = int(tail.tokens[0].index) < int(head.tokens[0].index)
            except:
                print(sen_id,jsentiment,head,">>>>",tail)

            if sentiment_type.symmetric and reverse:
                head, tail = util.swap(head, tail)

            sentiment = dataset.create_sentiment(sentiment_type, head_entity=head, tail_entity=tail, reverse=reverse)
            sentiments.append(sentiment)

        return sentiments

    def get_entity_type(self, idx) -> EntityType:
        entity = self._idx2entity_type[idx]
        return entity

    def get_sentiment_type(self, idx) -> sentimentType:
        sentiment = self._idx2sentiment_type[idx]
        return sentiment


    def get_dataset(self, label) -> Dataset:
        return self._datasets[label]

    @property
    def entity_types(self):
        return self._entity_types

    @property
    def sentiment_types(self):
        return self._sentiment_types

    @property
    def sentiment_type_count(self):
        return len(self._sentiment_types)

    @property
    def entity_type_count(self):
        return len(self._entity_types)