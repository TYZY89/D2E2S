from transformers import BertModel
from transformers import BertPreTrainedModel
from transformers import BertConfig
from torch import nn as nn
import torch
from trainer import util, sampling
import os
import math
from models.Syn_GCN import GCN
from models.Sem_GCN import SemGCN
from models.Attention_Module import SelfAttention
from models.TIN_GCN import TIN, FeatureStacking
import torch.nn.functional as F
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from models.Channel_Fusion import Orthographic_projection_fusion, TextCentredSP


USE_CUDA = torch.cuda.is_available()

def get_token(h: torch.tensor, x: torch.tensor, token: int):
    """ Get specific token embedding (e.g. [CLS]) """
    emb_size = h.shape[-1]

    token_h = h.view(-1, emb_size)
    flat = x.contiguous().view(-1)

    # get contextualized embedding of given token
    token_h = token_h[flat == token, :]

    return token_h

class D2E2SModel(BertPreTrainedModel):
    VERSION = '1.1'
    def __init__(self, config: BertConfig, cls_token: int, sentiment_types: int, entity_types: int, args):
        super(D2E2SModel, self).__init__(config)
        # 1、parameters init
        self.args = args
        self._size_embedding = self.args.size_embedding
        self._prop_drop = self.args.prop_drop
        self._freeze_transformer = self.args.freeze_transformer
        self.drop_rate = self.args.drop_out_rate
        self._is_bidirectional = self.args.is_bidirect
        self.layers = self.args.lstm_layers
        self._hidden_dim = self.args.hidden_dim
        self.mem_dim = self.args.mem_dim
        self._emb_dim = self.args.emb_dim
        self.output_size = self._emb_dim
        self.batch_size = self.args.batch_size
        self.USE_CUDA = USE_CUDA
        self.max_pairs = 100
        self.bert_feature_dim = self.args.bert_feature_dim
        self.gcn_dim = self.args.gcn_dim
        self.gcn_dropout = self.args.gcn_dropout

        # 2、BERT model
        self.bert = BertModel(config)
        # self.BertAdapterModel = BertAdapterModel(config)
        self.Syn_gcn = GCN()
        self.Sem_gcn = SemGCN(self.args)
        self.senti_classifier = nn.Linear(config.hidden_size * 3 + self._size_embedding * 2, sentiment_types)
        self.entity_classifier = nn.Linear(config.hidden_size * 2 + self._size_embedding, entity_types)
        self.size_embeddings = nn.Embedding(100, self._size_embedding)
        self.dropout = nn.Dropout(self._prop_drop)
        self._cls_token = cls_token
        self._sentiment_types = sentiment_types
        self._entity_types = entity_types
        self._max_pairs = self.max_pairs
        self.neg_span_all = 0
        self.neg_span = 0
        self.number = 1 

        # 3、LSTM Layers + Attention Layers
        self.lstm = nn.LSTM(self._emb_dim, int(self._hidden_dim), self.layers, batch_first=True,
                            bidirectional=self._is_bidirectional, dropout=self.drop_rate)
        self.attention_layer = SelfAttention(self.args)
        self.dropout1 = torch.nn.Dropout(0.5)
        self.dropout2 = torch.nn.Dropout(0)
        self.lstm_dropout = nn.Dropout(self.drop_rate)

        # 4、linear and sigmoid layers
        if self._is_bidirectional:
            self.fc = nn.Linear(int(self._hidden_dim * 2), self.output_size)
        else:
            self.fc = nn.Linear(int(self._hidden_dim), self.output_size)

        # 5、init_hidden
        weight = next(self.parameters()).data
        if self._is_bidirectional:
            self.number = 2

        if self.USE_CUDA:
            self.hidden = (
                weight.new(self.layers * self.number, self.batch_size, self._hidden_dim).zero_().float().cuda(),
                # self.hidden = 384
                weight.new(self.layers * self.number, self.batch_size, self._hidden_dim).zero_().float().cuda()
            )
        else:
            self.hidden = (weight.new(self.layers * self.number, self.batch_size, self._hidden_dim).zero_().float(),
                           weight.new(self.layers * self.number, self.batch_size, self._hidden_dim).zero_().float()
                           )

        # 6、weight initialization
        self.init_weights()
        if self._freeze_transformer:
            print("Freeze transformer weights")

            # freeze all transformer weights
            for param in self.bert.parameters():
                param.requires_grad = False

        # # 7、Mutual Biaffine Model
        # self.affine1 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        # self.affine2 = nn.Parameter(torch.Tensor(self.mem_dim, self.mem_dim))
        # self.gcn_drop = nn.Dropout(self.args.gcn_dropout)
        # # 7、MLP with Biaffine Attention
        # self.Biaffine_ATT = BiaffineAttention(self.bert_feature_dim, self.bert_feature_dim)

        # 7、feature merge model
        self.TIN = TIN(self.bert_feature_dim)
        # self.TextCentredSP = TextCentredSP(self.bert_feature_dim*2, self.shared_dim, self.private_dim)

    def _forward_train(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                       entity_sizes: torch.tensor, sentiments: torch.tensor, senti_masks: torch.tensor, adj):

        # Parameters init
        context_masks = context_masks.float()
        self.context_masks = context_masks
        batch_size = encodings.shape[0]
        seq_lens = encodings.shape[1]

        # encoder layer
        # h = self.BertAdapterModel(input_ids=encodings, attention_mask=self.context_masks)[0]
        h = self.bert(input_ids=encodings, attention_mask=self.context_masks)[0]
        self.output, _ = self.lstm(h, self.hidden)
        self.bert_lstm_output = self.lstm_dropout(self.output)
        self.bert_lstm_att_feature = self.bert_lstm_output

        # attention layers
        # bert_lstm_feature_attention = self.attention_layer(self.bert_lstm_output, self.bert_lstm_output, self.context_masks[:,:seq_lens])
        # self.bert_lstm_att_feature = self.bert_lstm_output + bert_lstm_feature_attention

        # gcn layer
        h_syn_ori, pool_mask_origin = self.Syn_gcn(adj, h)
        h_syn_gcn, pool_mask = self.Syn_gcn(adj, self.bert_lstm_att_feature)
        h_sem_ori, adj_sem_ori = self.Sem_gcn(h, encodings, seq_lens)
        h_sem_gcn, adj_sem_gcn = self.Sem_gcn(self.bert_lstm_att_feature, encodings, seq_lens)

        # fusion layer
        h1 = self.TIN(h, h_syn_ori, h_syn_gcn, h_sem_ori, h_sem_gcn, adj_sem_ori, adj_sem_gcn)
        h = self.attention_layer(h1, h1, self.context_masks[:, :seq_lens]) + h1
        # h_feature, h_syn_origin, h_syn_feature, h_sem_origin, h_sem_feature = self.TIN(h, h_syn_ori, h_syn_gcn, h_sem_ori, h_sem_gcn)
        # h = self.TextCentredSP(h_syn_feature, h_sem_feature)

        size_embeddings = self.size_embeddings(entity_sizes)
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings, self.args)

        # relation_classify
        h_large = h.unsqueeze(1).repeat(1, max(min(sentiments.shape[1], self._max_pairs), 1), 1, 1)
        senti_clf = torch.zeros([batch_size, sentiments.shape[1], self._sentiment_types]).to(self.senti_classifier.weight.device)

        # obtain sentiment logits
        # chunk processing to reduce memory usage
        for i in range(0, sentiments.shape[1], self._max_pairs):
            # classify sentiment candidates
            chunk_senti_logits = self._classify_sentiments(entity_spans_pool, size_embeddings,
                                                        sentiments, senti_masks, h_large, i)
            senti_clf[:, i:i + self._max_pairs, :] = chunk_senti_logits

        batch_loss = compute_loss(adj_sem_ori, adj_sem_gcn, adj)

        return entity_clf, senti_clf, batch_loss

    def _forward_eval(self, encodings: torch.tensor, context_masks: torch.tensor, entity_masks: torch.tensor,
                      entity_sizes: torch.tensor, entity_spans: torch.tensor, entity_sample_masks: torch.tensor, adj):
        context_masks = context_masks.float()
        self.context_masks = context_masks
        batch_size = encodings.shape[0]
        seq_lens = encodings.shape[1]

        # encoder layer
        # h = self.BertAdapterModel(input_ids=encodings, attention_mask=self.context_masks)[0]
        h = self.bert(input_ids=encodings, attention_mask=self.context_masks)[0]
        self.output, _ = self.lstm(h, self.hidden)
        self.bert_lstm_output = self.lstm_dropout(self.output)
        self.bert_lstm_att_feature = self.bert_lstm_output

        # attention layers
        # bert_lstm_feature_attention = self.attention_layer(self.bert_lstm_output, self.bert_lstm_output, self.context_masks[:,:seq_lens])
        # self.bert_lstm_att_feature = self.bert_lstm_output + bert_lstm_feature_attention
        # self.bert_lstm_att_feature = bert_lstm_feature_attention

        # gcn layer
        h_syn_ori, pool_mask_origin = self.Syn_gcn(adj, h)
        h_syn_gcn, pool_mask = self.Syn_gcn(adj, self.bert_lstm_att_feature)
        h_sem_ori, adj_sem_ori = self.Sem_gcn(h, encodings, seq_lens)
        h_sem_gcn, adj_sem_gcn = self.Sem_gcn(self.bert_lstm_att_feature, encodings, seq_lens)

        # fusion layer
        h1 = self.TIN(h, h_syn_ori, h_syn_gcn, h_sem_ori, h_sem_gcn, adj_sem_ori, adj_sem_gcn)
        h = self.attention_layer(h1, h1, self.context_masks[:, :seq_lens]) + h1
        # h_feature, h_syn_origin, h_syn_feature, h_sem_origin, h_sem_feature = self.TIN(h, h_syn_ori, h_syn_gcn, h_sem_ori, h_sem_gcn)
        # h = self.TextCentredSP(h_syn_feature, h_sem_feature)

        # entity_classify
        size_embeddings = self.size_embeddings(entity_sizes)  # embed entity candidate sizes
        entity_clf, entity_spans_pool = self._classify_entities(encodings, h, entity_masks, size_embeddings, self.args)

        # ignore entity candidates that do not constitute an actual entity for sentiments (based on classifier)
        ctx_size = context_masks.shape[-1]
        sentiments, senti_masks, senti_sample_masks = self._filter_spans(entity_clf, entity_spans,
                                                                    entity_sample_masks, ctx_size)
        senti_sample_masks = senti_sample_masks.float().unsqueeze(-1)
        h_large = h.unsqueeze(1).repeat(1, max(min(sentiments.shape[1], self._max_pairs), 1), 1, 1)
        senti_clf = torch.zeros([batch_size, sentiments.shape[1], self._sentiment_types]).to(
            self.senti_classifier.weight.device)

        # obtain sentiment logits
        # chunk processing to reduce memory usage
        for i in range(0, sentiments.shape[1], self._max_pairs):
            # classify sentiment candidates
            chunk_senti_logits = self._classify_sentiments(entity_spans_pool, size_embeddings,
                                                        sentiments, senti_masks, h_large, i)
            # apply sigmoid
            chunk_senti_clf = torch.sigmoid(chunk_senti_logits)
            senti_clf[:, i:i + self._max_pairs, :] = chunk_senti_clf

        senti_clf = senti_clf * senti_sample_masks  # mask

        # apply softmax
        entity_clf = torch.softmax(entity_clf, dim=2)

        return entity_clf, senti_clf, sentiments

    def _classify_entities(self, encodings, h, entity_masks, size_embeddings, args):
        # entity_masks: tensor(4,132,24) 4:batch_size, 132: entities count, 24: one sentence token count and one entity need 24 mask
        # size_embedding: tensor(4,132,25) 4：batch_size, 132:entities_size, 25:each entities Embedding Dimension
        # h: tensor(4,24,768) -> (4,1,24,768) -> (4,132,24,768)
        # m: tensor(4,132,24,1)
        # encoding: tensor(4,24)
        # entity_spans_pool: tensor(4，132，24，768) -> tensor(4,132,768)
        m = (entity_masks.unsqueeze(-1) == 0).float() * (-1e30)
        entity_spans_pool = m + h.unsqueeze(1).repeat(1, entity_masks.shape[1], 1, 1)

        self.args = args
        if self.args.span_generator == "Average" or self.args.span_generator == "Max":
            if self.args.span_generator == "Max":
                entity_spans_pool = entity_spans_pool.max(dim=2)[0]
            else:
                entity_spans_pool = entity_spans_pool.mean(dim=2, keepdim=True).squeeze(-2)

        # get cls token as candidate context representation
        entity_ctx = get_token(h, encodings, self._cls_token)

        # create candidate representations including context, max pooled span and size embedding
        entity_repr = torch.cat([entity_ctx.unsqueeze(1).repeat(1, entity_spans_pool.shape[1], 1),
                                 entity_spans_pool, size_embeddings], dim=2)
        entity_repr = self.dropout(entity_repr)

        # classify entity candidates
        entity_clf = self.entity_classifier(entity_repr)

        return entity_clf, entity_spans_pool

    def _classify_sentiments(self, entity_spans, size_embeddings, sentiments, senti_masks, h, chunk_start):
        batch_size = sentiments.shape[0]

        # create chunks if necessary
        if sentiments.shape[1] > self._max_pairs:
            sentiments = sentiments[:, chunk_start:chunk_start + self._max_pairs]
            senti_masks = senti_masks[:, chunk_start:chunk_start + self._max_pairs]
            h = h[:, :sentiments.shape[1], :]

        # get pairs of entity candidate representations
        entity_pairs = util.batch_index(entity_spans, sentiments)
        entity_pairs = entity_pairs.view(batch_size, entity_pairs.shape[1], -1)

        # get corresponding size embeddings
        size_pair_embeddings = util.batch_index(size_embeddings, sentiments)
        size_pair_embeddings = size_pair_embeddings.view(batch_size, size_pair_embeddings.shape[1], -1)

        # sentiment context (context between entity candidate pair)
        # mask non entity candidate tokens
        m = ((senti_masks == 0).float() * (-1e30)).unsqueeze(-1)
        senti_ctx = m + h
        # max pooling
        senti_ctx = senti_ctx.max(dim=2)[0]
        # set the context vector of neighboring or adjacent entity candidates to zero
        senti_ctx[senti_masks.to(torch.uint8).any(-1) == 0] = 0

        # create sentiment candidate representations including context, max pooled entity candidate pairs
        # and corresponding size embeddings
        senti_repr = torch.cat([senti_ctx, entity_pairs, size_pair_embeddings], dim=2)
        senti_repr = self.dropout(senti_repr)

        # classify sentiment candidates
        chunk_senti_logits = self.senti_classifier(senti_repr)
        return chunk_senti_logits

    def log_sample_total(self,neg_entity_count_all):
        log_path = os.path.join('./log/Sample/', 'countSample.txt')
        with open(log_path, mode='a', encoding='utf-8') as f:
            f.write('neg_entity_count_all: \n')
            self.neg_span_all += len(neg_entity_count_all)
            f.write(str(self.neg_span_all))
            f.write('\nneg_entity_count: \n')
            self.neg_span += len((neg_entity_count_all !=0).nonzero())
            f.write(str(self.neg_span))
            f.write('\n')
        f.close()

    def _filter_spans(self, entity_clf, entity_spans, entity_sample_masks, ctx_size):
        batch_size = entity_clf.shape[0]
        entity_logits_max = entity_clf.argmax(dim=-1) * entity_sample_masks.long()  # get entity type (including none)
        batch_sentiments = []
        batch_senti_masks = []
        batch_senti_sample_masks = []

        for i in range(batch_size):
            rels = []
            senti_masks = []
            sample_masks = []

            # get spans classified as entities
            self.log_sample_total(entity_logits_max[i])
            non_zero_indices = (entity_logits_max[i] != 0).nonzero().view(-1)
            non_zero_spans = entity_spans[i][non_zero_indices].tolist()
            non_zero_indices = non_zero_indices.tolist()


            # create sentiments and masks
            for i1, s1 in zip(non_zero_indices, non_zero_spans):
                for i2, s2 in zip(non_zero_indices, non_zero_spans):
                    if i1 != i2:
                        rels.append((i1, i2))
                        senti_masks.append(sampling.create_senti_mask(s1, s2, ctx_size))
                        sample_masks.append(1)

            if not rels:
                # case: no more than two spans classified as entities
                batch_sentiments.append(torch.tensor([[0, 0]], dtype=torch.long))
                batch_senti_masks.append(torch.tensor([[0] * ctx_size], dtype=torch.bool))
                batch_senti_sample_masks.append(torch.tensor([0], dtype=torch.bool))
            else:
                # case: more than two spans classified as entities
                batch_sentiments.append(torch.tensor(rels, dtype=torch.long))
                batch_senti_masks.append(torch.stack(senti_masks))
                batch_senti_sample_masks.append(torch.tensor(sample_masks, dtype=torch.bool))

        # stack
        device = self.senti_classifier.weight.device
        batch_sentiments = util.padded_stack(batch_sentiments).to(device)
        batch_senti_masks = util.padded_stack(batch_senti_masks).to(device)
        batch_senti_sample_masks = util.padded_stack(batch_senti_sample_masks).to(device)

        return batch_sentiments, batch_senti_masks, batch_senti_sample_masks

    def forward(self, *args, evaluate=False, **kwargs):
        if not evaluate:
            return self._forward_train(*args, **kwargs)
        else:
            return self._forward_eval(*args, **kwargs)

def compute_loss(p, q, k):

    p_loss = F.kl_div(F.log_softmax(p, dim=-1), F.softmax(k, dim=-1), reduction="none")
    k_loss = F.kl_div(F.log_softmax(k, dim=-1), F.softmax(p, dim=-1), reduction="none")

    p_loss = p_loss.sum()
    k_loss = k_loss.sum()
    total_loss = math.log(1 + 5 / (torch.abs((p_loss + k_loss) / 2)))

    return total_loss
