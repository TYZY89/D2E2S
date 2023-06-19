import copy
import math

import torch
import torch.nn.functional as F


def attention(query, key, value, mask=None, dropout=None):
    "Compute 'Scaled Dot Product Attention'"
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) \
             / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim = -1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def clones(module, N):
    return torch.nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


class MultiHeadedAttention(torch.nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(torch.nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x)


class SelfAttention(torch.nn.Module):
    def __init__(self, args):
        super(SelfAttention,self).__init__()
        self.args = args
        self.linear_q = torch.nn.Linear(args.lstm_dim * 2, args.lstm_dim * 2)
        self.w_query = torch.nn.Linear(args.lstm_dim * 2, args.lstm_dim)
        self.w_value = torch.nn.Linear(args.lstm_dim * 2, args.lstm_dim)
        self.v = torch.nn.Linear(args.lstm_dim, 1, bias=False)

    def forward(self, query, value, mask):
        attention_states = query
        attention_states_T = value
        attention_states_T = attention_states_T.permute([0, 2, 1])

        weights=torch.bmm(attention_states, attention_states_T)
        weights = weights.masked_fill(mask.unsqueeze(1).expand_as(weights)==0, float("-inf"))
        attention = F.softmax(weights,dim=2)

        merged=torch.bmm(attention, value)
        merged=merged * mask.unsqueeze(2).float().expand_as(merged)

        return merged

    def forward_perceptron(self, query, value, mask):
        attention_states = query
        attention_states = self.w_query(attention_states)
        attention_states = attention_states.unsqueeze(2).expand(-1,-1,attention_states.shape[1], -1)

        attention_states_T = value
        attention_states_T = self.w_value(attention_states_T)
        attention_states_T = attention_states_T.unsqueeze(2).expand(-1,-1,attention_states_T.shape[1], -1)
        attention_states_T = attention_states_T.permute([0, 2, 1, 3])

        weights = torch.tanh(attention_states+attention_states_T)
        weights = self.v(weights).squeeze(3)
        weights = weights.masked_fill(mask.unsqueeze(1).expand_as(weights)==0, float("-inf"))
        attention = F.softmax(weights,dim=2)

        merged = torch.bmm(attention, value)
        merged = merged * mask.unsqueeze(2).float().expand_as(merged)
        return merged
