import torch.nn as nn
import torch.nn.functional as F
import torch
import copy
import math
class SemGCN(nn.Module):

    def __init__(self, args, emb_dim=768, num_layers=2, gcn_dropout=0.1):
        super(SemGCN, self).__init__()
        self.args = args
        self.layers = num_layers
        self.emb_dim = emb_dim
        self.out_dim = emb_dim
        self.attention_heads = self.args.attention_heads
        self.mem_dim = self.args.hidden_dim
        # gcn layer
        self.W = nn.ModuleList()
        for layer in range(self.layers):
            input_dim = self.emb_dim if layer == 0 else self.out_dim
            self.W.append(nn.Linear(input_dim, input_dim))
        self.gcn_drop = nn.Dropout(gcn_dropout)
        self.attn = MultiHeadAttention(self.attention_heads, self.mem_dim * 2)

    def forward(self, inputs, encoding, seq_lens):
        tok = encoding
        src_mask = (tok != 0).unsqueeze(-2)
        maxlen = seq_lens
        mask_ = (torch.zeros_like(tok) != tok).float().unsqueeze(-1)[:, :maxlen]

        gcn_inputs = inputs
        attn_tensor = self.attn(gcn_inputs, gcn_inputs, src_mask)
        attn_adj_list = [attn_adj.squeeze(1) for attn_adj in torch.split(attn_tensor, 1, dim=1)]
        adj_ag = None

        for i in range(self.attention_heads):
            if adj_ag is None:
                adj_ag = attn_adj_list[i]
            else:
                adj_ag += attn_adj_list[i]
        adj_ag_new = adj_ag.clone()
        adj_ag_new /= self.attention_heads

        for j in range(adj_ag_new.size(0)):
            adj_ag_new[j] -= torch.diag(torch.diag(adj_ag_new[j]))
            adj_ag_new[j] += torch.eye(adj_ag_new[j].size(0)).cuda()
        adj_ag_new = mask_ * adj_ag_new

        # gcn layer
        denom_ag = adj_ag_new.sum(2).unsqueeze(2) + 1
        outputs = gcn_inputs

        for l in range(self.layers):
            Ax = adj_ag_new.bmm(outputs)
            AxW = self.W[l](Ax)
            AxW = AxW / denom_ag
            gAxW = F.relu(AxW)
            outputs = self.gcn_drop(gAxW) if l < self.layers - 1 else gAxW

        return outputs, adj_ag_new

def attention(query, key, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)

    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)

    return p_attn

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class MultiHeadAttention(nn.Module):

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0

        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 2)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, mask=None):
        mask = mask[:, :, :query.size(1)]
        if mask is not None:
            mask = mask.unsqueeze(1)

        nbatches = query.size(0)
        query, key = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
                      for l, x in zip(self.linears, (query, key))]

        attn = attention(query, key, mask=mask, dropout=self.dropout)

        return attn