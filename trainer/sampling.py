import random
import torch
from trainer import util

def pos_entity_sample(sen,context_size, entity_types, entity_masks, entity_start_masks,entity_end_masks, entity_sizes):

    pos_entity_spans, pos_entity_types, pos_entity_masks, pos_entity_sizes, pos_entity_start_masks, pos_entity_end_masks = [], [], [], [],[],[]
    for e in sen.entities:
        pos_entity_spans.append(e.span)
        pos_entity_types.append(e.entity_type.index)
        pos_entity_masks.append(create_entity_mask(*e.span, context_size))
        pos_entity_start_masks.append(create_entity_s_e_mask(*e.span, context_size,1))
        pos_entity_end_masks.append(create_entity_s_e_mask(*e.span, context_size,0))

        pos_entity_sizes.append(len(e.tokens))
    """
    "CLS", "The", "service", "was", "friendly", "and", "the", "atmosphere", "was", "casual", ".", "SEP"
      0      1        2        3         4        5      6          7         8        9      10    11
    head_entity = {Entity} service
        sentiment_type = {sentimentType} <trainer.entities.sentimentType object at 0x0000018A545D66B0>
        identifier = {str} 'POSITIVE'
        index = {int} 2
        short_name = {str} 'POS'
        symmetric = {bool} False                                                                           
        verbose_name = {str} 'POSITIVE'
    tail_entity = {Entity} friendly
    
    e = {Entity} friendly
         entity_type = {EntityType} <trainer.entities.EntityType object at 0x0000018A545D68F0>
         phrase = {str} 'friendly'
         span = {tuple: 2} (4, 5)
         span_end = {int} 5
         span_start = {int} 4
         tokens = {TokenSpan: 1} <trainer.entities.TokenSpan object at 0x0000018A54FCCBB0>
              span = {tuple: 2} (4, 5)
              span_end = {int} 5
              span_start = {int} 4
    pos_entity_end_masks = {list: 2} [tensor([False, False,  True, False, False, False, False, False, False, False, False, False]), 
                                      tensor([False, False, False, False,  True, False, False, False, False, False, False, False])]
    pos_entity_masks = {list: 2} [tensor([False, False,  True, False, False, False, False, False, False, False, False, False]), 
                                  tensor([False, False, False, False,  True, False, False, False, False, False, False, False])]
    pos_entity_sizes = {list: 1} [1]
    pos_entity_spans = {list: 2} [(2, 3), (4, 5)]
    pos_entity_start_masks = {list: 2} [tensor([False, False,  True, False, False, False, False, False, False, False, False, False]), 
                                        tensor([False, False, False, False,  True, False, False, False, False, False, False, False])]
    pos_entity_types = {list: 2} [1, 2]
    """

    entity_types = entity_types + pos_entity_types
    entity_masks = entity_masks + pos_entity_masks
    entity_start_masks =  entity_start_masks + pos_entity_start_masks
    entity_end_masks = entity_end_masks + pos_entity_end_masks
    entity_sizes = entity_sizes + pos_entity_sizes

    return pos_entity_spans, entity_types, entity_masks, entity_start_masks,entity_end_masks,entity_sizes

def neg_entity_sample(sen, pos_entity_spans,neg_entity_count, max_span_size, token_count,context_size,entity_types, entity_masks, entity_start_masks,entity_end_masks, entity_sizes):
    neg_entity_spans, neg_entity_sizes = [], []
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = sen.tokens[i:i + size].span
            if span not in pos_entity_spans:
                neg_entity_spans.append(span)
                neg_entity_sizes.append(size)

    if len(neg_entity_spans) < neg_entity_count:
        neg_entity_count = len(neg_entity_spans) * 10
    else:
        neg_entity_count = len(neg_entity_spans)

    neg_entity_samples = random.sample(list(zip(neg_entity_spans, neg_entity_sizes)),
                                       min(len(neg_entity_spans), int(neg_entity_count)))
    neg_entity_spans, neg_entity_sizes = zip(*neg_entity_samples) if neg_entity_samples else ([], [])

    neg_entity_masks = [create_entity_mask(*span, context_size) for span in neg_entity_spans]
    neg_entity_start_masks = [create_entity_s_e_mask(*span, context_size, 1) for span in neg_entity_spans]
    neg_entity_end_masks = [create_entity_s_e_mask(*span, context_size, 0) for span in neg_entity_spans]
    neg_entity_types = [0] * len(neg_entity_spans)

    entity_types = entity_types + neg_entity_types
    entity_masks = entity_masks + neg_entity_masks
    entity_start_masks =  entity_start_masks + neg_entity_start_masks
    entity_end_masks = entity_end_masks + neg_entity_end_masks
    entity_sizes = entity_sizes + list(neg_entity_sizes)

    return neg_entity_spans,entity_types, entity_masks, entity_start_masks,entity_end_masks,entity_sizes

def pos_senti_sample(sen, pos_entity_spans, context_size):
    pos_rels, pos_senti_spans, pos_senti_types, pos_senti_masks = [], [], [], []
    for rel in sen.sentiments:
        s1, s2 = rel.head_entity.span, rel.tail_entity.span
        pos_rels.append((pos_entity_spans.index(s1), pos_entity_spans.index(s2)))
        pos_senti_spans.append((s1, s2))
        pos_senti_types.append(rel.sentiment_type)
        pos_senti_masks.append(create_senti_mask(s1, s2, context_size))
    return pos_senti_spans, pos_senti_types,pos_rels, pos_senti_masks

def neg_senti_sample(pos_entity_spans, pos_senti_spans, pos_senti_types):
    neg_senti_spans = []
    for i1, s1 in enumerate(pos_entity_spans):
        for i2, s2 in enumerate(pos_entity_spans):
            rev = (s2, s1)
            rev_symmetric = rev in pos_senti_spans and pos_senti_types[pos_senti_spans.index(rev)].symmetric

            if s1 != s2 and (s1, s2) not in pos_senti_spans and not rev_symmetric:
                neg_senti_spans.append((s1, s2))
    return neg_senti_spans

def create_entity_sample_mask(entity_masks, entity_types, entity_start_masks, entity_end_masks, entity_sizes, context_size):
    if entity_masks:
        entity_types = torch.tensor(entity_types, dtype=torch.long)
        entity_masks = torch.stack(entity_masks)
        entity_start_masks = torch.stack(entity_start_masks)
        entity_end_masks = torch.stack(entity_end_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_sample_masks = torch.ones([entity_masks.shape[0]], dtype=torch.bool)
    else:
        entity_types = torch.zeros([1], dtype=torch.long)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)
    return entity_sample_masks,entity_types,entity_masks,entity_start_masks,entity_end_masks,entity_sizes

def train_create_sample(sen,neg_entity_count: int, neg_senti_count: int, max_span_size: int, senti_type_count: int):
    encodings = sen.encoding
    adj = sen.adj
    token_count = len(sen.tokens)
    context_size = len(encodings)

    # create entity sample
    entity_types, entity_masks, entity_start_masks, entity_end_masks, entity_sizes =[],[],[],[],[]

    # create positive sample in training process.
    pos_entity_spans, entity_types, entity_masks, entity_start_masks,entity_end_masks,entity_sizes = pos_entity_sample(sen,context_size, entity_types, entity_masks, entity_start_masks,entity_end_masks,entity_sizes)
    # create negative sample in training process.
    neg_entity_spans,entity_types, entity_masks, entity_start_masks,entity_end_masks,entity_sizes = neg_entity_sample(sen, pos_entity_spans,
                                                                                           neg_entity_count, max_span_size, token_count,context_size,
                                                                                           entity_types, entity_masks, entity_start_masks,entity_end_masks, entity_sizes)
    # create entity mask
    entity_sample_masks,entity_types,entity_masks,entity_start_masks,entity_end_masks,entity_sizes = create_entity_sample_mask(entity_masks, entity_types, entity_start_masks, entity_end_masks, entity_sizes, context_size)

    pos_senti_spans, pos_senti_types,pos_rels, pos_senti_masks= pos_senti_sample(sen, pos_entity_spans, context_size)

    neg_senti_spans = neg_senti_sample(pos_entity_spans, pos_senti_spans, pos_senti_types)
    neg_senti_count = len(neg_entity_spans)
    neg_senti_spans = random.sample(neg_senti_spans, min(len(neg_senti_spans), neg_senti_count))
    neg_rels = [(pos_entity_spans.index(s1), pos_entity_spans.index(s2)) for s1, s2 in neg_senti_spans]
    neg_senti_masks = [create_senti_mask(*spans, context_size) for spans in neg_senti_spans]
    neg_senti_types = [0] * len(neg_senti_spans)




    rels = pos_rels + neg_rels
    senti_types = [r.index for r in pos_senti_types] + neg_senti_types
    senti_masks = pos_senti_masks + neg_senti_masks
    assert len(entity_masks) == len(entity_sizes) == len(entity_types) == len(entity_start_masks) == len(entity_end_masks)
    assert len(rels) == len(senti_masks) == len(senti_types)

    encodings = torch.tensor(encodings, dtype=torch.long)
    context_masks = torch.ones(context_size, dtype=torch.bool)
    adj = torch.tensor(adj, dtype=torch.float)


    if rels:
        rels = torch.tensor(rels, dtype=torch.long)
        senti_masks = torch.stack(senti_masks)
        senti_types = torch.tensor(senti_types, dtype=torch.long)
        senti_sample_masks = torch.ones([rels.shape[0]], dtype=torch.bool)
    else:
        rels = torch.zeros([1, 2], dtype=torch.long)
        senti_types = torch.zeros([1], dtype=torch.long)
        senti_masks = torch.zeros([1, context_size], dtype=torch.bool)
        senti_sample_masks = torch.zeros([1], dtype=torch.bool)

    senti_types_onehot = torch.zeros([senti_types.shape[0], senti_type_count], dtype=torch.float32)
    senti_types_onehot.scatter_(1, senti_types.unsqueeze(1), 1)
    senti_types_onehot = senti_types_onehot[:, 1:]  # all zeros for 'none' sentiment

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_start_masks = entity_start_masks,entity_end_masks = entity_end_masks,
                entity_sizes=entity_sizes, entity_types=entity_types,
                rels=rels, senti_masks=senti_masks, senti_types=senti_types_onehot,
                entity_sample_masks=entity_sample_masks, senti_sample_masks=senti_sample_masks,
                adj = adj)

def create_test_sample(sen, max_span_size: int):
    encodings = sen.encoding
    adj = sen.adj
    token_count = len(sen.tokens)
    context_size = len(encodings)
    # create test entity
    entity_start_masks, entity_end_masks = [],[]
    entity_spans = []
    entity_masks = []
    entity_sizes = []
    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = sen.tokens[i:i + size].span
            entity_spans.append(span)
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_start_masks.append(create_entity_s_e_mask(*span, context_size, 1))
            entity_end_masks.append(create_entity_s_e_mask(*span, context_size, 0))
            entity_sizes.append(size)

    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    adj = torch.tensor(adj, dtype=torch.float)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.zeros(context_size, dtype=torch.bool)
    context_masks[:len(_encoding)] = 1

    # entities
    if entity_masks:
        entity_start_masks = torch.stack(entity_start_masks)
        entity_end_masks = torch.stack(entity_end_masks)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)

        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no entities)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_start_masks=entity_start_masks, entity_end_masks=entity_end_masks,
                entity_sizes=entity_sizes, entity_spans=entity_spans, entity_sample_masks=entity_sample_masks,adj = adj)

def create_entity_index(start, end):
    index = []
    for i in range(start,end):
        index.append(i)
    return index
def create_entity_s_e_mask(start, end, context_size,s_e):
    mask = torch.zeros(context_size, dtype=torch.bool)
    if s_e:
        mask[start] = 1
    else:
        mask[end-1] = 1
    return mask

def create_entity_mask(start, end, context_size):
    mask = torch.zeros(context_size, dtype=torch.bool)
    mask[start:end] = 1
    return mask

def create_senti_mask(s1, s2, context_size):
    start = s1[1] if s1[1] < s2[0] else s2[1]
    end = s2[0] if s1[1] < s2[0] else s1[0]
    mask = create_entity_mask(start, end, context_size)
    return mask

def collate_fn_padding(batch):
    padded_batch = dict()
    keys = batch[0].keys()

    for key in keys:
        samples = [s[key] for s in batch]

        if not batch[0][key].shape:
            padded_batch[key] = torch.stack(samples)
        else:
            padded_batch[key] = util.padded_stack([s[key] for s in batch])

    return padded_batch

def create_eval_sample(sen, max_span_size: int):
    encodings = sen.encoding
    adj = sen.adj
    token_count = len(sen.tokens)
    context_size = len(encodings)

    # create entity candidates
    entity_start_masks, entity_end_masks = [], []
    entity_spans = []
    entity_masks = []
    entity_sizes = []

    for size in range(1, max_span_size + 1):
        for i in range(0, (token_count - size) + 1):
            span = sen.tokens[i:i + size].span
            entity_spans.append(span)
            entity_masks.append(create_entity_mask(*span, context_size))
            entity_start_masks.append(create_entity_s_e_mask(*span, context_size, 1))
            entity_end_masks.append(create_entity_s_e_mask(*span, context_size, 0))
            entity_sizes.append(size)

    # create tensors
    # token indices
    _encoding = encodings
    encodings = torch.zeros(context_size, dtype=torch.long)
    adj = torch.tensor(adj, dtype=torch.float)
    encodings[:len(_encoding)] = torch.tensor(_encoding, dtype=torch.long)

    # masking of tokens
    context_masks = torch.zeros(context_size, dtype=torch.bool)
    context_masks[:len(_encoding)] = 1

    # entities
    if entity_masks:
        entity_start_masks = torch.stack(entity_start_masks)
        entity_end_masks = torch.stack(entity_end_masks)
        entity_masks = torch.stack(entity_masks)
        entity_sizes = torch.tensor(entity_sizes, dtype=torch.long)
        entity_spans = torch.tensor(entity_spans, dtype=torch.long)

        # tensors to mask entity samples of batch
        # since samples are stacked into batches, "padding" entities possibly must be created
        # these are later masked during evaluation
        entity_sample_masks = torch.tensor([1] * entity_masks.shape[0], dtype=torch.bool)
    else:
        # corner case handling (no entities)
        entity_masks = torch.zeros([1, context_size], dtype=torch.bool)
        entity_sizes = torch.zeros([1], dtype=torch.long)
        entity_spans = torch.zeros([1, 2], dtype=torch.long)
        entity_sample_masks = torch.zeros([1], dtype=torch.bool)

    return dict(encodings=encodings, context_masks=context_masks, entity_masks=entity_masks,
                entity_start_masks=entity_start_masks, entity_end_masks=entity_end_masks,
                entity_sizes=entity_sizes, entity_spans=entity_spans, entity_sample_masks=entity_sample_masks,adj = adj)

