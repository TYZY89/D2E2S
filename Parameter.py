import argparse
import torch

def train_argparser():

    dataset_files = {
        '14lap': {
            'train': './data/14lap/train_dep_triple_polarity_result.json',
            'test': './data/14lap/test_dep_triple_polarity_result.json',
            'types_path': './data/types.json'
        },
        '14res': {
            'train': './data/14res/train_dep_triple_polarity_result.json',
            'test': './data/14res/test_dep_triple_polarity_result.json',
            'types_path': './data/types.json'
        },
        '15res': {
            'train': './data/15res/train_dep_triple_polarity_result.json',
            'test': './data/15res/test_dep_triple_polarity_result.json',
            'types_path': './data/types.json'
        },
        '16res': {
            'train': './data/16res/train_dep_triple_polarity_result.json',
            'test': './data/16res/test_dep_triple_polarity_result.json',
            'types_path': './data/types.json'
        }
    }

    # model argument
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='14res', type=str, help='14res, 15res, 16res, 14lap')

    parser.add_argument('--drop_out_rate', type=float, default=0.5, help='drop out rate.')
    parser.add_argument('--is_bidirect', default=True, help='Do use bi-RNN layer.')
    parser.add_argument('--use_gated', default=False, help='Do use gcnconv and gatedgraphconv.')
    parser.add_argument('--hidden_dim', type=int, default=384, help='hidden layer dimension.')
    parser.add_argument('--emb_dim', type=int, default=768, help='Word embedding dimension.')
    parser.add_argument('--lstm_layers', type=int, default=2, help='Number of LSTM layers.')
    parser.add_argument('--lstm_dim', type=int, default=384, help='dimension of lstm cell')
    parser.add_argument('--prefix', type=str, default="data/", help='dataset and embedding path prefix')
    parser.add_argument('--span_generator', type=str, default="Max", choices=["Max", "Average"], help='option: Max, Average')
    parser.add_argument('--attention_heads', default=1, type=int, help='number of multi-attention heads')
    parser.add_argument('--num_layers', type=int, default=2, help='Num of GCN layers.')
    parser.add_argument('--mem_dim', type=int, default=768, help='mutual biaffine men dim.')
    parser.add_argument('--gcn_dropout', type=float, default=0.2, help='GCN layer dropout rate.')
    parser.add_argument('--pooling', default='avg', type=str, help='[max, avg, sum]')
    parser.add_argument('--gcn_dim', type=int, default=300, help='dimension of gcn')
    parser.add_argument('--bert_feature_dim', type=int, default=768, help='dimension of pretrained bert feature')
    parser.add_argument("--seed", default=42, type=int, help="Random seed for initialization")

    parser.add_argument('--max_span_size', type=int, default=8, help="Maximum size of spans")
    parser.add_argument('--lowercase', action='store_true', default=True, help="If False, training not case sentive")
    parser.add_argument('--max_pairs', type=int, default=1000,help="During training and evaluation, the maximum number of entity pairs will be processed")
    parser.add_argument('--sen_filter_threshold', type=float, default=0.4, help="Filter threshold for sentiment triplet")
    parser.add_argument('--sampling_limit', type=int, default=100, help="There is a maximum number of samples in queue")
    parser.add_argument('--neg_entity_count', type=int, default=100,help="The number of negative entities samples for each sample")
    parser.add_argument('--neg_triple_count', type=int, default=100, help="The number of negative triplets samples for each sample")

    parser.add_argument('--tokenizer_path', default='bert-base-uncased', type=str, help="tokenizer load_path")
    parser.add_argument('--cpu', action='store_true', default=False,help="If true, train/evaluate on CPU even if a CUDA device is available")
    parser.add_argument('--size_embedding', type=int, default=25, help="Dimensionality of size embedding,E_{k}")
    parser.add_argument('--sampling_processes', type=int, default=4,help="The number of sampling processes")
    parser.add_argument('--prop_drop', type=float, default=0.1, help="Probability of dropout used in D2E2S")
    parser.add_argument('--freeze_transformer', action='store_true', default=False, help="Freezing Bert's parameters for easy test")
    parser.add_argument('--batch_size', type=int, default=16, help="Training batch size")
    parser.add_argument('--epochs', type=int, default=120, help="Number of epochs")
    parser.add_argument('--lr', type=float, default=5e-5, help="Learning rate")
    parser.add_argument('--lr_warmup', type=float, default=0.1, help="Proportion of total train iterations to warmup in linear increase/decrease schedule")
    parser.add_argument('--log_path', type=str,default="log/", help="Path do directory where training/evaluation logs are stored")
    parser.add_argument('--train_log_iter', type=int, default=1, help="Log training process every x iterations")
    parser.add_argument('--pretrained_bert_name', default='bert-base-uncased', type=str)
    parser.add_argument('--weight_decay', type=float, default=0.01, help="Weight decay to apply")
    parser.add_argument('--max_grad_norm', type=float, default=1.0, help="Maximum gradient norm")
    parser.add_argument('--init_eval', action='store_true', default=False, help="If true, evaluate validation set before training")
    parser.add_argument('--final_eval', action='store_true', default=False,help="Evaluate the model only after training, not at every epoch")
    parser.add_argument('--store_predictions', action='store_true', default=True, help="If true, store predictions on disc (in log directory)")
    parser.add_argument('--store_examples', action='store_true', default=True, help="If true, store evaluation examples on disc (in log directory)")
    parser.add_argument('--example_count', type=int, default=None, help="Count of evaluation example to store (if store_examples == True)")
    parser.add_argument('--save_path', type=str, default="data/save/", help="Path to directory where model checkpoints are stored")
    parser.add_argument('--save_optimizer', action='store_true', default=False,help="Save optimizer alongside model")
    parser.add_argument('--device', type=str, default="cuda", help='gpu or cpu')


    opt = parser.parse_args()
    opt.label = opt.dataset

    opt.dataset_file = dataset_files[opt.dataset]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)
    return opt
