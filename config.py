import argparse


def model_config(parser):
    ### extractor
    # word embedding
    parser.add_argument('--word_dim', type=int, default=300)

    # convolutional sentence encoder
    parser.add_argument('--filter_sizes', nargs='+', type=int, default=[3, 4, 5])
    parser.add_argument('--num_feature_maps', type=int, default=100)

    ### abstracter

    return parser


def data_config(parser):
    parser.add_argument('--train_path', default='data/cnn-dailymail/train')
    parser.add_argument('--valid_path', default='data/cnn-dailymail/valid')
    parser.add_argument('--vocab_dir', default='data/cnn-dailymail/vocab')
    parser.add_argument('--lazy_ratio', type=float, default=0.1)
    return parser


def train_config(parser):
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--mode', default='e',
                        help="available modes: e (extractor), a (abstracter), r (reinforcement)")
    return parser


def set_args():
    parser = argparse.ArgumentParser()
    parser = data_config(parser)
    parser = model_config(parser)
    parser = train_config(parser)
    args = parser.parse_args()
    return vars(args)
