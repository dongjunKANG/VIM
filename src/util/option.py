import argparse

def parse_train_argnet():
    parser = argparse.ArgumentParser()

    #general settings
    parser.add_argument(
        '--gpu',
        type=int,
        help='Number of gpu.')
    parser.add_argument(
        '--num_epochs',
        type=int,
        help='Number of epochs.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='The initial learning rate for training.')
    parser.add_argument(
        '--logging_step',
        type=int,
        help='The logging step for training.')
    parser.add_argument(
        '--warmup_ratio',
        type=float,
        help='Warmup ratio for training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size for training.')
    parser.add_argument(
        '--model',
        choices=['gpt', 'gpt-j', 'gpt-neo', 'OPT'],
        help='Model type (currently supports `gpt` / `gpt-j` / `gpt-neo` / `OPT`).')
    args = parser.parse_args()
    return args


def parse_inference():
    parser = argparse.ArgumentParser()

    #general settings
    parser.add_argument(
        '--seed',
        type=int,
        help='The seed for reproducibility (optional).')
    parser.add_argument(
        '--gpu',
        type=int,
        help='Number of gpu.')
    parser.add_argument(
        '--learning_rate',
        type=float,
        help='The initial learning rate for training.')
    parser.add_argument(
        '--batch_size',
        type=int,
        help='Batch size for training.')
    parser.add_argument(
        '--model',
        choices=['gpt', 'gpt-j', 'gpt-neo', 'OPT'],
        help='Model type (currently supports `gpt` / `gpt-j` / `gpt-neo` / `OPT`).')
    parser.add_argument(
        '--mode',
        choices=['default', 'user', 'negative'],
        help='Mode type (currently supports `default` / `user`/ `negative`).')
    args = parser.parse_args()
    return args