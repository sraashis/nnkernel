import torch as _torch

cuda_available = _torch.cuda.is_available()
num_gpus = _torch.cuda.device_count()

metrics_eps = 10e-5
metrics_num_precision = 5

import argparse as _ap

import random as _random

from collections import OrderedDict as _ODict


def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'


class StoreDictKeyPairSS(_ap.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        my_dict = _ODict()
        for kv in values.split(","):
            k, v = kv.split("=")
            my_dict[k] = v
        setattr(namespace, self.dest, my_dict)


default_args = _ap.ArgumentParser()

default_args.add_argument('-ph', '--phase', default=None, choices=['train', 'test'], type=str,
                          help='Can be train/test.')
default_args.add_argument("-b", "--batch_size", default=32, type=int, help="Mini-batch size.")
default_args.add_argument('-ep', '--epochs', default=51, type=int, help='Number of epochs.')
default_args.add_argument('-lr', '--learning_rate', default=0.001, type=float, help='Learning rate.')
default_args.add_argument('-gpus', '--gpus', default=[0], nargs='*', type=int, help='How many gpus to use?')
default_args.add_argument('-pin', '--pin_memory', default=True, type=boolean_string, help='Pin Memory.')
default_args.add_argument('-nw', '--num_workers', default=4, type=int,
                          help='Number of workers to work on data loading.')
default_args.add_argument('-data', '--dataset_dir', default='', type=str, help='Root path to Datasets.')
default_args.add_argument('-lim', '--load_limit', default=1e11, type=int, help='Data load limit')
default_args.add_argument('-log', '--log_dir', default='net_logs', type=str, help='Logging directory.')
default_args.add_argument('-pt', '--pretrained_path', default=None, type=str,
                          help='Full path to pretrained weights(It will be '
                               'loaded before training.)')
default_args.add_argument('-v', '--verbose', default=True, type=boolean_string,
                          help='Prints information on different steps.')
default_args.add_argument('-s', '--seed', default=_random.randint(0, int(1e11)), type=int, help='Seed')
default_args.add_argument('-f', '--force', default=False, type=boolean_string, help='Force')
default_args.add_argument('-pat', '--patience', default=31, type=int, help='Early Stopping patience epochs.')
default_args.add_argument('-lsp', '--load_sparse', default=False, type=boolean_string, help='Load sparse dataset.')
default_args.add_argument('-nf', '--num_folds', default=None, type=int, help='Number of folds.')
default_args.add_argument('-rt', '--split_ratio', default=[0.6, 0.2, 0.2], nargs='*', type=float,
                          help='Split ratio. Eg: 0.6 0.2 0.2 or 0.8 0.2')

default_args.add_argument('--node-rank', default=0, type=int,
                          help='Node rank for distributed training')
default_args.add_argument('--num-nodes', default=1, type=int,
                          help='Number of nodes for distributed training')
default_args.add_argument('--world-size', default=None, type=int,
                          help='Number of nodes for distributed training')
default_args.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                          help='Url used to set up distributed training')
default_args.add_argument('--dist-backend', default='nccl', type=str,
                          help='Distributed backend')
