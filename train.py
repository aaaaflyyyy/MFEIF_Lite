import warnings
warnings.filterwarnings('ignore')

import argparse
import logging
import torch
from argparse import Namespace

# import wandb

from pipeline.train_c import Train
from utils.environment_probe import EnvironmentProbe


def parse_args() -> Namespace:
    # args parser
    parser = argparse.ArgumentParser()

    # universal opt
    parser.add_argument('--id', default='MFEIF_c_woAttn', help='train process identifier')
    parser.add_argument('--folder', default='../datasets/FLIR_ADAS_v2', help='data root path')
    parser.add_argument('--size', default=256, help='resize image to the specified size')
    parser.add_argument('--cache', default='cache', help='weights cache folder')

    # checkpoint opt
    parser.add_argument('--epochs', type=int, default=4, help='epoch to train')
    # optimizer opt
    parser.add_argument('--learning_rate', type=float, default=1e-4, help='learning rate')
    # dataloader opt
    parser.add_argument('--batch_size', type=int, default=16, help='dataloader batch size')
    parser.add_argument('--num_workers', type=int, default=8, help='dataloader workers number')

    # experimental opt
    parser.add_argument('--debug', action='store_true', help='debug mode (default: off)')

    return parser.parse_args()


if __name__ == '__main__':
    config = parse_args()
    logging.basicConfig(level='INFO')
    torch.cuda.empty_cache()
    torch.multiprocessing.set_start_method('spawn')

    environment_probe = EnvironmentProbe()
    train_process = Train(environment_probe, config)
    train_process.run()
