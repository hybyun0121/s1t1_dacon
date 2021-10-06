import os
import gc
import copy
import math
import argparse

import numpy as np

import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d
from torch.optim.lr_scheduler import ReduceLROnPlateau, CosineAnnealingLR
from tqdm.notebook import tqdm

from sklearn.model_selection import train_test_split

from torch_geometric.nn import CGConv

from torch_geometric.nn import global_mean_pool, graclus
from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from train import train_mode
from preprocessing import smile2graph

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--optimizer', type=str, default='adamW')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--learning_rate', type=float, default=0.01)
    parser.add_argument('--dev_rate', type=float, default=0.3)
    parser.add_argument('--epochs', type=int, default=50)

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    file_path = os.path.join(os.getcwd(), 'data')

    sm2graph = smile2graph(file_path)
    train, test = sm2graph.add_mol_info()

    train_list = sm2graph.get_mol_graph(train, is_train=True)
    test_list = sm2graph.get_mol_graph(test, is_train=False)

    trainer = train_mode(args, train_list, device)
    trainer.train_pipe()

if __name__ == '__main__':
    main()