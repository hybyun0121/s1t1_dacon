import math
import copy
import torch
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from torch.optim.lr_scheduler import CosineAnnealingLR

from torch_geometric.data import Dataset
from torch_geometric.loader import DataLoader

from model import GNN
from utils import EarlyStopping

class train_mode():
    def __init__(self, args, data, device):
        self.args = args
        self.deivce = device

        self.train_set, self.dev_set = train_test_split(data, test_size=args.dev_rate, shuffle=True)
        self.train_data = DataLoader(self.train_set, batch_size=args.batch_size)
        self.test_data = DataLoader(self.dev_set, batch_size=args.batch_size)

        batch = None
        for g in self.train_data:
            batch = g
            break
        self.n_node_fet = batch.num_node_features
        self.n_edge_fet = batch.num_edge_features

    def build_network(self, hidden_dim, device):
        model = GNN(input_dims=self.n_node_fet,
                    edge_dim=self.n_edge_fet,
                    hidden_dim=hidden_dim,
                    out_dim=2,
                    device=device)

        return model.to(self.device)

    def build_optimizer(self, network, optimizer, learning_rate):
        if optimizer == "adam":
            optimizer = torch.optim.Adam(network.parameters(), lr=learning_rate)
        elif optimizer == "adamW":
            optimizer = torch.optim.AdamW(network.parameters(), lr=learning_rate,
                                          weight_decay=0.05)
        return optimizer

    def train(self, loader, model, opt, criterion):
        model.train()

        total_loss = 0
        for data in tqdm(loader):
            data = data.to(self.deivce)
            opt.zero_grad()
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, torch.tensor(data.y).to(self.deivce))
            total_loss += loss.item()
            loss.backward()
            opt.step()

        total_loss = total_loss
        Loss = round(100 * (total_loss / len(loader.dataset)), 5)

        return Loss

    def test(self, loader, model, criterion):
        model.eval()

        total_loss = 0
        for data in loader:
            data = data.to(self.device)
            out = model(data.x, data.edge_index, data.edge_attr, data.batch)
            loss = criterion(out, torch.tensor(data.y).to(self.device))
            total_loss += loss.item()

        total_loss = total_loss
        Loss = round(100 * (total_loss / len(loader.dataset)), 5)
        return Loss

    def train_pipe(self):
        criterion = torch.nn.L1Loss()
        best_model = None
        best_val_loss = 1.

        model = self.build_network(hidden_dim=self.args.hidden_dim,
                                   device=self.device)
        optimizer = self.build_optimizer(model, self.args.optimizer,
                                         self.args.learning_rate)

        # lr decay
        # scheduler = ReduceLROnPlateau(optimizer, 'min', patience=10, min_lr=1e-05)

        # Cosine annealing lr
        Q = math.floor(len(self.train_set) / 256) * 5
        scheduler = CosineAnnealingLR(optimizer, T_max=Q, eta_min=1e-06)

        # early stopping
        early_stopping = EarlyStopping(patience=10, verbose=True)

        criterion = torch.nn.L1Loss()

        for epoch in tqdm(range(1, args.epochs)):
            train_loss = self.train(self.train_data, model, optimizer, criterion)
            val_loss = self.test(self.dev_data, model, criterion)

            scheduler.step(val_loss)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_model = copy.deepcopy(model)

            print(f'Epoch: {epoch:03d}, Train loss : {train_loss:.5f}, Valid loss: {val_loss:.5f}')

            early_stopping(val_loss, model)
            if early_stopping.early_stop:
                print("Early stopping")
                break

        best_result = self.test(self.dev_data, best_model, criterion)
