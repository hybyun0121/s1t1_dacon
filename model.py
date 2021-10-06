import torch
import torch.nn.functional as F
from torch.nn import Linear, BatchNorm1d

from torch_geometric.nn import CGConv
from torch_geometric.nn import global_mean_pool


class GNN(torch.nn.Module):
    def __init__(self, input_dims, edge_dim, hidden_dim, out_dim, device):
        super(GNN, self).__init__()
        self.device = device
        self.hidden1 = hidden_dim
        self.hidden2 = hidden_dim//2

        self.conv1 = CGConv(input_dims, dim=edge_dim, batch_norm=True)
        self.conv2 = CGConv(input_dims, dim=edge_dim, batch_norm=True)
        self.conv3 = CGConv(input_dims, dim=edge_dim, batch_norm=True)
        self.conv4 = CGConv(input_dims, dim=edge_dim, batch_norm=True)

        self.lin1 = Linear(input_dims, self.hidden1)
        self.bn1 = BatchNorm1d(self.hidden1)
        self.lin2 = Linear(self.hidden1, self.hidden2)
        self.bn2 = BatchNorm1d(self.hidden2)
        self.lin3 = Linear(self.hidden2, out_dim)

    def forward(self, x, edge_index, edge_attr, batch):
        x = x.type(torch.FloatTensor).to(self.device)
        edge_index = edge_index.to(self.device)
        edge_attr = edge_attr.type(torch.FloatTensor).to(self.device)

        x = self.conv1(x, edge_index, edge_attr)
        x = self.conv2(x, edge_index, edge_attr)
        x = self.conv3(x, edge_index, edge_attr)
        x = self.conv4(x, edge_index, edge_attr)
        x = global_mean_pool(x, batch)

        x = F.relu(self.bn1(self.lin1(x)))
        x = F.relu(self.bn2(self.lin2(x)))
        return self.lin3(x)