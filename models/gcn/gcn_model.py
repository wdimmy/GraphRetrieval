import torch
import torch.nn as nn 
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder
from torch_geometric.utils import degree


### GCN convolution along the graph structure
class GCNConv(MessagePassing):
    def __init__(self, emb_dim, source="ogb"):
        super(GCNConv, self).__init__(aggr='add')
        self.source = source 
        self.linear = torch.nn.Linear(emb_dim, emb_dim)
        self.root_emb = torch.nn.Embedding(1, emb_dim)
        if self.source == "ogb" or self.source == "lsc":
            self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        elif self.source == "image":
            self.bond_encoder = nn.Linear(1, emb_dim)
        else:
            raise ValueError("{} is not supported".format(source))

    def forward(self, x, edge_index, edge_attr):
        x = self.linear(x)
        if self.source == "ogb" or self.source == "lsc":
            edge_embedding = self.bond_encoder(edge_attr)
        elif self.source == "image":
            edge_embedding = self.bond_encoder(edge_attr.unsqueeze(1))

        row, col = edge_index
        #edge_weight = torch.ones((edge_index.size(1), ), device=edge_index.device)
        deg = degree(row, x.size(0), dtype = x.dtype) + 1
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0

        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        return self.propagate(edge_index, x=x, edge_attr = edge_embedding, norm=norm) + F.relu(x + self.root_emb.weight) * 1./deg.view(-1,1)

    def message(self, x_j, edge_attr, norm):
        return norm.view(-1, 1) * F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out

if __name__ == "__main__":
    pass
