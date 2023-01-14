import torch
import torch.nn as nn 
from torch import Tensor
from torch_geometric.typing import OptTensor
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import BondEncoder


### GIN convolution along the graph structure
class GINConv(MessagePassing):
    def __init__(self, emb_dim, source="ogb"):
        '''
            emb_dim (int): node embedding dimensionality
        '''

        super(GINConv, self).__init__(aggr = "add")
        self.source = source 
        self.mlp = torch.nn.Sequential(torch.nn.Linear(emb_dim, 2*emb_dim),
                                       torch.nn.BatchNorm1d(2*emb_dim), torch.nn.ReLU(),
                                       torch.nn.Linear(2*emb_dim, emb_dim))
        self.eps = torch.nn.Parameter(torch.Tensor([0]))


        if self.source == "ogb" or self.source == "lsc":
            self.bond_encoder = BondEncoder(emb_dim = emb_dim)
        elif self.source == "image":
            self.bond_encoder = nn.Linear(1, emb_dim)
        elif self.source == "ddi":
            self.bond_encoder = nn.Linear(3, emb_dim)
        else:
            raise ValueError("{} is not supported".format(source))

    def forward(self, x, edge_index, edge_attr):
        if self.source == "ogb" or self.source == "lsc" or self.source == "ddi":
            edge_embedding = self.bond_encoder(edge_attr)
        else:
            edge_embedding = self.bond_encoder(edge_attr.unsqueeze(1))
        out = self.mlp((1 + self.eps) *x + self.propagate(edge_index, x=x, edge_attr=edge_embedding))

        return out

    def message(self, x_j: Tensor, edge_attr: OptTensor):
        if edge_attr is not None:
            return F.relu(x_j + edge_attr)
        else:
            return x_j

    def update(self, aggr_out):
        return aggr_out

if __name__ == "__main__":
    pass
