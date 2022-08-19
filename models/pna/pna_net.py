import torch.nn.functional as F
from ogb.graphproppred.mol_encoder import AtomEncoder
from models.pna.pna_model import *
from torch_geometric.nn import BatchNorm, global_mean_pool
import torch.nn as nn


class PNA_NET(torch.nn.Module):
    def __init__(self, num_task, deg, emb_dim, num_layer, source, dataset):
        super(PNA_NET, self).__init__()
        self.source = source
        self.dataset = dataset 
        self.emb_dim = emb_dim
        self.deg = deg 
        self.num_task = num_task 
        self.num_layer = num_layer 
        if self.source == "ogb":
            self.node_emb = AtomEncoder(emb_dim=self.emb_dim)
        elif self.source == "image":
            if self.dataset == "CIFAR10":
               self.node_emb =  nn.Linear(3, self.emb_dim)
            elif self.dataset == "MNIST":
               self.node_emb =  nn.Linear(1, self.emb_dim)
        elif self.source == "lsc":
            self.node_emb =  AtomEncoder(emb_dim=self.emb_dim)

        aggregators = ['mean', 'min', 'max', 'std']
        scalers = ['identity', 'amplification', 'attenuation']

        self.convs = ModuleList()
        self.batch_norms = ModuleList()
        for _ in range(self.num_layer):
            conv = PNAConvSimple(in_channels=self.emb_dim, out_channels=self.emb_dim, aggregators=aggregators,
                                 scalers=scalers, deg=self.deg)
            self.convs.append(conv)
            self.batch_norms.append(BatchNorm(self.emb_dim))

        self.graph_pred_linear = Sequential(Linear(self.emb_dim, self.emb_dim//2), ReLU(),
                              Linear(self.emb_dim//2, self.emb_dim//4), ReLU(),
                              Linear(self.emb_dim//4, self.num_task))
        self.F = nn.Sigmoid()

    def get_graph_representation(self, batched_data):
        x = self.node_emb(batched_data.x)
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            h = F.relu(batch_norm(conv(x, batched_data.edge_index, None)))
            x = h + x  # residual#
            x = F.dropout(x, 0.3, training=self.training)
        h_graph = global_mean_pool(x, batched_data.batch)
        return h_graph


    def forward(self, batched_data):
        x = self.node_emb(batched_data.x)
        for conv, batch_norm in zip(self.convs, self.batch_norms):
            h = F.relu(batch_norm(conv(x, batched_data.edge_index, None)))
            x = h + x  # residual#
            x = F.dropout(x, 0.3, training=self.training)

        h_graph = global_mean_pool(x, batched_data.batch)
        if self.source == "lsc":
            return torch.clamp(self.graph_pred_linear(h_graph), min=0, max=50)
        else:
            return self.graph_pred_linear(h_graph)


       