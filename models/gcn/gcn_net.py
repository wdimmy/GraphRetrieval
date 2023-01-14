import torch
from torch_geometric.nn import  global_mean_pool
import torch.nn.functional as F
import torch.nn as nn
from models.gcn.gcn_model import GCNConv
from ogb.graphproppred.mol_encoder import AtomEncoder

### GNN to generate node embedding
class GCN_NET(torch.nn.Module):
    """
    Output:
        node representations
    """
    def __init__(self, num_tasks, num_layer, emb_dim, source, dataset, drop_ratio = 0.5, JK = "last", residual = False):
        '''
            emb_dim (int): node embedding dimensionality
            num_layer (int): number of GNN message passing layers
        '''
        super(GCN_NET, self).__init__()
        self.source = source 
        self.dataset = dataset 
        self.num_tasks = num_tasks
        self.num_layer = num_layer
        self.drop_ratio = drop_ratio
        self.emb_dim = emb_dim
        self.JK = JK
        ### add residual connection or not
        self.residual = residual

        if self.source == "ogb":
            self.node_emb = AtomEncoder(emb_dim=self.emb_dim)
        elif self.source == "image":
            if self.dataset == "CIFAR10":
               self.node_emb =  nn.Linear(3, self.emb_dim)
            elif self.dataset == "MNIST":
               self.node_emb = nn.Linear(1, self.emb_dim)
        elif self.source == "lsc":
            self.node_emb = AtomEncoder(emb_dim=self.emb_dim)
        elif self.source == "ddi":
            self.node_emb = nn.Linear(9, self.emb_dim)

        ###List of GNNs
        self.convs = torch.nn.ModuleList()
        self.batch_norms = torch.nn.ModuleList()

        for layer in range(num_layer):
            self.convs.append(GCNConv(emb_dim, self.source))
            self.batch_norms.append(torch.nn.BatchNorm1d(emb_dim))
        
        self.pool = global_mean_pool
        self.graph_pred_linear = nn.Linear(self.emb_dim, self.num_tasks)
        
    def get_graph_representation(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        ### computing input node embedding
        h_list = [self.node_emb(x)]
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            h_node = h_list[-1]
        elif self.JK == "sum":
            h_node = 0
            for layer in range(self.num_layer + 1):
                h_node += h_list[layer]

        h_graph = self.pool(h_node, batched_data.batch)
        return h_graph

    def forward(self, batched_data):
        x, edge_index, edge_attr, batch = batched_data.x, batched_data.edge_index, batched_data.edge_attr, batched_data.batch
        ### computing input node embedding
        h_list = [self.node_emb(x)]
        for layer in range(self.num_layer):
            h = self.convs[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)
            if layer == self.num_layer - 1:
                #remove relu for the last layer
                h = F.dropout(h, self.drop_ratio, training = self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training = self.training)

            if self.residual:
                h += h_list[layer]

            h_list.append(h)

        ### Different implementations of Jk-concat
        if self.JK == "last":
            h_node = h_list[-1]
        elif self.JK == "sum":
            h_node = 0
            for layer in range(self.num_layer + 1):
                h_node += h_list[layer]

        h_graph = self.pool(h_node, batched_data.batch)
        return self.graph_pred_linear(h_graph)

       
if __name__ == '__main__':
     pass 