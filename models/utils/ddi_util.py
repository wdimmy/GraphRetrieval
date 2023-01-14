import torch.nn as nn
import torch
from torch_geometric.data import Data


class Classifier(nn.Module):
     def __init__(self, embed_size, num_classes):
         super().__init__()
         self.embed_size = embed_size
         self.num_classes = num_classes
         self.classifier = nn.Sequential(
             nn.Linear(2*self.embed_size, 2*self.embed_size),
             nn.ReLU(),
             nn.Linear(2*self.embed_size, self.embed_size),
             nn.ReLU(),
             nn.Linear(embed_size, self.num_classes)
         )

     def forward(self, x, y):
         concat = torch.cat([x, y], dim=-1)
         return self.classifier(concat)


class CustomizedBatch:
    def __init__(self, x, edge_index, batch, y, edge_attr=None):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.batch = batch
        self.y = y


class PairData(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_index_t=None, x_t=None, edge_feat_s=None, edge_feat_t=None, y_s=None, y_t=None, batch_s=None, batch_t=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.edge_feat_s = edge_feat_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
        self.edge_feat_t = edge_feat_t
        self.x_t = x_t
        self.batch_s = batch_s
        self.batch_t = batch_t
        self.y_s = y_s
        self.y_t = y_t

    def __inc__(self, key, value, *args, **kwargs):
        if key == "batch_s" or key == "batch_t":
            return 1
        if key == 'edge_index_s':
            return self.x_s.size(0)
        if key == 'edge_index_t' or key == 'batch_t':
            return self.x_t.size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)

    @property
    def num_nodes(self):
        return self.x_s.size(0) + self.x_t.size(1)

