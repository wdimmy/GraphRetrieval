import torch
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data


class PairData(Data):
    def __init__(self, edge_index_s=None, x_s=None, edge_index_t=None, x_t=None, y_s=None, y_t=None, batch_s=None, batch_t=None):
        super().__init__()
        self.edge_index_s = edge_index_s
        self.x_s = x_s
        self.edge_index_t = edge_index_t
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


edge_index_s = torch.tensor([
     [0, 0, 0, 0],
     [1, 2, 3, 4],
 ])
x_s = torch.randn(5, 16)  # 5 nodes.
edge_index_t = torch.tensor([
     [0, 0, 0],
     [1, 2, 3],
 ])
x_t = torch.randn(4, 16)  # 4 nodes.

data = PairData(edge_index_s, x_s, edge_index_t, x_t, batch_s = torch.zeros(5), batch_t = torch.zeros(4), y_s=torch.tensor([1]), y_t=torch.tensor([2]))
data_list = [data, data, data, data, data, data]

loader = DataLoader(data_list, batch_size=4)
batch = next(iter(loader))

print(batch)
print(batch.y_s)
print(batch.y_t)
print(batch.edge_index_s)
print(batch.batch_s)
