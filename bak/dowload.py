# from tdc.utils import get_label_map
# from tdc.multi_pred import DDI
# data = DDI(name = 'TWOSIDES')
# split = data.get_split()
# breakpoint()
# label = get_label_map(name = 'TWOSIDES', task = 'DDI', name_column = 'Side Effect Name')
# print(len(label))
#
import torch
from torch_geometric.data import Data
from ogb.utils import smiles2graph
import ogb
graph = smiles2graph("CC1(C(N2C(S1)C(C2=O)NC(=O)C(C3=CC=CC=C3)N)C(=O)[O-])C")
print(ogb.__version__)
# filename = "./data/twosides.csv"
# with open(filename) as file:
#     data = Data(x=torch.tensor(graph["node_feat"], dtype=torch.float), edge_index=torch.tensor(graph["edge_index"], dtype=torch.long))
#



