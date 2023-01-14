source = "USPTO_Catalyst"
#train_datalist = pickle.load(open("./data/" + source + "_train.pkl", "rb"))

from ogb.utils import smiles2graph
from tdc.multi_pred import Catalyst
from tdc.multi_pred import DDI
import torch
from torch_geometric.data import Data

device = torch.device("cuda:" + str(0)) if torch.cuda.is_available() else torch.device("cpu")


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

# data = DDI(name=source)

# from tdc.utils import get_label_map
# labels = get_label_map(name = 'USPTO_Catalyst', task = 'Catalyst')

from tdc.multi_pred import DDI
data = DDI(name = 'DrugBank')
split = data.get_split()

from tdc.utils import get_label_map
labels = get_label_map(name = 'DrugBank', task = 'DDI')

#labels = get_label_map(name = 'TWOSIDES', task = 'DDI', name_column = 'Side Effect Name')
breakpoint()
keys = list(set([key for key in labels.keys()])).sort()


split = data.get_split()
train_data = split["train"] # class 'pandas.core.frame.DataFrame'>
valid_data = split["valid"] # class 'pandas.core.frame.DataFrame'>
test_data = split["test"] # class 'pandas.core.frame.DataFrame'>

for y, graph_str1, graph_str2 in zip(train_data["Y"], train_data["Drug1"], train_data["Drug2"]):
    graph1 = smiles2graph(graph_str1)
    graph2 = smiles2graph(graph_str2)
    data = PairData(edge_index_s=torch.tensor(graph1["edge_index"], dtype=torch.long),
                    edge_feat_s=torch.tensor(graph1["edge_feat"], dtype=torch.float),
                    x_s=torch.tensor(graph1["node_feat"], dtype=torch.float),
                    edge_index_t=torch.tensor(graph2["edge_index"], dtype=torch.long),
                    edge_feat_t=torch.tensor(graph2["edge_feat"], dtype=torch.float),
                    x_t=torch.tensor(graph2["node_feat"], dtype=torch.float),
                    y_s=torch.tensor([y - 1]).long(),
                    y_t=torch.tensor([y - 1]).long(),
                    batch_s=torch.zeros(len(graph1["node_feat"])).long(),
                    batch_t=torch.zeros(len(graph2["node_feat"])).long()
                    )




