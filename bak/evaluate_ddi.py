import torch.nn as nn
import torch, os, pickle
from models.gcn.gcn_net import GCN_NET
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from tqdm import tqdm
from collections import defaultdict
from models.gcn.gcn_config import args
from matplotlib import pyplot
import numpy

class Classifier(nn.Module):
    def __init__(self, embed_size, num_classes):
        super().__init__()
        self.embed_size = embed_size
        self.num_classes = num_classes
        self.classifier = nn.Sequential(
            nn.Linear(2 * self.embed_size, 2 * self.embed_size),
            nn.ReLU(),
            nn.Linear(2 * self.embed_size, self.embed_size),
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


device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
model = GCN_NET(num_tasks=args.num_classes, num_layer=args.num_layer, emb_dim= args.emb_dim,
                        drop_ratio= args.drop_ratio, source= args.source, dataset= args.dataset).to(device)

classifier = Classifier(args.emb_dim, args.num_classes).to(device)

trained_path = "outputs/USPTO_Catalyst_gcn_0.pth"
trained_weights = torch.load(trained_path)
model.load_state_dict(trained_weights["model"])
classifier.load_state_dict(trained_weights["classifier"])

model.eval()
classifier.eval()
total_num = 0
correct = 0

if os.path.exists(args.output_model_dir + "/train.pkl"):
    train_datalist = pickle.load(open(args.output_model_dir + "/train.pkl", "rb"))
    valid_datalist = pickle.load(open(args.output_model_dir + "/valid.pkl", "rb"))
    test_datalist = pickle.load(open(args.output_model_dir + "/valid.pkl", "rb"))
else:
    raise ValueError("Path does not exist!")

train_results = defaultdict(int)

for batch in train_datalist:
    train_results[batch.y_s[0].item()] += 1

sorted_train = sorted(train_results.items(), key=lambda item: item[1], reverse=True)
total_num = len(train_datalist)
train_x = []
for i in range(1, 888):
    if i in train_results:
        train_x.append(train_results[i])
    else:
        train_x.append(0)

bins = numpy.linspace(1, 889, 888)
pyplot.hist(train_x, bins, alpha=0.5, label='x')
pyplot.savefig(args.output_model_dir + "/train_distribute.png")

exit()
train_loader = DataLoader(train_datalist, batch_size=1)
valid_loader = DataLoader(valid_datalist, batch_size=1)
test_loader = DataLoader(test_datalist,   batch_size=1)

test_results = defaultdict(list)
for step, batch in enumerate(tqdm(test_loader, desc="Iteration")):
    batch = batch.to(device)
    with torch.no_grad():
        graph1_embed = model.get_graph_representation(CustomizedBatch(batch.x_s, batch.edge_index_s, batch.batch_s, batch.y_s, batch.edge_feat_s))
        graph2_embed = model.get_graph_representation(CustomizedBatch(batch.x_t, batch.edge_index_t, batch.batch_t, batch.y_t, batch.edge_feat_t))
        pred = classifier(graph1_embed, graph2_embed)
        pred_label = pred.argmax(dim=1)
        if pred_label[0].item() == batch.y_s[0].item():
            test_results[batch.y_s.detach().cpu().numpy()[0]].append(1)
        else:
            test_results[batch.y_s.detach().cpu().numpy()[0]].append(0)

test_result_dict = {key: sum(value)/len(value) for key, value in test_results.items()}
sorted_test = sorted(test_result_dict.items(), key=lambda item: item[1], reverse=True)
print(sorted_train[:30])
print(sorted_test[:30])



