from ogb.lsc import PygPCQM4MDataset, PCQM4MEvaluator, PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator
from torch_geometric.loader import DataLoader
import pickle
dataset = PygPCQM4MDataset(root = 'dataset/')

split_idx = dataset.get_idx_split()
test_dataset = dataset[split_idx["valid"]]
train_dataset = dataset[split_idx["train"]]
test_loader  =  DataLoader(test_dataset,   batch_size=128, shuffle=False, num_workers = 2)
train_loader =  DataLoader(train_dataset,  batch_size=128, shuffle=False, num_workers = 2)

results = []
for batch in test_loader:
    results.extend(batch.y.cpu().tolist())
pickle.dump(results, open("pcqm4m_test.pkl", "wb"))

print('save test')
results = []
for batch in train_loader:
    results.extend(batch.y.cpu().tolist())
pickle.dump(results, open("pcqm4m_train.pkl", "wb"))

