import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn import ModuleList
from torch.nn import Sequential, ReLU, Linear
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch_geometric.utils import degree
from ogb.graphproppred import PygGraphPropPredDataset, Evaluator
from torch_geometric.datasets.gnn_benchmark_dataset import GNNBenchmarkDataset
from ogb.lsc import PygPCQM4MDataset, PCQM4MEvaluator, PygPCQM4Mv2Dataset, PCQM4Mv2Evaluator
from ogb.graphproppred.mol_encoder import AtomEncoder
from torch_geometric.loader import DataLoader
from torch_geometric.nn import BatchNorm, global_mean_pool
from models.pna.pna_net import *
from models.pna.pna_config import args
from pathlib import Path
from os.path import join
import numpy as np
from tqdm import tqdm
import logging
import os
import csv
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F 
from models.retrieval.retrieval_model import Retrieval
from models.retrieval.attention import Attention
from models.retrieval.graphretrieval import GraphRetrieval
from models.utils.evaluator import CMetrics

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
crossentroy_criterion = torch.nn.CrossEntropyLoss().to(device)
nllloss_criterion = nn.NLLLoss().to(device)
cls_criterion = torch.nn.BCEWithLogitsLoss().to(device)
reg_criterion = torch.nn.MSELoss().to(device)

def train(model, device, loader, optimizer, task_type, retrieval_engine=None):
    model.train()
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1 or batch.batch[-1] == 0:
            pass
        else:
            if args.retrieval:
                batch_retrieval_graph_list = retrieval_engine.encode(batch, train=train)
                batch_retrieval_graph_list = [graph.to(device) for graph in batch_retrieval_graph_list]
                pred = model.forward(batch, batch_retrieval_graph_list)
                if pred.size(-1) == 1 and len(batch.y.size()) == 1:
                     pred = pred.view(-1,)
                optimizer.zero_grad()
                if args.source == "ogb" or args.source == "lsc":
                ## ignore nan targets (unlabeled) when computing training loss.
                    is_labeled = batch.y == batch.y
                    if "classification" in task_type:
                        loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                    else:
                        loss= reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                elif args.source == "image":
                    loss = nllloss_criterion(pred, batch.y)
            else:
                 pred = model.forward(batch)
                 if pred.size(-1) == 1 and len(batch.y.size()) == 1:
                     pred = pred.view(-1,)

                 optimizer.zero_grad()
                 if args.source == "ogb" or args.source == "lsc":
                    ## ignore nan targets (unlabeled) when computing training loss.
                    is_labeled = batch.y == batch.y
                    if "classification" in task_type:
                        loss = cls_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                    else:
                        loss= reg_criterion(pred.to(torch.float32)[is_labeled], batch.y.to(torch.float32)[is_labeled])
                 elif args.source == "image":
                    loss = crossentroy_criterion (pred, batch.y)
            
            loss.backward()
            optimizer.step()

def eval(model, device, loader, retrieval_engine=None, evaluator=None, metric_name="rocauc"):
    model.eval()
    y_true = []
    y_pred = []
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        if batch.x.shape[0] == 1:
            pass
        else:
            with torch.no_grad():
                if args.retrieval:
                    batch_retrieval_graph_list = retrieval_engine.encode(batch, train=False)
                    batch_retrieval_graph_list = [graph.to(device) for graph in batch_retrieval_graph_list]
                    pred = model.forward(batch, batch_retrieval_graph_list)
                    if pred.size(-1) == 1 and len(batch.y.size()) == 1:
                         pred = pred.view(-1,)
                else:
                    pred = model.forward(batch)
                    if pred.size(-1) == 1 and len(batch.y.size()) == 1:
                         pred = pred.view(-1,)
            
            y_true.append(batch.y.detach().cpu())
            y_pred.append(pred.detach().cpu())
           

    y_true = torch.cat(y_true, dim = 0).numpy()
    y_pred = torch.cat(y_pred, dim = 0).numpy()
    pred_input_dict = {"y_true": y_true, "y_pred": y_pred}
    if evaluator is not None:
        return evaluator.eval(pred_input_dict)[metric_name]
    else:
        roc_list = []
        for i in range(y_true.shape[1]):
            # AUC is only defined when there is at least one positive data.
            if np.sum(y_true[:, i] == 1) > 0 and np.sum(y_true[:, i] == 0) > 0:
                is_valid = y_true[:, i] >= 0
                roc_list.append(roc_auc_score(y_true[is_valid, i], y_pred[is_valid, i]))
            else:
                print('{} is invalid'.format(i))

        if len(roc_list) < y_true.shape[1]:
            print(len(roc_list))
            print('Some target is missing!')
            print('Missing ratio: %f' % (1 - float(len(roc_list)) / y_true.shape[1]))
        return sum(roc_list) / len(roc_list)

def main():
    # Training settings
    Path(args.output_model_dir).mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    ### automatic dataloading and splitting

    if args.source == "ogb":
        dataset = PygGraphPropPredDataset(name = args.dataset)
        if args.feature == 'full':
            pass
        elif args.feature == 'simple':
            print('using simple feature')
            # only retain the top two node/edge features
            dataset.data.x = dataset.data.x[:,:2]
            dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

        split_idx = dataset.get_idx_split()
        ### automatic evaluator. takes dataset name as input
        train_dataset = dataset[split_idx["train"]]
        valid_dataset = dataset[split_idx["valid"]]
        test_dataset = dataset[split_idx["test"]]
    
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers = args.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        test_loader =  DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        num_classes = dataset.num_tasks
        task_type = "classification"

    elif args.source == "image":
        train_dataset = GNNBenchmarkDataset("dataset", name=args.dataset)
        valid_dataset = GNNBenchmarkDataset("dataset", name=args.dataset, split="val")
        test_dataset = GNNBenchmarkDataset("dataset", name=args.dataset, split="test")
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers = args.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        test_loader =  DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        num_classes = 10
        task_type = "classification"
    
    elif args.source == "lsc":
        
        if args.dataset == "PCQM4M":
             dataset = PygPCQM4MDataset(root = 'dataset/')
        elif args.dataset == "PCQM4Mv2":
             dataset = PygPCQM4Mv2Datasett(root = 'dataset/')
        else:
                raise ValueError("{} is not supported".format(args.dataset))

        split_idx = dataset.get_idx_split()
        train_dataset = dataset[split_idx["train"]][:1000]
        valid_dataset = dataset[split_idx["valid"]][:500]
        test_dataset = dataset[split_idx["valid"]][:500] # test set is hidden
    
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers = args.num_workers)
        valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        test_loader =  DataLoader(test_dataset,  batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
        num_classes = 1
        task_type = "regression"
    else:
         raise ValueError("{} is not supported".format(args.source))

    retrieval_engine = None 
    if args.retrieval:
        retrieval_dataset = dataset[split_idx["train"]]
        # using the training dataset
        retrieval_engine = Retrieval(retrieval_dataset, num_classes, args, args.gnn)
        deg = torch.zeros(20, dtype=torch.long)
        for data in train_loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        base_model = PNA_NET(num_classes, deg, emb_dim=args.emb_dim, num_layer=args.num_layer, source=args.source, dataset=args.dataset).to(device)
        if not torch.cuda.is_available():
            base_model.load_state_dict(torch.load(args.pretrained_model_path,
                                                map_location=torch.device('cpu'))["model"], strict=False)
        else:
            base_model.load_state_dict(torch.load(args.pretrained_model_path)["model"], strict=False)
            
        adapter = Attention(args.emb_dim, args.k)
        model   = GraphRetrieval(gnn=base_model, adapter=adapter, source=args.source, dataset=args.dataset)
    else:
        deg = torch.zeros(20, dtype=torch.long)
        for data in train_loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        model = PNA_NET(num_classes, deg, emb_dim=args.emb_dim, num_layer=args.num_layer, source=args.source, dataset=args.dataset).to(device)
    
    model.to(device)

    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    if 'classification' == task_type:
        best_val_perf = -100
    else:
        best_val_perf = 100
    best_train_perf = 0
    best_test_perf = 0
    for epoch in range(1, args.epochs + 1):
        improve_flag = False
        print("=====Epoch {}".format(epoch))
        print('Training...')
        train(model, device, train_loader, optimizer, task_type, retrieval_engine=retrieval_engine)
        print('Evaluating...')
        
        if args.source == "ogb":
            if args.dataset == "ogbg-molmuv":
                train_perf = eval(model, device, train_loader, retrieval_engine)
                valid_perf = eval(model, device, valid_loader, retrieval_engine)
                test_perf = eval(model, device, test_loader,  retrieval_engine)
            else:
                train_perf = eval(model, device, train_loader, retrieval_engine, Evaluator(args.dataset), "rocauc")
                valid_perf = eval(model, device, valid_loader, retrieval_engine, Evaluator(args.dataset), "rocauc")
                test_perf =  eval(model, device, test_loader, retrieval_engine,  Evaluator(args.dataset), "rocauc")
        elif args.source == "image":
             train_perf = eval(model, device, train_loader, retrieval_engine, CMetrics("acc"), "acc")
             valid_perf = eval(model, device, valid_loader, retrieval_engine, CMetrics("acc"), "acc")
             test_perf = eval(model, device, test_loader,  retrieval_engine, CMetrics("acc"), "acc")
        
        elif args.source == "lsc":
             if args.dataset == "PCQM4M":
                 train_perf = eval(model, device, train_loader, retrieval_engine, PCQM4MEvaluator(), "mae")
                 valid_perf = eval(model, device, valid_loader, retrieval_engine, PCQM4MEvaluator(), "mae")
                 test_perf = eval(model, device, test_loader,  retrieval_engine,  PCQM4MEvaluator(), "mae")
             elif args.dataset == "PCQM4Mv2": 
                 train_perf = eval(model, device, train_loader, retrieval_engine, PCQM4Mv2Evaluator(), "mae")
                 valid_perf = eval(model, device, valid_loader, retrieval_engine, PCQM4Mv2Evaluator(), "mae")
                 test_perf = eval(model, device, test_loader,  retrieval_engine,  PCQM4Mv2Evaluator(), "mae")
             else:
                raise ValueError("{} is not supported".format(args.dataset))
        
        else:
            raise ValueError("{} is not supported".format(args.source))
                

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
        if 'classification' == task_type:
            if valid_perf > best_val_perf:
                improve_flag=True
                best_val_perf = valid_perf
                best_test_perf = test_perf
                best_train_perf = train_perf
                if args.output_model_dir != "":
                    output_model_path = join(args.output_model_dir,
                                             "{}_{}_{}.pth".format(args.dataset, args.gnn, args.k))
                    saved_model_dict = {
                        "model": model.state_dict()
                    }
                    torch.save(saved_model_dict, output_model_path)
        else:
            #regression problem
            if valid_perf < best_val_perf:
                best_val_perf = valid_perf
                best_test_perf = test_perf
                best_train_perf = train_perf
                if args.output_model_dir != "":
                    output_model_path = join(args.output_model_dir,
                                             "{}_{}_{}.pth".format(args.dataset, "pna", args.k))
                    saved_model_dict = {
                        "model": model.state_dict()
                    }
                    torch.save(saved_model_dict, output_model_path)

    print('Finished training!')
    print('Best validation score: {}'.format(best_val_perf))
    print('Test score: {}'.format(best_test_perf))

    if args.retrieval:
        args.retrieval=0
        print("test whether the parameter has been changed:")
        if args.dataset == "ogbg-molmuv":
            train_perf = eval(model.gnn, device, train_loader)
            valid_perf = eval(model.gnn, device, valid_loader)
            test_perf =  eval(model.gnn, device, test_loader)
        else:
            train_perf = eval(model.gnn, device, train_loader, evaluator= evaluator)
            valid_perf = eval(model.gnn, device, valid_loader, evaluator= evaluator)
            test_perf =  eval(model.gnn, device, test_loader, evaluator= evaluator)
        
        print("base_model_train_performance:", test_perf)
        print("base_model_valid_performance:", valid_perf)
        print("base_model_test_performance:", test_perf)

    if not os.path.exists("save/total.csv"):
        csvwriter = csv.writer(open("save/total.csv", "w"))
        csvwriter.writerow(["dataset", "model", "use_retrieval", "retrieval_num", "evaluation_metric", "train", "val", "test"])
    else:
        csvwriter = csv.writer(open("save/total.csv", "a"))

    csvwriter.writerow([args.dataset, args.gnn, args.retrieval, args.k, args.metrics, best_train_perf, best_val_perf, best_test_perf])

def main_test():
    dataset = PygGraphPropPredDataset(name = args.dataset)
    if args.feature == 'full':
        pass
    elif args.feature == 'simple':
        print('using simple feature')
        # only retain the top two node/edge features
        dataset.data.x = dataset.data.x[:,:2]
        dataset.data.edge_attr = dataset.data.edge_attr[:,:2]

    split_idx = dataset.get_idx_split()
    ### automatic evaluator. takes dataset name as input
    evaluator = Evaluator(args.dataset)

    train_loader = DataLoader(dataset[split_idx["train"]], batch_size=args.batch_size, shuffle=True,  num_workers = args.num_workers)
    valid_loader = DataLoader(dataset[split_idx["valid"]], batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    test_loader =  DataLoader(dataset[split_idx["test"]],  batch_size=args.batch_size, shuffle=False, num_workers = args.num_workers)
    num_classes = dataset.num_tasks

    if args.retrieval:
        # using the training dataset
        retrieval_engine = Retrieval(train_dataset, num_classes, args, args.gnn)
        deg = torch.zeros(20, dtype=torch.long)
        for data in train_loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        base_model = PNA_NET(num_classes, deg, emb_dim=args.emb_dim, num_layer=args.num_layer, source=args.source, dataset=args.dataset).to(device)
        adapter = Attention(args.emb_dim, args.k)
        model  = GraphRetrieval(gnn=base_model, adapter=adapter, source=args.source, dataset=args.dataset)

        trained_model_path = "./save/{}_{}_{}.pth".format(args.dataset, args.gnn, args.k)
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(trained_model_path ,
                                             map_location=torch.device('cpu'))["model"], strict=False)
        else:
           model.load_state_dict(torch.load(trained_model_path )["model"], strict=False)
        model.to(device)
        if args.dataset == "ogbg-molmuv":
            train_perf = eval(model, device, train_loader, retrieval_engine)
            valid_perf = eval(model, device, valid_loader, retrieval_engine)
            test_perf = eval(model, device, test_loader,  retrieval_engine)
        else:
            train_perf = eval(model, device, train_loader, retrieval_engine, evaluator)
            valid_perf = eval(model, device, valid_loader, retrieval_engine, evaluator)
            test_perf =  eval(model, device, test_loader, retrieval_engine, evaluator)
        
        print("graphretrieval_train_performance:", test_perf)
        print("graphretrieval_valid_performance:", valid_perf)
        print("graphretrieval_test_performance:", test_perf)

        args.retrieval=0
        print("test whether the parameter has been changed:")
        if args.dataset == "ogbg-molmuv":
            train_perf = eval(model.gnn, device, train_loader)
            valid_perf = eval(model.gnn, device, valid_loader)
            test_perf =  eval(model.gnn, device, test_loader)
        else:
            train_perf = eval(model.gnn, device, train_loader, evaluator= evaluator)
            valid_perf = eval(model.gnn, device, valid_loader, evaluator= evaluator)
            test_perf =  eval(model.gnn, device, test_loader, evaluator= evaluator)
        
        print("base_model_train_performance:", test_perf)
        print("base_model_valid_performance:", valid_perf)
        print("base_model_test_performance:", test_perf)

    else:
        trained_model_path = "./save/{}_{}_{}.pth".format(args.dataset, args.gnn, "0")
        deg = torch.zeros(20, dtype=torch.long)
        for data in train_loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())

        model = PNA_NET(num_classes, deg, emb_dim=args.emb_dim, num_layer=args.num_layer, source=args.source, dataset=args.dataset).to(device)
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(trained_model_path ,
                                             map_location=torch.device('cpu'))["model"], strict=False)
        else:
           model.load_state_dict(torch.load(trained_model_path )["model"], strict=False)
        model.to(device)
        if args.dataset == "ogbg-molmuv":
            train_perf = eval(model, device, train_loader)
            valid_perf = eval(model, device, valid_loader)
            test_perf =  eval(model, device, test_loader)
        else:
            train_perf = eval(model, device, train_loader, evaluator= evaluator)
            valid_perf = eval(model, device, valid_loader, evaluator= evaluator)
            test_perf =  eval(model, device, test_loader, evaluator= evaluator)
        
        print("base_model_train_performance:", test_perf)
        print("base_model_valid_performance:", valid_perf)
        print("base_model_test_performance:", test_perf)

if __name__ == "__main__":
    main()