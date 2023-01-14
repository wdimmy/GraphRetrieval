import torch.nn as nn
import torch.optim as optim
from models.pna.pna_net import PNA_NET
from tqdm import tqdm
from os.path import join
from torch_geometric.utils import degree
from models.retrieval.retrieval_model import Retrieval
from models.retrieval.attention import Attention
from models.retrieval.graphretrieval import GraphRetrieval
import os
import csv
from models.gcn.gcn_config import args
import torch
from torch_geometric.loader import DataLoader
from models.utils.ddi_util import Classifier, CustomizedBatch, PairData
torch.autograd.set_detect_anomaly(True)

device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
crossentroy_criterion = torch.nn.CrossEntropyLoss().to(device)
nllloss_criterion = nn.NLLLoss().to(device)
cls_criterion = torch.nn.BCEWithLogitsLoss().to(device)
reg_criterion = torch.nn.MSELoss().to(device)


def train(model, device, loader, optimizer, task_type, epoch, classifier, retrieval_engine=None):
    model.train()
    classifier.train()
    with tqdm(loader, unit="batch") as tepoch:
        tepoch.set_description(f"Epoch {epoch}")
        for batch in tepoch:
            batch = batch.to(device)
            if args.retrieval:
                batch_retrieval_graph_list = retrieval_engine.encode(batch, train=train)
                batch_retrieval_graph_list = [graph.to(device) for graph in batch_retrieval_graph_list]
                pred = model.forward(batch, batch_retrieval_graph_list, device)
                if pred.size(-1) == 1 and len(batch.y.size()) == 1:
                    pred = pred.view(-1, )
                optimizer.zero_grad()
                loss = nllloss_criterion(pred, batch.y_s)
            else:
                graph1_embed = model.get_graph_representation(CustomizedBatch(batch.x_s, batch.edge_index_s, batch.batch_s, batch.y_s, batch.edge_feat_s))
                graph2_embed = model.get_graph_representation(CustomizedBatch(batch.x_t, batch.edge_index_t, batch.batch_t, batch.y_t, batch.edge_feat_t))
                pred = classifier(graph1_embed, graph2_embed)
                optimizer.zero_grad()
                loss = crossentroy_criterion(pred, batch.y_s)

            loss.backward()
            optimizer.step()
            tepoch.set_postfix(loss=loss.item())


def eval(model, device, loader, classifier, retrieval_engine=None):
    model.eval()
    classifier.eval()
    total_num = 0
    correct = 0
    for step, batch in enumerate(tqdm(loader, desc="Iteration")):
        batch = batch.to(device)
        with torch.no_grad():
            if args.retrieval:
                batch_retrieval_graph_list = retrieval_engine.encode(batch, train=train)
                batch_retrieval_graph_list = [graph.to(device) for graph in batch_retrieval_graph_list]
                pred = model.forward(batch, batch_retrieval_graph_list, device)
                if pred.size(-1) == 1 and len(batch.y.size()) == 1:
                    pred = pred.view(-1, )
                correct += (pred.argmax(dim=1) == batch.y_s).sum()
                total_num += batch.y_s.size(0)

            else:
                graph1_embed = model.get_graph_representation(CustomizedBatch(batch.x_s, batch.edge_index_s, batch.batch_s, batch.y_s, batch.edge_feat_s))
                graph2_embed = model.get_graph_representation(CustomizedBatch(batch.x_t, batch.edge_index_t, batch.batch_t, batch.y_t, batch.edge_feat_t))
                pred = classifier(graph1_embed, graph2_embed)

                correct += (pred.argmax(dim=1) == batch.y_s).sum()
                total_num += batch.y_s.size(0)

    print("Accuracy=", correct/total_num)
    return correct / total_num


def main():
    # Training settings
    ### automatic dataloading and splitting
    if args.source == "ddi":
        from tdc.multi_pred import Catalyst
        from ogb.utils import smiles2graph
        data = Catalyst(name='USPTO_Catalyst', path="./dataset")
        split = data.get_split()

        train_data = split["train"]
        valid_data = split["valid"]
        test_data = split["test"]

        train_datalist = []
        valid_datalist = []
        test_datalist = []

        for y, graph_str1, graph_str2 in zip(train_data["Y"], train_data["Reactant"], train_data["Product"]):
            graph1 = smiles2graph(graph_str1)
            graph2 = smiles2graph(graph_str2)
            data = PairData(edge_index_s=torch.tensor(graph1["edge_index"], dtype=torch.long),
                            edge_feat_s=torch.tensor(graph1["edge_feat"], dtype=torch.float),
                            x_s=torch.tensor(graph1["node_feat"], dtype=torch.float),
                            edge_index_t=torch.tensor(graph2["edge_index"], dtype=torch.long),
                            edge_feat_t=torch.tensor(graph2["edge_feat"], dtype=torch.float),
                            x_t=torch.tensor(graph2["node_feat"], dtype=torch.float),
                            y_s=torch.tensor([y-1]).long(),
                            y_t=torch.tensor([y-1]).long(),
                            batch_s=torch.zeros(len(graph1["node_feat"])).long(),
                            batch_t=torch.zeros(len(graph2["node_feat"])).long()
                            )
            train_datalist.append(data)

        for y, graph_str1, graph_str2 in zip(valid_data["Y"], valid_data["Reactant"], valid_data["Product"]):
            graph1 = smiles2graph(graph_str1)
            graph2 = smiles2graph(graph_str2)
            data = PairData(edge_index_s=torch.tensor(graph1["edge_index"], dtype=torch.long),
                            edge_feat_s=torch.tensor(graph1["edge_feat"], dtype=torch.float),
                            x_s=torch.tensor(graph1["node_feat"], dtype=torch.float),
                            edge_index_t=torch.tensor(graph2["edge_index"], dtype=torch.long),
                            edge_feat_t=torch.tensor(graph2["edge_feat"], dtype=torch.float),
                            x_t=torch.tensor(graph2["node_feat"], dtype=torch.float),
                            y_s=torch.tensor([y-1]).long(),
                            y_t=torch.tensor([y-1]).long(),
                            batch_s=torch.zeros(len(graph1["node_feat"])).long(),
                            batch_t=torch.zeros(len(graph2["node_feat"])).long()
                            )
            valid_datalist.append(data)

        for y, graph_str1, graph_str2 in zip(test_data["Y"], test_data["Reactant"], test_data["Product"]):
            graph1 = smiles2graph(graph_str1)
            graph2 = smiles2graph(graph_str2)
            data = PairData(edge_index_s=torch.tensor(graph1["edge_index"], dtype=torch.long),
                            edge_feat_s=torch.tensor(graph1["edge_feat"], dtype=torch.float),
                            x_s=torch.tensor(graph1["node_feat"], dtype=torch.float),
                            edge_index_t=torch.tensor(graph2["edge_index"], dtype=torch.long),
                            edge_feat_t=torch.tensor(graph2["edge_feat"], dtype=torch.float),
                            x_t=torch.tensor(graph2["node_feat"], dtype=torch.float),
                            y_s=torch.tensor([y-1]).long(),
                            y_t=torch.tensor([y-1]).long(),
                            batch_s=torch.zeros(len(graph1["node_feat"])).long(),
                            batch_t=torch.zeros(len(graph2["node_feat"])).long()
                            )
            test_datalist.append(data)
        print("Finish Loading data end")
        train_loader = DataLoader(train_datalist, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        valid_loader = DataLoader(valid_datalist, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
        test_loader = DataLoader(test_datalist,   batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

        args.num_classes = 888
        num_classes = 888
        task_type = "classification"

    retrieval_engine = None
    if args.retrieval:
        # using the training dataset
        retrieval_engine = Retrieval(train_datalist, num_classes, args, args.gnn)
        deg = torch.zeros(20, dtype=torch.long)
        for data in train_loader:
            d = degree(data.edge_index_s[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        base_model = PNA_NET(num_classes, deg, emb_dim=args.emb_dim, num_layer=args.num_layer, source=args.source, dataset=args.dataset).to(device)
        classifier = Classifier(args.emb_dim, args.num_classes).to(device)
        if not torch.cuda.is_available():
            base_model.load_state_dict(torch.load(args.pretrained_model_path,
                                                  map_location=torch.device('cpu'))["model"], strict=False)
            classifier.load_state_dict(torch.load(args.pretrained_model_path,
                                                  map_location=torch.device('cpu'))["classifier"], strict=False)
        else:
            base_model.load_state_dict(torch.load(args.pretrained_model_path)["model"], strict=False)
            classifier.load_state_dict(torch.load(args.pretrained_model_path)["classifier"], strict=False)
        if args.source == "ddi":
            adapter = Attention(2 * args.emb_dim, args.k)
        else:
            adapter = Attention(args.emb_dim, args.k)
        model = GraphRetrieval(gnn=base_model, adapter=adapter, source=args.source, dataset=args.dataset, classifier=classifier).to(device)

    else:
        deg = torch.zeros(20, dtype=torch.long)
        for data in train_loader:
            d = degree(data.edge_index_s[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        model = PNA_NET(num_classes, deg, emb_dim=args.emb_dim, num_layer=args.num_layer, source=args.source, dataset=args.dataset).to(device)
        classifier = Classifier(args.emb_dim, args.num_classes).to(device)

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
        train(model, device, train_loader, optimizer, task_type, epoch, classifier, retrieval_engine=retrieval_engine)
        print('Evaluating...')

        if args.source == "ddi":
            if epoch % 5 == 0:
                train_perf = eval(model, device, train_loader, classifier, retrieval_engine)
            else:
                train_perf = 0
            valid_perf = eval(model, device, valid_loader, classifier, retrieval_engine)
            test_perf = eval(model,  device, test_loader, classifier, retrieval_engine)

        else:
            raise ValueError("{} is not supported".format(args.source))

        print({'Train': train_perf, 'Validation': valid_perf, 'Test': test_perf})
        if 'classification' == task_type:
            if valid_perf > best_val_perf:
                improve_flag = True
                best_val_perf = valid_perf
                best_test_perf = test_perf
                best_train_perf = train_perf
                if args.output_model_dir != "":
                    output_model_path = join(args.output_model_dir,
                                             "{}_{}_{}.pth".format(args.dataset, args.gnn, args.k))
                    saved_model_dict = {
                        "model": model.state_dict(),
                        "classifier": classifier.state_dict()
                    }
                    torch.save(saved_model_dict, output_model_path)
        else:
            # regression problem
            if valid_perf < best_val_perf:
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

    print('Finished training!')
    print('Best validation score: {}'.format(best_val_perf))
    print('Test score: {}'.format(best_test_perf))

    if not os.path.exists(args.output_model_dir + "/total.csv"):
        csvwriter = csv.writer(open(args.output_model_dir + "/total.csv", "w"))
        csvwriter.writerow(
            ["dataset", "model", "use_retrieval", "retrieval_num", "evaluation_metric", "train", "val", "test"])
    else:
        csvwriter = csv.writer(open(args.output_model_dir + "/total.csv", "a"))

    csvwriter.writerow(
        [args.dataset, args.gnn, args.retrieval, args.k, args.metrics, best_train_perf, best_val_perf, best_test_perf])


if __name__ == "__main__":
    main()