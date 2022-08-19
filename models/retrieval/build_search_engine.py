import faiss
from torch_geometric.loader import DataLoader
from models.retrieval.customized_batch import BatchMasking
from models.pna.pna_model import *
from models.gin.gin_net import GIN_NET
from models.pna.pna_net import PNA_NET
from models.gcn.gcn_net import GCN_NET
from models.superpixels_pna.pna_net import PNANet
from sklearn.preprocessing import Normalizer
from torch_geometric.data import Batch

def build_search_engine(train_dataset, num_task, args, model_type, batch_size=32):
    device = torch.device("cuda:" + str(args.device)) if torch.cuda.is_available() else torch.device("cpu")
    if model_type == "gin":
        model = GIN_NET(num_tasks=num_task, num_layer=args.num_layer,
                       emb_dim= args.emb_dim, source=args.source, dataset=args.dataset)
        args.pretrained_model_path = "./save/{}_{}_{}.pth".\
            format(args.dataset, "gin", "0")

        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(args.pretrained_model_path,
                                             map_location=torch.device('cpu'))["model"], strict=False)
        else:
            model.load_state_dict(torch.load(args.pretrained_model_path)["model"], strict=False)

        print("Load pretrained model successfully!")
        model.to(device)
        model.eval()
        index = faiss.IndexFlatL2(args.emb_dim)
        index_to_graph_map = {}
        total_num = len(train_dataset)
        i = 0
        graph_embeddings = []
        while i < total_num:
            graph_list = train_dataset[i:i+batch_size]
            batch = Batch.from_data_list([graph for graph in graph_list])
            #batch = BatchMasking.from_data_list(graph_list)
            batch = batch.to(device)
            with torch.no_grad():
               graphs_representation = model.get_graph_representation(batch)
               graph_embeddings.append(graphs_representation)
            i += batch_size

        graph_embeddings = torch.cat(graph_embeddings, dim=0)
        graph_embeddings = graph_embeddings.cpu().detach().numpy()
        print("Index is trained:", index.is_trained)

        index.train(graph_embeddings)
        print("Index is trained:", index.is_trained)
        index.add(graph_embeddings)
        for i, graph_data in enumerate(train_dataset):
            index_to_graph_map[i] = graph_data

        return index, index_to_graph_map, model

    elif model_type == "gcn":
        model = GCN_NET(num_tasks=num_task, num_layer=args.num_layer,
                       emb_dim= args.emb_dim, source=args.source, dataset=args.dataset)
        args.pretrained_model_path = "./save/{}_{}_{}.pth".\
            format(args.dataset, "gcn", "0")

        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(args.pretrained_model_path,
                                             map_location=torch.device('cpu'))["model"], strict=False)
        else:
            model.load_state_dict(torch.load(args.pretrained_model_path)["model"], strict=False)

        print("Load pretrained model successfully!")
        model.to(device)
        model.eval()
        index = faiss.IndexFlatL2(args.emb_dim)
        index_to_graph_map = {}
        total_num = len(train_dataset)
        i = 0
        graph_embeddings = []
        while i < total_num:
            graph_list = train_dataset[i:i+batch_size]
            batch = Batch.from_data_list(graph_list)
            #batch = BatchMasking.from_data_list(graph_list)
            batch = batch.to(device)
            with torch.no_grad():
               graphs_representation = model.get_graph_representation(batch)
               graph_embeddings.append(graphs_representation)
            i += batch_size

        graph_embeddings = torch.cat(graph_embeddings, dim=0)
        graph_embeddings = graph_embeddings.cpu().detach().numpy()
        print("Index is trained:", index.is_trained)
        index.train(graph_embeddings)
        print("Index is trained:", index.is_trained)
        index.add(graph_embeddings)
        for i, graph_data in enumerate(train_dataset):
            index_to_graph_map[i] = graph_data

        return index, index_to_graph_map, model

    elif model_type == "pna":
        args.pretrained_model_path = "./save/{}_{}_{}.pth".\
            format(args.dataset, "pna", "0")
        
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  num_workers = args.num_workers)
        deg = torch.zeros(20, dtype=torch.long)
        for data in train_loader:
            d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            deg += torch.bincount(d, minlength=deg.numel())
        
        model = PNA_NET(num_task, deg, emb_dim=args.emb_dim, num_layer=args.num_layer, source=args.source, dataset=args.dataset).to(device)
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(args.pretrained_model_path,
                                             map_location=torch.device('cpu'))["model"], strict=False)
        else:
            model.load_state_dict(torch.load(args.pretrained_model_path)["model"], strict=False)

        print("Load pretrained model successfully!")
        model.to(device)
        model.eval()
        index = faiss.IndexFlatL2(args.emb_dim)
        index_to_graph_map = {}
        total_num = len(train_dataset)
        i = 0
        graph_embeddings = []
        while i < total_num:
            graph_list = train_dataset[i:i + batch_size]
            batch = Batch.from_data_list(graph_list)
            batch = batch.to(device)
            with torch.no_grad():
                graphs_representation = model.get_graph_representation(batch)
                graph_embeddings.append(graphs_representation)
            i += batch_size

        graph_embeddings = torch.cat(graph_embeddings, dim=0)
        graph_embeddings = graph_embeddings.cpu().detach().numpy()
        print("Index is trained:", index.is_trained)
        index.train(graph_embeddings)
        print("Index is trained:", index.is_trained)
        index.add(graph_embeddings)
        for i, graph_data in enumerate(train_dataset):
            index_to_graph_map[i] = graph_data

        return index, index_to_graph_map, model

    elif model_type == "superpixels_pna":
        pretrained_model_path = "./save/{}_{}_{}.pth". \
            format(args.dataset, "superpixels_pna", "0")
        pretrained_model_path_param = "./save/{}_{}_{}_param.pth". \
            format(args.dataset, "superpixels_pna", "0")

        net_param = torch.load(pretrained_model_path_param)

        model = PNANet(net_param)
        if not torch.cuda.is_available():
            model.load_state_dict(torch.load(pretrained_model_path,
                                             map_location=torch.device('cpu'))["model"], strict=False)
        else:
            model.load_state_dict(torch.load(pretrained_model_path)["model"], strict=False)

        print("Load pretrained model successfully!")
        model.to(device)
        model.eval()
        print("args:", net_param["out_dim"])
        index = faiss.IndexFlatIP(net_param["out_dim"])
        #index = faiss.IndexFlatL2(net_param["out_dim"])
        index_to_graph_map = {}
        total_num = 1000 # len(train_dataset)
        i = 0
        graph_embeddings = []
        while i < total_num:
            graph_list = zip(train_dataset[i:i + batch_size][0],train_dataset[i:i + batch_size][1])
            batch_graphs, batch_labels, batch_snorm_n, batch_snorm_e = BatchMasking.dgl_collate(graph_list)
            batch_x = batch_graphs.ndata['feat'].to(device)  # num x feat
            batch_e = batch_graphs.edata['feat'].to(device)
            batch_snorm_e = batch_snorm_e.to(device)
            batch_labels = batch_labels.to(device)
            batch_snorm_n = batch_snorm_n.to(device)  # num x 1
            with torch.no_grad():
                graphs_representation = model.get_graph_representation(batch_graphs, batch_x, batch_e, batch_snorm_n, batch_snorm_e)
                graph_embeddings.append(graphs_representation)
            i += batch_size

        graph_embeddings = torch.cat(graph_embeddings, dim=0)
        graph_embeddings = graph_embeddings.cpu().detach().numpy()
        transformer = Normalizer().fit(graph_embeddings)
        graph_embeddings = transformer.transform(graph_embeddings)
        print("Index is trained:", index.is_trained)
        index.train(graph_embeddings)
        print("Index is trained:", index.is_trained)
        index.add(graph_embeddings)
        for i, graph_data in enumerate(train_dataset):
            index_to_graph_map[i] = graph_data

        return index, index_to_graph_map, model
