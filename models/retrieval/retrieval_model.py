from sys import breakpointhook
from models.retrieval.build_search_engine import *
import numpy as np
import dgl


def collate(samples):
    # The input samples is a list of pairs (graph, label).
    graphs, labels = map(list, zip(*samples))
    labels = torch.tensor(np.array(labels))
    tab_sizes_n = [graphs[i].number_of_nodes() for i in range(len(graphs))]
    tab_snorm_n = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_n]
    snorm_n = torch.cat(tab_snorm_n).sqrt()
    tab_sizes_e = [graphs[i].number_of_edges() for i in range(len(graphs))]
    tab_snorm_e = [torch.FloatTensor(size, 1).fill_(1. / float(size)) for size in tab_sizes_e]
    snorm_e = torch.cat(tab_snorm_e).sqrt()
    for idx, graph in enumerate(graphs):
        graphs[idx].ndata['feat'] = graph.ndata['feat'].float()
        graphs[idx].edata['feat'] = graph.edata['feat'].float()
    batched_graph = dgl.batch(graphs)
    return (batched_graph, labels, snorm_n, snorm_e)


class Retrieval():

    def __init__(self, train_dataset, num_tasks, args, model_type):
        self.Index, self.IndexGraphMap, self.model = build_search_engine(train_dataset, num_tasks,
                                                             args, model_type=model_type)
        self.k = args.k

    def encode(self, batched_data, pixels=False, train=True):
        with torch.no_grad():
            if pixels:
                g, h, e, snorm_n, snorm_e = batched_data
                raw_graph = self.model.get_graph_representation(g, h, e, snorm_n, snorm_e)
            else:
                raw_graph = self.model.get_graph_representation(batched_data)
        retrieval_indexes = []
        normalize_flag = True
        if normalize_flag:
            raw_graph = raw_graph.cpu().detach().numpy()
            transformer = Normalizer().fit(raw_graph)
            raw_graph = transformer.transform(raw_graph)

        if train: # retrieval dropout
            _, I = self.Index.search(raw_graph, self.k+1)
            for i in range(1, self.k+1):
                retrieval_index = [int(index[i]) for index in I]
                retrieval_indexes.append(retrieval_index)
        else:
            _, I = self.Index.search(raw_graph, self.k)
            for i in range(0, self.k):
                retrieval_index = [int(index[i]) for index in I]
                retrieval_indexes.append(retrieval_index)

        batch_retrieval_graph_list = []
        for retrieval_index in retrieval_indexes:
            graph_list = []
            [graph_list.append(self.IndexGraphMap[index]) for index in retrieval_index]
            if pixels:
                batch_retrieval_graph = collate(graph_list)
            else:
                batch_retrieval_graph = Batch.from_data_list(graph_list)
            batch_retrieval_graph_list.append(batch_retrieval_graph)

        return batch_retrieval_graph_list
