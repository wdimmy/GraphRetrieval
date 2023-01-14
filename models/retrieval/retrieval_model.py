from models.retrieval.build_search_engine import *


class Retrieval():
    def __init__(self, train_dataset, num_tasks, args, model_type):
        self.Index, self.IndexGraphMap, self.model = build_search_engine(train_dataset, num_tasks,
                                                             args, model_type=model_type)
        self.k = args.k
        self.source = args.source

    def encode(self, batched_data, train=True):
        with torch.no_grad():
            if self.source == "ddi":
                graph1_embed = self.model.get_graph_representation(
                    CustomizedBatch(batched_data.x_s, batched_data.edge_index_s, batched_data.batch_s, batched_data.y_s,
                                    batched_data.edge_feat_s))
                graph2_embed = self.model.get_graph_representation(
                    CustomizedBatch(batched_data.x_t, batched_data.edge_index_t, batched_data.batch_t, batched_data.y_t,
                                    batched_data.edge_feat_t))
                raw_graph = torch.cat([graph1_embed, graph2_embed], dim=-1).cpu().detach().numpy()
            else:
                raw_graph = self.model.get_graph_representation(batched_data).cpu().detach().numpy()

        retrieval_indexes = []
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
            batch_retrieval_graph = Batch.from_data_list(graph_list)
            batch_retrieval_graph_list.append(batch_retrieval_graph)
        return batch_retrieval_graph_list
