import torch
from torch.nn import functional as F 
import torch_geometric
from models.utils.ddi_util import CustomizedBatch


class GraphRetrieval(torch.nn.Module):
    def __init__(self, gnn, adapter, source, dataset, classifier=None):
        super().__init__()
        self.source = source
        self.dataset = dataset 
        self.gnn = gnn
        self.classifier = classifier
        for param in self.gnn.parameters():
            param.requires_grad = False
        self.adapter = adapter 

    def forward(self, batched_data, batch_retrieval_graph_list, device=None):
        if self.source == "ddi":
            graph1_embed = self.gnn.get_graph_representation(
                CustomizedBatch(batched_data.x_s, batched_data.edge_index_s, batched_data.batch_s, batched_data.y_s, batched_data.edge_feat_s))
            graph2_embed = self.gnn.get_graph_representation(
                CustomizedBatch(batched_data.x_t, batched_data.edge_index_t, batched_data.batch_t, batched_data.y_t, batched_data.edge_feat_t))
            graph_embeddings = torch.cat([graph1_embed, graph2_embed], dim=-1).clone()
            g_label = torch.nn.Softmax(dim=1)(self.classifier(graph1_embed, graph2_embed))  # b * n_class
        else:
            graph_embeddings = self.gnn.get_graph_representation(batched_data)
            if self.source == "ogb": # sigmoid
                 g_label = torch.sigmoid(self.gnn.graph_pred_linear(graph_embeddings)) # b * n_class
            elif self.source == "lsc":
                 g_label = self.gnn.graph_pred_linear(graph_embeddings)# b * n_class
            else:
                 g_label = torch.nn.Softmax(dim=1)(self.gnn.graph_pred_linear(graph_embeddings))# b * n_class

        r_labels = []
        H = []
        H.append(graph_embeddings)
        for batch_retrieval_graph in batch_retrieval_graph_list:
            if self.source == "ogb": # sigmoid
                r_labels.append(batch_retrieval_graph.y.detach())
            elif self.source == "lsc":
                r_labels.append(batch_retrieval_graph.y.unsqueeze(1).detach())
            elif self.source == "ddi":
                r_labels.append(batch_retrieval_graph.y_s.detach())
            else:
                r_labels.append(batch_retrieval_graph.y.detach())
            if self.source == "ddi":
                graph1_embed = self.gnn.get_graph_representation(
                    CustomizedBatch(batched_data.x_s, batched_data.edge_index_s, batched_data.batch_s, batched_data.y_s,
                                    batched_data.edge_feat_s))
                graph2_embed = self.gnn.get_graph_representation(
                    CustomizedBatch(batched_data.x_t, batched_data.edge_index_t, batched_data.batch_t, batched_data.y_t,
                                    batched_data.edge_feat_t))
                h_retrieval_graph = torch.cat([graph1_embed, graph2_embed], dim=-1)
            else:
                h_retrieval_graph = self.gnn.get_graph_representation(batch_retrieval_graph)

            H.append(h_retrieval_graph)

        concat_H = torch.cat(H, -1).view(graph_embeddings.size(0), -1, graph_embeddings.size(1)).contiguous() # bs x (k+1) x d_model
        scores = self.adapter(graph_embeddings, concat_H)

        labels = torch.stack(r_labels, dim=1) # bx(k)
        attention = F.softmax(scores, 1)
        adjusted_label = torch.zeros(g_label.size()).to(device)
        for batch_i in range(g_label.size(0)):
            if g_label.size(1) == 1: #sigmoid
                 if attention.size(1) > 2:
                     adjusted_label[batch_i][0] = attention[batch_i][0] * g_label[batch_i][0] + attention[batch_i][1:] @ labels[batch_i].float().squeeze()
                 else:
                     adjusted_label[batch_i][0] = attention[batch_i][0] * g_label[batch_i][0] + attention[batch_i][1:] * labels[batch_i].float().squeeze()

            elif len(labels[batch_i]) > len(batch_retrieval_graph_list)+1:
                 number_label = g_label.size(1)
                 k_plus = len(labels[batch_i]) // number_label
                 adjusted_label[batch_i] = attention[batch_i][0] * g_label[batch_i]

                 for k in range(0, k_plus):
                     class_i = 0
                     for j in range(k*number_label, k * number_label + number_label):
                          if not labels[batch_i][j].isnan():
                             g_label[batch_i][class_i] = adjusted_label[batch_i][class_i] + attention[batch_i][k+1] * labels[batch_i][j]
                          class_i += 1
            else:
                adjusted_label[batch_i] = attention[batch_i][0] * g_label[batch_i]
                for class_i in range(g_label.size(1)):
                    for k in range(labels.size(1)):
                        if labels[batch_i][k] == class_i:
                            adjusted_label[batch_i][class_i] = adjusted_label[batch_i][class_i] + attention[batch_i][k]
        return adjusted_label
