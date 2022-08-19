import torch 
import torch.nn as nn 
from torch.nn import functional as F 

class GraphRetrieval(torch.nn.Module):
    def __init__(self, gnn, adapter, source, dataset):
        super().__init__()
        self.source = source
        self.dataset = dataset 
        self.gnn = gnn 
        for param in self.gnn.parameters():
            param.requires_grad = False
        self.adapter = adapter 

    def forward(self, batched_data, batch_retrieval_graph_list):
        graph_embeddings = self.gnn.get_graph_representation(batched_data)
        #g_scores = self.adapter(graph_embeddings, graph_embeddings)  # bs * 1
        if self.source == "ogb" and graph_embeddings.size(-1) == 1: # sigmoid
             g_label = torch.sigmoid(self.gnn.graph_pred_linear(graph_embeddings)) # b * n_class
        elif self.source == "lsc":
             g_label = self.gnn.graph_pred_linear(graph_embeddings)# b * n_class
        else:
             g_label = torch.nn.Softmax(dim=1)(self.gnn.graph_pred_linear(graph_embeddings))# b * n_class

        #r_scores = []
        r_labels = [g_label]
        H = []
        H.append(graph_embeddings)
        for batch_retrieval_graph in batch_retrieval_graph_list:
            if self.source == "ogb" and graph_embeddings.size(-1) == 1: # sigmoid
                r_labels.append(batch_retrieval_graph.y.detach())
            elif self.source == "lsc":
                r_labels.append(batch_retrieval_graph.y.unsqueeze(1).detach())
            else:
                tmp_labels = g_label.new(g_label.shape).zero_()
                for i in range(tmp_labels.size(0)):
                    try:
                      tmp_labels[i, batch_retrieval_graph.y[i]] = 1.
                    except:
                        pass # skip nan
                r_labels.append(tmp_labels)
            h_retrieval_graph = self.gnn.get_graph_representation(batch_retrieval_graph)
            #r_score = self.adapter(graph_embeddings, h_retrieval_graph)
            H.append(h_retrieval_graph)
            #r_scores.append(r_score)
        concat_H = torch.cat(H, -1).view(graph_embeddings.size(0), -1, graph_embeddings.size(1)).contiguous() # bs x (k+1) x d_model
        scores = self.adapter(graph_embeddings, concat_H)

        # r_scores = torch.cat(r_scores, 1)  # bs * k
        # scores = torch.cat((g_scores, r_scores), 1)
        labels = torch.cat(r_labels, -1) # bx(k+1)
        attention = F.softmax(scores, 1) 
        labels = labels.view(attention.size(0), attention.size(1), -1)
        attention = attention.view(attention.size(0), -1, 1)
        attention = attention.repeat(1,1, labels.size(-1)).transpose(-2,-1)

        output = attention @ labels 
        output = output.sum(dim=-2)
        return output
