from audioop import bias
import torch.nn as nn
import torch
from torch.nn import init 

class Attention(nn.Module):

    def __init__(self, d_model, k, dropout=0.1):
        super().__init__()

        self.d_model = d_model
        self.q_linear = nn.Linear(d_model, d_model, bias=False)
        self.k_linear = nn.Linear(d_model, d_model, bias=False)
        self.bias = nn.Parameter(torch.Tensor(k+1))
        self.bias.data.zero_()


    def forward(self, q, k):
        bs = q.size(0)
        q = self.q_linear(q).view(bs, -1, self.d_model)
        k = self.k_linear(k).view(bs, -1, self.d_model)
        scores = torch.matmul(q, k.transpose(-2, -1)) + self.bias 

        # size: bs * 1 * 1
        return scores.squeeze(1)




