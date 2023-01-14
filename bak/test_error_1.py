import torch
import torch.nn as nn


class test_error(nn.Module):

    def __init__(self):
        super(test_error, self).__init__()

        self.a = torch.nn.Embedding(10, 5)

    def forward(self, x):
        h = self.a(x)  # [B, 5]
        attention = torch.nn.Softmax(dim=-1)(h)
        for batch_i in range(h.size(0)):
            h[batch_i][0] = attention[0][0] * h[batch_i][0]
        return h


criterion = nn.CrossEntropyLoss()

model = test_error().cuda()
inpt = torch.LongTensor(range(10)).unsqueeze(0).cuda()
label = torch.LongTensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 4]).cuda()
opt = model(inpt).squeeze(0)
# add the sigma
# opt[0][1] = opt[0][1] + 0.1
loss = criterion(opt, label)
loss.backward()
