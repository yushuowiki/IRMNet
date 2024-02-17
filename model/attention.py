import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, hidden_size, bias=False)
        )

    def forward(self, z):
        # z_list=[]
        # for j in range(z.size(0)):
        #     z1=z[j]
        #     w = self.project(z1)
        #     print(w.shape)
        #     beta = torch.softmax(w, dim=1)
        #     za=(beta * z1).sum(1)
        #     print(za.shape)
        #     z_list.append(za)
        # z_out=torch.cat(z_list,dim=0)
        # return z_out
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        # return (beta * z).permute(0, 2, 1).sum(1)
        return (beta * z).sum(1)