import torch
import torch.nn as nn


class AR(nn.Module):
    def __init__(self, p):
        super(AR, self).__init__()
        self.p = p
        self.linear_D = nn.Linear(in_features=p, out_features=1, bias=True)
        self.linear_G = nn.Linear(in_features=p, out_features=1, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_normal_(self.linear_D.weight, gain=gain)
        nn.init.xavier_normal_(self.linear_G.weight, gain=gain)

    def forward(self, record_GD):
        Ds = torch.stack([record_GD['St'][i][0] for i in range(len(record_GD['St']))], dim=-1)
        Gs = torch.stack([record_GD['St'][i][1] for i in range(len(record_GD['St']))], dim=-1)
        bs, num_nodes, p = Ds.shape
        res_D = self.linear_D(Ds).reshape(bs, num_nodes)
        res_G = self.linear_G(Gs).reshape(bs, num_nodes, num_nodes)
        del Ds
        del Gs
        return res_D, res_G
