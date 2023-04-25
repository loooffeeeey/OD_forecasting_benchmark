import torch
import torch.nn as nn
import torch.nn.functional as F

import Config


class LSTNet(nn.Module):
    def __init__(self, p, refAR):
        super(LSTNet, self).__init__()
        self.p = p
        self.refAR = refAR

        self.L = 2 * p
        self.combDim = self.p + 1

        # stConv
        self.l_stConv_last_D = nn.Linear(in_features=1, out_features=1, bias=True)
        self.l_stConv_current_D = nn.Linear(in_features=1, out_features=1, bias=True)
        self.l_stConv_last_G = nn.Linear(in_features=1, out_features=1, bias=True)
        self.l_stConv_current_G = nn.Linear(in_features=1, out_features=1, bias=True)

        self.bn_stConv_D = nn.BatchNorm1d(num_features=1)
        self.bn_stConv_G = nn.BatchNorm1d(num_features=1)

        # GRU
        self.gru_D = nn.GRU(1, 1)
        self.gru_G = nn.GRU(1, 1)

        self.bn_gru_D = nn.BatchNorm1d(num_features=1)
        self.bn_gru_G = nn.BatchNorm1d(num_features=1)

        # Att
        self.l_att_l_D = nn.Linear(in_features=1, out_features=1, bias=False)
        self.l_att_r_D = nn.Linear(in_features=1, out_features=1, bias=False)
        self.l_att_l_G = nn.Linear(in_features=1, out_features=1, bias=False)
        self.l_att_r_G = nn.Linear(in_features=1, out_features=1, bias=False)

        self.l_att_comb_D = nn.Linear(in_features=self.combDim, out_features=1, bias=True)
        self.l_att_comb_G = nn.Linear(in_features=self.combDim, out_features=1, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_normal_(self.l_stConv_last_D.weight, gain=gain)
        nn.init.xavier_normal_(self.l_stConv_current_D.weight, gain=gain)
        nn.init.xavier_normal_(self.l_stConv_last_G.weight, gain=gain)
        nn.init.xavier_normal_(self.l_stConv_current_G.weight, gain=gain)

        nn.init.xavier_normal_(self.l_att_comb_D.weight, gain=gain)
        nn.init.xavier_normal_(self.l_att_comb_G.weight, gain=gain)

        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_normal_(self.l_att_l_D.weight, gain=gain)
        nn.init.xavier_normal_(self.l_att_r_D.weight, gain=gain)
        nn.init.xavier_normal_(self.l_att_l_G.weight, gain=gain)
        nn.init.xavier_normal_(self.l_att_r_G.weight, gain=gain)

    def forward(self, recordGD):
        padDs = torch.stack(
            [torch.zeros(recordGD[Config.LSTNET_TEMP_FEAT][0][0].shape, device=recordGD[Config.LSTNET_TEMP_FEAT][0][0].device)] +
            [recordGD[Config.LSTNET_TEMP_FEAT][i][0] for i in range(len(recordGD[Config.LSTNET_TEMP_FEAT]))]
        )
        padDs = padDs.reshape(padDs.shape + (1,))

        padGs = torch.stack(
            [torch.zeros(recordGD[Config.LSTNET_TEMP_FEAT][0][1].shape, device=recordGD[Config.LSTNET_TEMP_FEAT][0][1].device)] +
            [recordGD[Config.LSTNET_TEMP_FEAT][i][1] for i in range(len(recordGD[Config.LSTNET_TEMP_FEAT]))]
        )
        padGs = padGs.reshape(padGs.shape + (1,))

        # stConv
        stConvD = self.l_stConv_last_D(padDs[:self.L]) + self.l_stConv_current_D(padDs[1:])
        stConvG = self.l_stConv_last_G(padGs[:self.L]) + self.l_stConv_current_G(padGs[1:])
        del padDs
        del padGs

        bs, num_nodes = stConvD.shape[-3], stConvD.shape[-2]

        # stConvBatchNorm
        stConvD = self.bn_stConv_D(stConvD.reshape(-1, 1)).reshape(self.L, bs, num_nodes, 1)
        stConvG = self.bn_stConv_G(stConvG.reshape(-1, 1)).reshape(self.L, bs, num_nodes, num_nodes, 1)

        # GRU
        reshapeStConvD = stConvD.reshape(self.L, -1, 1)
        del stConvD
        tempO_D, tempH_D = self.gru_D(reshapeStConvD)
        tempO_D = tempO_D[-self.p:]

        reshapeStConvG = stConvG.reshape(self.L, -1, 1)
        del stConvG
        tempO_G, tempH_G = self.gru_G(reshapeStConvG)
        tempO_G = tempO_G[-self.p:]

        del reshapeStConvD
        del reshapeStConvG
        del tempH_D
        del tempH_G

        # GRUBatchNorm
        tempO_D = self.bn_gru_D(tempO_D.reshape(-1, 1)).reshape(self.p, -1, 1)
        tempO_G = self.bn_gru_G(tempO_G.reshape(-1, 1)).reshape(self.p, -1, 1)

        # Att: left - each, right - last
        lastO_D = torch.stack([tempO_D[-1] for _ in range(self.p)])
        alphaD = F.leaky_relu(self.l_att_l_D(tempO_D) + self.l_att_r_D(lastO_D))
        weightedD = alphaD * tempO_D
        del lastO_D
        del alphaD
        stackD = torch.stack([weightedD[i].reshape(-1) for i in range(self.p)] + [tempO_D[-1].reshape(-1)], dim=-1)
        res_D = self.l_att_comb_D(stackD)
        del tempO_D
        del weightedD
        del stackD
        res_D = res_D.reshape(bs, num_nodes)

        lastO_G = torch.stack([tempO_G[-1] for _ in range(self.p)])
        alphaG = F.leaky_relu(self.l_att_l_G(tempO_G) + self.l_att_r_G(lastO_G))
        weightedG = alphaG * tempO_G
        del lastO_G
        del alphaG
        stackG = torch.stack([weightedG[i].reshape(-1) for i in range(self.p)] + [tempO_G[-1].reshape(-1)], dim=-1)
        res_G = self.l_att_comb_G(stackG)
        del tempO_G
        del weightedG
        del stackG
        res_G = res_G.reshape(bs, num_nodes, num_nodes)

        # Evaluate refAR
        self.refAR.eval()
        resAR_D, resAR_G = self.refAR(recordGD)

        res_D += resAR_D
        res_G += resAR_G
        del resAR_D
        del resAR_G

        return res_D, res_G
