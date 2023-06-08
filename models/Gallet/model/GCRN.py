import torch
import torch.nn as nn
import torch.nn.functional as F

from .SpatAttLayer import SpatAttLayer


class GCRN(nn.Module):
    def __init__(self, num_nodes=361, hidden_dim=16):
        super(GCRN, self).__init__()

        self.num_nodes = num_nodes
        self.hidden_dim = hidden_dim

        self.num_dim = 1

        self.spat_embed_dim = int((self.num_dim + 1) * self.hidden_dim)     # Embedding dimension after spatial feature extraction
        self.temp_embed_dim = self.spat_embed_dim   # Embedding dimension after temporal feature extraction
        self.tran_embed_dim = self.temp_embed_dim   # Embedding dimension after transition projection

        # Spatial Attention Layer
        self.spatLayer_D = SpatAttLayer(feat_dim=1, hidden_dim=self.hidden_dim, num_heads=1, att=False, gate=False, merge='mean', num_dim=self.num_dim, cat_orig=True, use_pre_w=False)
        self.spatLayer_G = SpatAttLayer(feat_dim=self.num_nodes, hidden_dim=self.hidden_dim, num_heads=1, att=False, gate=False, merge='mean', num_dim=self.num_dim, cat_orig=True, use_pre_w=False)

        # Temporal Layer (GRU)
        self.tempLayer_D = nn.LSTM(input_size=self.spat_embed_dim, hidden_size=self.temp_embed_dim)
        self.tempLayer_G = nn.LSTM(input_size=self.spat_embed_dim, hidden_size=self.temp_embed_dim)
        self.bn_D = nn.BatchNorm1d(num_features=self.temp_embed_dim)
        self.bn_G = nn.BatchNorm1d(num_features=self.temp_embed_dim)

        # Transfer Function
        self.tran_d_l = nn.Linear(in_features=self.temp_embed_dim, out_features=1, bias=True)
        self.tran_g_l = nn.Linear(in_features=self.temp_embed_dim, out_features=self.tran_embed_dim, bias=True)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('linear')
        nn.init.xavier_normal_(self.tran_d_l.weight, gain=gain)
        nn.init.xavier_normal_(self.tran_g_l.weight, gain=gain)

    def forward(self, record_GCRN: list):
        # Extract spatial features
        for i in range(len(record_GCRN)):
            record_GCRN[i][-1].ndata['v'] = record_GCRN[i][-1].ndata['d']
        spat_embed_t_D = torch.stack([self.spatLayer_D(list(gs)) for gs in record_GCRN])

        for i in range(len(record_GCRN)):
            record_GCRN[i][-1].ndata['v'] = record_GCRN[i][-1].ndata['g']
        spat_embed_t_G = torch.stack([self.spatLayer_G(list(gs)) for gs in record_GCRN])

        if spat_embed_t_D.device.type == 'cuda':
            torch.cuda.empty_cache()

        num_records = spat_embed_t_D.shape[0]

        # Extract temporal features
        o_D, (h_D, c_D) = self.tempLayer_D(spat_embed_t_D.reshape(num_records, -1, self.spat_embed_dim))
        temp_embed_t_D = h_D.reshape(-1, self.num_nodes, self.temp_embed_dim) + torch.mean(spat_embed_t_D, dim=0)
        norm_temp_embed_t_D = self.bn_D(torch.transpose(temp_embed_t_D, -2, -1))
        temp_embed_t_D = torch.transpose(norm_temp_embed_t_D, -2, -1)
        del spat_embed_t_D
        del o_D
        del h_D
        del c_D
        del norm_temp_embed_t_D

        o_G, (h_G, c_G) = self.tempLayer_G(spat_embed_t_G.reshape(num_records, -1, self.spat_embed_dim))
        temp_embed_t_G = h_G.reshape(-1, self.num_nodes, self.temp_embed_dim) + torch.mean(spat_embed_t_G, dim=0)
        norm_temp_embed_t_G = self.bn_G(torch.transpose(temp_embed_t_G, -2, -1))
        temp_embed_t_G = torch.transpose(norm_temp_embed_t_G, -2, -1)
        del spat_embed_t_G
        del o_G
        del h_G
        del c_G
        del norm_temp_embed_t_G

        if temp_embed_t_D.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Transfer D
        res_D = self.tran_d_l(temp_embed_t_D).reshape(-1, self.num_nodes)
        del temp_embed_t_D

        # Transfer G
        reshaped_temp_embed_t_G = temp_embed_t_G.reshape(-1, self.num_nodes, 1, self.temp_embed_dim)
        gl = reshaped_temp_embed_t_G.repeat(1, 1, self.num_nodes, 1)
        gr = torch.transpose(reshaped_temp_embed_t_G, -3, -2).repeat(1, self.num_nodes, 1, 1)
        res_G = torch.sum(self.tran_g_l(gl) * gr, dim=-1)
        del temp_embed_t_G
        del reshaped_temp_embed_t_G
        del gl
        del gr

        return res_D, res_G

