import sys

import numpy as np
import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F

from .PwGaANLayer import MultiHeadPwGaANLayer


class SpatAttLayer(nn.Module):
    def __init__(self, feat_dim, hidden_dim, num_heads, att=True, gate=True, merge='mean', num_dim=3, cat_orig=True, use_pre_w=True):
        super(SpatAttLayer, self).__init__()
        self.feat_dim = feat_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads

        self.att = att
        self.gate = gate
        self.merge = merge

        self.num_dim = num_dim

        self.use_pre_w = use_pre_w

        self.dimSpatAttLayers = nn.ModuleList([
            MultiHeadPwGaANLayer(self.feat_dim, self.hidden_dim, self.num_heads,
                                 merge=self.merge, att=self.att, gate=self.gate, use_pre_w=self.use_pre_w)
            for _ in range(self.num_dim)
        ])

        self.proj_fc = nn.Linear(self.feat_dim, self.hidden_dim, bias=False)

        self.cat_orig = cat_orig
        orig_flag = 1 if self.cat_orig else 0

        # BatchNorm
        self.bn = nn.BatchNorm1d(num_features=int(self.hidden_dim * (self.num_dim + orig_flag)))
        if self.merge == 'mean':
            self.bn = nn.BatchNorm1d(num_features=int(self.hidden_dim * (self.num_dim + orig_flag)))
        elif self.merge == 'cat':
            self.bn = nn.BatchNorm1d(num_features=int(self.hidden_dim * (self.num_dim * self.num_heads + orig_flag)))

        self.reset_parameters()

    def reset_parameters(self):
        """ Reinitialize learnable parameters. """
        gain = nn.init.calculate_gain('leaky_relu')
        nn.init.xavier_normal_(self.proj_fc.weight, gain=gain)

    def forward(self, gs: list):
        if len(gs) != self.num_dim:
            sys.stderr.write('input has %d graphs but %d are set initially.\n' % (len(gs), self.num_dim))
            exit(-27)

        feat = gs[-1].ndata['v']
        feat = F.dropout(feat, 0.1)
        for i in range(len(gs)):
            gs[i].ndata['v'] = feat

        proj_feat = self.proj_fc(feat)
        del feat

        for i in range(len(gs)):
            gs[i].ndata['proj_z'] = proj_feat

        out_proj_feat = proj_feat.reshape(gs[-1].batch_size, -1, self.hidden_dim)
        del proj_feat

        hs = [self.dimSpatAttLayers[i](gs[i]) for i in range(len(gs))]

        h = torch.cat(([out_proj_feat] if self.cat_orig else []) + hs, dim=-1)
        del out_proj_feat
        del hs

        normH = self.bn(torch.transpose(h, -2, -1))
        reshapedH = torch.transpose(normH, -2, -1)
        del h
        del normH

        return reshapedH


if __name__ == '__main__':
    """ Test: Remove dot in the package importing to avoid errors """
    GDVQ = np.load('test/GDVQ.npy', allow_pickle=True).item()
    V = GDVQ['V']
    (dfg, dbg,), _ = dgl.load_graphs('test/FBGraphs.dgl')
    (dgg,), _ = dgl.load_graphs('test/GeoGraph.dgl')
    V = torch.from_numpy(V)

    spatAttLayer = SpatAttLayer(feat_dim=43, hidden_dim=16, num_heads=3, gate=True, num_dim=3, cat_orig=True, use_pre_w=True)
    print(V, V.shape)
    dfg.ndata['v'] = V
    dbg.ndata['v'] = V
    dgg.ndata['v'] = V
    out = spatAttLayer([dfg, dbg, dgg])
    print(out, out.shape)
    test = out.detach().numpy()
