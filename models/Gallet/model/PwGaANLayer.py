import sys

import dgl
import torch
import torch.nn as nn
import torch.nn.functional as F


class PwGaANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads=1, att=True, gate=True, use_pre_w=True):
        super(PwGaANLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.num_heads = num_heads

        self.use_pre_w = use_pre_w

        self.att = att
        if self.att:
            # Shared Weight W_a for AttentionNet
            self.Wa = nn.Linear(self.in_dim, self.out_dim, bias=False)
            # AttentionNet outer linear layer
            # split fc to avoid cat
            self.att_out_fc_l = nn.Linear(self.out_dim, 1, bias=False)
            self.att_out_fc_r = nn.Linear(self.out_dim, 1, bias=False)

        # Head gate layer
        self.gate = gate
        if self.gate:
            # split fc to avoid cat
            self.gate_fc_l = nn.Linear(self.in_dim, 1, bias=False)
            self.gate_fc_m = nn.Linear(self.out_dim, 1, bias=False)
            self.gate_fc_r = nn.Linear(self.in_dim, 1, bias=False)
            self.Wg = nn.Linear(self.in_dim, self.out_dim, bias=False)

        self.reset_parameters()

    def reset_parameters(self):
        """ Reinitialize learnable parameters. """
        gain = nn.init.calculate_gain('leaky_relu')
        if self.att:
            nn.init.xavier_normal_(self.Wa.weight, gain=gain)
            nn.init.xavier_normal_(self.att_out_fc_l.weight, gain=gain)
            nn.init.xavier_normal_(self.att_out_fc_r.weight, gain=gain)
        if self.gate:
            gain = nn.init.calculate_gain('sigmoid')
            nn.init.xavier_normal_(self.Wg.weight, gain=gain)
            nn.init.xavier_normal_(self.gate_fc_l.weight, gain=gain)
            nn.init.xavier_normal_(self.gate_fc_m.weight, gain=gain)
            nn.init.xavier_normal_(self.gate_fc_r.weight, gain=gain)

    def edge_attention(self, edges):
        a = self.att_out_fc_l((edges.data['pre_w'] if self.use_pre_w else 1) * edges.src['z']) + self.att_out_fc_r(edges.dst['z'])
        return {'e': F.leaky_relu(a)}

    def message_func(self, edges):
        """ Specify messages to be propagated along edges """
        # The messages will be sent to the mailbox
        # mailbox['proj_z']: z->x, so we need z's projected features
        # mailbox['e']: z->x has a e for attention calculation
        if self.gate:
            pwFeat = (edges.data['pre_w'] if self.use_pre_w else 1) * edges.src['v']
            return {'proj_z': edges.src['proj_z'], 'e': edges.data['e'], 'pre_v_g': pwFeat}
        else:
            if self.att:
                return {'proj_z': edges.src['proj_z'], 'e': edges.data['e']}
            else:
                return {'proj_z': edges.src['proj_z'], 'pre_w': edges.data['pre_w']} if self.use_pre_w else \
                    {'proj_z': edges.src['proj_z']}

    def reduce_func(self, nodes):
        """ Specify how messages are processed and propagated to nodes """
        if self.att:
            # Aggregate features to nodes
            alpha = F.softmax(nodes.mailbox['e'], dim=1)
            alpha = F.dropout(alpha, 0.1)
            h = torch.sum(alpha * nodes.mailbox['proj_z'], dim=1)
        else:
            h = torch.sum(nodes.mailbox['pre_w'] * nodes.mailbox['proj_z'], dim=1) if self.use_pre_w else \
                torch.sum(nodes.mailbox['proj_z'], dim=1)

        # head gates
        if self.gate:
            pwFeat = nodes.mailbox['pre_v_g']
            gateProj = self.Wg(pwFeat)
            maxFeat = torch.max(gateProj, dim=1)[0]
            meanFeat = torch.mean(pwFeat, dim=1)
            gFCVal = self.gate_fc_l(nodes.data['v']) + self.gate_fc_m(maxFeat) + self.gate_fc_r(meanFeat)
            gVal = torch.sigmoid(gFCVal)
            h = gVal * h

        return {'h': h}

    def forward(self, g: dgl.DGLGraph):
        with g.local_scope():
            if self.att:
                feat = g.ndata['v']

                # Wa: shared attention to features v (or h for multiple GAT layers)
                z = self.Wa(feat)
                g.ndata['z'] = z

                # AttentionNet
                g.apply_edges(self.edge_attention)

            # Message Passing
            g.update_all(self.message_func, self.reduce_func)
            res = (g.ndata['proj_z'] + g.ndata['h']).reshape(self.num_heads, int(g.batch_size / self.num_heads), -1, self.out_dim)
            return res


class MultiHeadPwGaANLayer(nn.Module):
    def __init__(self, in_dim, out_dim, num_heads, merge='cat', att=True, gate=True, use_pre_w=True):
        super(MultiHeadPwGaANLayer, self).__init__()
        self.att = att
        self.gate = gate
        if (not self.att) and self.gate:
            sys.stderr.write('[MultiHeadPwGaANLayer] Use Attention = %s but Use Gate = %s!\n' % (str(self.att), str(self.gate)))
            exit(-985)

        self.use_pre_w = use_pre_w

        self.num_heads = num_heads
        self.pwGaAN = PwGaANLayer(in_dim, out_dim, num_heads=self.num_heads, att=self.att, gate=self.gate, use_pre_w=self.use_pre_w)

        self.merge = merge

    def forward(self, g: dgl.DGLGraph):
        batch_g = dgl.batch([g for i in range(self.num_heads)])
        head_outs = self.pwGaAN(batch_g)
        del batch_g
        if self.merge == 'cat':
            return head_outs.permute(1, 2, 0, 3).reshape(head_outs.shape[-3], head_outs.shape[-2], -1)
        elif self.merge == 'mean':
            return torch.mean(head_outs, dim=0)
        else:
            return head_outs.permute(1, 2, 0, 3).reshape(head_outs.shape[-3], head_outs.shape[-2], -1)  # Default: cat
