import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class ScaledDotProductAttention(nn.Module):
    def __init__(self, query_dim, embed_dim, merge='sum', num_channel=4):
        super(ScaledDotProductAttention, self).__init__()
        self.query_dim = query_dim
        self.embed_dim = embed_dim
        self.merge = merge
        self.num_channel = num_channel    # for cat

        self.rooted_embed_dim = math.sqrt(self.embed_dim)

        # Query, Key, Value weights
        self.Wq = nn.Linear(self.query_dim, self.embed_dim, bias=False)
        self.Wk = nn.Linear(self.embed_dim, self.embed_dim, bias=False)
        self.Wv = nn.Linear(self.embed_dim, self.embed_dim, bias=False)

        # BatchNorm
        self.bn = nn.BatchNorm1d(num_features=self.embed_dim)
        if self.merge == 'mean' or self.merge == 'sum':
            self.bn = nn.BatchNorm1d(num_features=self.embed_dim)
        elif self.merge == 'cat':
            self.bn = nn.BatchNorm1d(num_features=self.embed_dim * self.num_channel)

        self.reset_parameters()

    def reset_parameters(self):
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_normal_(self.Wq.weight, gain=gain)
        nn.init.xavier_normal_(self.Wk.weight, gain=gain)
        nn.init.xavier_normal_(self.Wv.weight, gain=gain)

    def apply_scaled_dot_product_attention(self, query_feat, embedding_feat):
        proj_Q = self.Wq(query_feat)
        proj_K = self.Wk(embedding_feat)
        proj_V = self.Wv(embedding_feat)

        # Note that batch size is the first dimension, while the last two dimensions are the ones we care about.
        scores = torch.matmul(proj_Q, torch.transpose(proj_K, -2, -1)) / self.rooted_embed_dim
        del proj_Q
        del proj_K
        norm_scores = F.softmax(scores, dim=-1)
        norm_scores = F.dropout(norm_scores, 0.1)
        del scores

        output = torch.matmul(norm_scores, proj_V)
        del norm_scores
        del proj_V

        return output + embedding_feat

    def forward(self, query_feat, embed_feat_list):
        embed_outputs = [self.apply_scaled_dot_product_attention(query_feat, embed_feat) for embed_feat in embed_feat_list]
        output = sum(embed_outputs)
        if self.merge == 'sum':
            output = sum(embed_outputs)
        elif self.merge == 'mean':
            output = sum(embed_outputs) / len(embed_outputs)
        elif self.merge == 'cat':
            output = torch.cat(embed_outputs, dim=-1)
        else:   # Default: sum, as Gallat
            output = sum(embed_outputs)

        normOut = self.bn(torch.transpose(output, -2, -1))
        reshapedOut = torch.transpose(normOut, -2, -1)
        del output
        del normOut

        return reshapedOut
