import torch
import torch.nn as nn

from Config import TEMP_FEAT_NAMES


class TempRecurrentLayer(nn.Module):
    def __init__(self, embed_dim, merge='mean'):
        super(TempRecurrentLayer, self).__init__()

        self.embed_dim = embed_dim

        self.recurrentBlocks = nn.ModuleList([
            RecurrentBlock(embed_dim=self.embed_dim)
            for _ in range(len(TEMP_FEAT_NAMES))
        ])

        self.merge = merge

        if self.merge == 'mean':
            self.bn = nn.BatchNorm1d(num_features=self.embed_dim)
        elif self.merge == 'cat':
            self.bn = nn.BatchNorm1d(num_features=self.embed_dim * len(TEMP_FEAT_NAMES))
        else:
            self.bn = nn.BatchNorm1d(num_features=self.embed_dim)

    def forward(self, embed_feat_dict):
        rec_embed_list = [self.recurrentBlocks[temp_feat_i](embed_feat_dict[TEMP_FEAT_NAMES[temp_feat_i]])
                          for temp_feat_i in range(len(TEMP_FEAT_NAMES))]

        # Combine
        if self.merge == 'mean':
            output = sum(rec_embed_list) / len(rec_embed_list)
        elif self.merge == 'cat':
            output = torch.cat(rec_embed_list, dim=-1)
        else:
            output = sum(rec_embed_list) / len(rec_embed_list)  # Default: mean

        norm_output = self.bn(torch.transpose(output, -2, -1))
        output = torch.transpose(norm_output, -2, -1)
        del rec_embed_list
        del norm_output

        return output


class RecurrentBlock(nn.Module):
    def __init__(self, embed_dim):
        super(RecurrentBlock, self).__init__()

        self.embed_dim = embed_dim

        self.blk_module = nn.LSTM(input_size=self.embed_dim, hidden_size=self.embed_dim)

    def forward(self, record_list: list):
        records = torch.stack(record_list)
        num_records, num_nodes = records.shape[0], records.shape[-2]
        o, (h, c) = self.blk_module(records.reshape(num_records, -1, self.embed_dim))
        res = h.reshape(-1, num_nodes, self.embed_dim)
        del records
        del o
        del h
        del c

        return res

