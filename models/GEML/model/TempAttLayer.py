import torch.nn as nn

from .ScaledDotProductAttention import ScaledDotProductAttention

from Config import TEMP_FEAT_NAMES


class TempAttLayer(nn.Module):
    def __init__(self, query_dim, embed_dim, rec_merge='sum', comb_merge='sum'):
        super(TempAttLayer, self).__init__()
        self.query_dim = query_dim
        self.embed_dim = embed_dim
        self.rec_merge = rec_merge      # merging method for historical record attention
        self.comb_merge = comb_merge    # merging method for combination attention

        # Scaled Dot Product Attention
        self.recScaledDotProductAttention = ScaledDotProductAttention(self.query_dim, self.embed_dim, merge=self.rec_merge, num_channel=len(TEMP_FEAT_NAMES))
        self.combScaledDotProductAttention = ScaledDotProductAttention(self.query_dim, self.embed_dim, merge=self.comb_merge, num_channel=len(TEMP_FEAT_NAMES))

    def forward(self, query_feat, embed_feat_dict):
        rec_embed_list = [self.recScaledDotProductAttention(query_feat, embed_feat_dict[temp_feat]) for temp_feat in TEMP_FEAT_NAMES]
        comb_embed = self.combScaledDotProductAttention(query_feat, rec_embed_list)
        del rec_embed_list
        return comb_embed
