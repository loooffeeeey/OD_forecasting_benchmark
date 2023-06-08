import torch
import torch.nn as nn

from .SpatAttLayer import SpatAttLayer
from .TempAttLayer import TempAttLayer
from .TempRecurrentLayer import TempRecurrentLayer
from .TranAttLayer import TranAttLayer
from HistoricalAverage import avgRec

from Config import TEMP_FEAT_NAMES, REF_EXTENT, HA_FEAT_DEFAULT, GALLATEXT_TEMP_USE_ATT


class GallatExt(nn.Module):
    def __init__(self, feat_dim=43, query_dim=41, hidden_dim=16, num_heads=3, num_dim=3, tune=True, ref_AR=None):
        super(GallatExt, self).__init__()
        self.num_heads = num_heads

        self.feat_dim = feat_dim
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim

        self.num_dim = num_dim

        self.spat_embed_dim = int((self.num_dim + 1) * self.hidden_dim)     # Embedding dimension after spatial feature extraction
        self.temp_embed_dim = self.spat_embed_dim   # Embedding dimension after temporal feature extraction

        # Reference-based Tuning Block
        self.tune = tune
        self.ref_AR = ref_AR

        # Spatial Attention Layer
        self.spatAttLayer = SpatAttLayer(feat_dim=self.feat_dim, hidden_dim=self.hidden_dim, num_heads=self.num_heads, gate=True, merge='mean', num_dim=self.num_dim, cat_orig=True, use_pre_w=True)

        # Temporal Attention Layer
        self.tempLayer = TempAttLayer(query_dim=self.query_dim, embed_dim=self.spat_embed_dim, rec_merge='mean', comb_merge='mean') \
            if GALLATEXT_TEMP_USE_ATT else TempRecurrentLayer(embed_dim=self.spat_embed_dim, merge='mean')

        # Transferring Attention Layer
        self.tranAttLayer = TranAttLayer(embed_dim=self.temp_embed_dim)

    def forward(self, record, record_GD, query, predict_G=False, ref_extent=REF_EXTENT):
        # Calculate reference
        if self.tune:
            if self.ref_AR:
                self.ref_AR.eval()
                ref_D, ref_G = self.ref_AR(record_GD)
            else:
                ref_D, ref_G = avgRec(record_GD, scheme=HA_FEAT_DEFAULT)
        else:
            ref_D, ref_G = (None, None)

        # Extract spatial features
        spat_embed_dict = {}
        for temp_feat in TEMP_FEAT_NAMES:
            spat_embed_dict[temp_feat] = [self.spatAttLayer(list(gs)) for gs in record[temp_feat]]

        if query.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Extract temporal features
        temp_embed = self.tempLayer(query, spat_embed_dict) \
            if GALLATEXT_TEMP_USE_ATT else self.tempLayer(spat_embed_dict)

        if query.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Transferring features to perform predictions
        res = self.tranAttLayer(temp_embed, predict_G, ref_D, ref_G, ref_extent)

        return res


class GallatExtFull(nn.Module):
    def __init__(self, feat_dim=43, query_dim=41, hidden_dim=16, num_heads=3, num_dim=3, tune=True, ref_AR=None):
        super(GallatExtFull, self).__init__()
        self.num_heads = num_heads

        self.feat_dim = feat_dim
        self.query_dim = query_dim
        self.hidden_dim = hidden_dim

        self.num_dim = num_dim

        self.spat_embed_dim = int((self.num_dim * self.num_heads + 1) * self.hidden_dim)    # Embedding dimension after spatial feature extraction
        self.temp_embed_dim = int(4 * self.spat_embed_dim)    # Embedding dimension after temporal feature extraction

        # Reference-based Tuning Block
        self.tune = tune
        self.ref_AR = ref_AR

        # Spatial Attention Layer
        self.spatAttLayer = SpatAttLayer(feat_dim=self.feat_dim, hidden_dim=self.hidden_dim, num_heads=self.num_heads, gate=True, merge='cat', num_dim=self.num_dim, cat_orig=True, use_pre_w=True)

        # Temporal Attention Layer
        self.tempLayer = TempAttLayer(query_dim=self.query_dim, embed_dim=self.spat_embed_dim, rec_merge='mean', comb_merge='cat') \
            if GALLATEXT_TEMP_USE_ATT else TempRecurrentLayer(embed_dim=self.spat_embed_dim, merge='cat')

        # Transferring Attention Layer
        self.tranAttLayer = TranAttLayer(embed_dim=self.temp_embed_dim)

    def forward(self, record, record_GD, query, predict_G=False, ref_extent=REF_EXTENT):
        # Calculate reference
        if self.tune:
            if self.ref_AR:
                self.ref_AR.eval()
                ref_D, ref_G = self.ref_AR(record_GD)
            else:
                ref_D, ref_G = avgRec(record_GD, scheme=HA_FEAT_DEFAULT)
        else:
            ref_D, ref_G = (None, None)

        # Extract spatial features
        spat_embed_dict = {}
        for temp_feat in TEMP_FEAT_NAMES:
            spat_embed_dict[temp_feat] = [self.spatAttLayer(list(gs)) for gs in record[temp_feat]]

        if query.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Extract temporal features
        temp_embed = self.tempLayer(query, spat_embed_dict) \
            if GALLATEXT_TEMP_USE_ATT else self.tempLayer(spat_embed_dict)

        if query.device.type == 'cuda':
            torch.cuda.empty_cache()

        # Transferring features to perform predictions
        res = self.tranAttLayer(temp_embed, predict_G, ref_D, ref_G, ref_extent)

        return res
