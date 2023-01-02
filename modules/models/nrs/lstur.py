import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from modules.models.general import AttLayer, Conv1D
from modules.models.nrs.rs_base import MindNRSBase
from modules.utils import read_json


class LSTURRSModel(MindNRSBase):
    """
    Implementation of LSTRU model
    Ref: An, Mingxiao et al. “Neural News Recommendation with Long- and Short-term User Representations.” ACL (2019).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.masking_probability = 1.0 - kwargs.get("long_term_masking_probability", 0.1)
        self.use_category, self.num_filters = kwargs.get("use_category", False), kwargs.get("num_filters", 300)
        self.window_size = kwargs.get("window_size", 3)
        self.user_embed_method = kwargs.get("user_embed_method", "concat")
        padding = (self.window_size - 1) // 2
        assert 2 * padding == self.window_size - 1, "Kernel size must be an odd number"
        self.news_encode_layer = Conv1D(self.embedding_dim, self.num_filters, self.window_size)
        self.news_att_layer = AttLayer(self.num_filters, self.attention_hidden_dim)  # output size is [N, num_filters]
        news_dim = self.num_filters
        user_dim = kwargs.get("user_dim", 100)
        if self.use_category:
            news_dim = self.num_filters + self.category_dim * 2
        if self.user_embed_method == "init" or self.user_embed_method == "concat":
            default_path = os.path.join(kwargs.get("data_dir"), "utils", f"MIND_uid_{kwargs.get('mind_type')}.json")
            uid_path = kwargs.get("uid_path", default_path)
            if not os.path.exists(uid_path):
                raise ValueError("User ID dictionary is not found, please check your config file")
            uid2index = read_json(uid_path)
            self.user_embedding = nn.Embedding(len(uid2index) + 1, user_dim)  # count from 1
            self.user_affine = nn.Linear(user_dim, news_dim)
        self.user_encode_layer = nn.GRU(news_dim, news_dim, batch_first=True, bidirectional=False)
        if self.user_embed_method == "concat":
            self.user_affine = None
            self.transform_layer = nn.Linear(news_dim+user_dim, news_dim)
        self.user_att_layer = None  # no attentive layer for LSTUR model

    def news_encoder(self, input_feat):
        """input_feat: Size is [N * H, S]"""
        # 1. worod embedding
        embed = self.dropouts(self.embedding_layer(**input_feat))
        # 2. CNN
        features = self.news_encode_layer(embed.transpose(1, 2)).transpose(1, 2)
        # 3. attention
        vector, weight = self.news_att_layer(self.dropouts(features))
        if self.use_category:
            category_vector = self.category_embedding(input_feat["category"])
            subvert_vector = self.subvert_embedding(input_feat["subvert"])
            vector = torch.cat([vector, self.dropouts(category_vector), self.dropouts(subvert_vector)], dim=1)
        return {"news_embed": vector, "news_weight": weight}

    def user_encoder(self, input_feat):
        history_news, user_ids = input_feat["history_news"], input_feat["uid"]
        history_length = input_feat["history_length"].cpu()
        packed_y = pack_padded_sequence(history_news, history_length, batch_first=True, enforce_sorted=False)
        if self.user_embed_method == "init":
            user_embed = F.relu(self.user_affine(self.user_embedding(user_ids)), inplace=True)
            _, user_vector = self.user_encode_layer(packed_y, user_embed.unsqueeze(dim=0))
            user_vector = user_vector.squeeze(dim=0)
        elif self.user_embed_method == "concat":
            user_embed = self.user_embedding(user_ids)
            last_hidden = self.user_encode_layer(packed_y)[1].squeeze(dim=0)
            user_vector = F.relu(self.transform_layer(torch.cat((last_hidden, user_embed), dim=1)), inplace=True)
        else:  # default use last hidden output from GRU network
            user_vector = self.user_encode_layer(packed_y)[1].squeeze(dim=0)
        return {"user_embed": user_vector}
