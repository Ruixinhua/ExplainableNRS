import os
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence

from modules.models.general import AttLayer
from modules.models.nrs.rs_base import MindNRSBase
from modules.utils import read_json


class LSTURRSModel(MindNRSBase):
    """
    Implementation of LSTRU model
    Ref: An, Mingxiao et al. “Neural News Recommendation with Long- and Short-term User Representations.” ACL (2019).
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category_num, self.num_filters = kwargs.get("category_num", 300), kwargs.get("num_filters", 300)
        self.use_category, self.use_sub = kwargs.get("use_category", 0), kwargs.get("use_subcategory", 0)
        self.user_embed_method = kwargs.get("user_embed_method", "concat")
        padding = (self.window_size - 1) // 2
        assert 2 * padding == self.window_size - 1, "Kernel size must be an odd number"
        self.news_encode_layer = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.num_filters, self.window_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        self.news_att_layer = AttLayer(self.num_filters, self.attention_hidden_dim)  # output size is [N, num_filters]
        if self.use_category and self.use_sub:
            self.category_embedding = nn.Embedding(self.category_num, self.num_filters)
        news_dim = self.num_filters * 3 if self.use_category and self.use_sub else self.num_filters  # dim of news
        if self.user_embed_method == "init" or self.user_embed_method == "concat":
            default_path = os.path.join(kwargs.get("data_dir"), "utils", f"MIND_uid_{kwargs.get('mind_type')}.json")
            uid_path = kwargs.get("uid_path", default_path)
            if not os.path.exists(uid_path):
                raise ValueError("User ID dictionary is not found, please check your config file")
            uid2index = read_json(uid_path)
            self.user_embedding = nn.Embedding(len(uid2index) + 1, news_dim)  # count from 1
        self.user_encode_layer = nn.GRU(news_dim, news_dim, batch_first=True, bidirectional=False)
        if self.user_embed_method == "concat":
            self.transform_layer = nn.Linear(news_dim * 2, news_dim)
        self.user_att_layer = None  # no attentive layer for LSTUR model

    def text_encode(self, input_feat):
        y = self.dropouts(self.embedding_layer(input_feat))
        y = self.news_encode_layer(y.transpose(1, 2)).transpose(1, 2)
        return self.news_att_layer(self.dropouts(y))

    def news_encoder(self, input_feat):
        """input_feat: Size is [N * H, S]"""
        news_embed, weight = self.text_encode(input_feat)
        if self.use_category or self.use_sub:  # TODO: optimize input format
            news, cat = self.load_news_feat(input_feat, use_category=True)
            input_feat["news"] = news
            cat_embed = self.category_embedding(cat)
            news_embed = torch.cat([torch.reshape(cat_embed, (cat_embed.shape[0], -1)), news_embed], dim=1)
        return {"news_embed": news_embed, "news_weight": weight}

    def user_encoder(self, input_feat):
        y, user_ids = input_feat["history_news"], input_feat["uid"]
        packed_y = pack_padded_sequence(y, input_feat["history_length"].cpu(), batch_first=True, enforce_sorted=False)
        if self.user_embed_method == "init":
            user_embed = self.user_embedding(user_ids)
            _, y = self.user_encode_layer(packed_y, user_embed.unsqueeze(dim=0))
            y = y.squeeze(dim=0)
        elif self.user_embed_method == "concat":
            user_embed = self.user_embedding(user_ids)
            _, y = self.user_encode_layer(packed_y)
            y = self.transform_layer(torch.cat((y.squeeze(dim=0), user_embed), dim=1))
        else:  # default use last hidden output from GRU network
            y = self.user_encode_layer(packed_y)[1].squeeze(dim=0)
        return y
