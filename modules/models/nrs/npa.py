import os
import torch.nn.functional as F
import torch.nn as nn

from modules.models import PersonalizedAttentivePooling
from modules.models.nrs.rs_base import MindNRSBase
from modules.utils import read_json, get_default_upath


class NPARSModel(MindNRSBase):
    """
    Implementation of NPM model
    Wu, Chuhan et al. “NPA: Neural News Recommendation with Personalized Attention.”
    Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (2019): n. pag.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category_num, self.num_filters = kwargs.get("category_num", 300), kwargs.get("num_filters", 300)
        self.user_emb_dim, self.window_size = kwargs.get("user_emb_dim", 100), kwargs.get("window_size", 3)
        self.use_uid = kwargs.get("use_uid", True)
        padding = (self.window_size - 1) // 2
        assert 2 * padding == self.window_size - 1, "Kernel size must be an odd number"
        self.news_encode_layer = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.num_filters, self.window_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        uid2index = read_json(get_default_upath(**kwargs))
        self.user_embedding = nn.Embedding(len(uid2index) + 1, self.user_emb_dim)
        self.transform_news = nn.Linear(self.user_emb_dim, self.attention_hidden_dim)
        self.transform_user = nn.Linear(self.user_emb_dim, self.attention_hidden_dim)
        self.news_att_layer = PersonalizedAttentivePooling(self.num_filters, self.attention_hidden_dim)
        self.user_att_layer = PersonalizedAttentivePooling(self.num_filters, self.attention_hidden_dim)

    def text_encode(self, input_feat):
        news_vector = self.dropouts(self.embedding_layer(**input_feat))
        news_vector = self.dropouts(self.news_encode_layer(news_vector.transpose(1, 2)).transpose(1, 2))
        return news_vector

    def news_encoder(self, input_feat):
        """input_feat: Size is [N * H, S]"""
        news_emb = self.text_encode(input_feat)
        user_emb = F.relu(self.transform_news(self.user_embedding(input_feat["uid"])), inplace=True)
        news_vector, news_weight = self.news_att_layer(news_emb, user_emb)
        return {"news_embed": news_vector, "news_weight": news_weight}

    def user_encoder(self, input_feat):
        news_emb = input_feat["history_news"]
        user_emb = F.relu(self.transform_user(self.user_embedding(input_feat["uid"])), inplace=True)
        user_vector, user_weight = self.user_att_layer(news_emb, user_emb)
        return {"user_embed": user_vector, "user_weight": user_weight}
