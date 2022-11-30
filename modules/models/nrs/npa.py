import os
import torch
import torch.nn as nn

from modules.models import PersonalizedAttentivePooling
from modules.models.nrs.rs_base import MindNRSBase
from modules.utils import read_json


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
        default_path = os.path.join(kwargs.get("data_dir"), "utils", f"MIND_uid_{kwargs.get('mind_type')}.json")
        uid_path = kwargs.get("uid_path", default_path)
        if not os.path.exists(uid_path):
            raise ValueError("User ID dictionary is not found, please check your config file")
        uid2index = read_json(uid_path)
        self.user_embedding = nn.Embedding(len(uid2index) + 1, self.user_emb_dim)
        self.transform_news = nn.Linear(self.user_emb_dim, self.attention_hidden_dim)
        self.transform_user = nn.Linear(self.user_emb_dim, self.attention_hidden_dim)
        self.news_att_layer = PersonalizedAttentivePooling(self.num_filters, self.attention_hidden_dim)
        self.user_att_layer = PersonalizedAttentivePooling(self.num_filters, self.attention_hidden_dim)
        # self.news_att_layer = AttLayer(self.embedding_dim, self.attention_hidden_dim)
        # self.user_att_layer = AttLayer(self.embedding_dim, self.attention_hidden_dim)

    def text_encode(self, input_feat):
        y = self.dropouts(self.embedding_layer(input_feat))
        y = self.dropouts(self.news_encode_layer(y.transpose(1, 2)).transpose(1, 2))
        return y

    def news_encoder(self, input_feat):
        """input_feat: Size is [N * H, S]"""
        news_emb = self.text_encode(input_feat)
        user_emb = self.transform_news(self.user_embedding(input_feat["uid"]))
        y = self.news_att_layer(news_emb, user_emb)
        return {"news_embed": y[0], "news_weight": y[1]}

    def user_encoder(self, input_feat):
        news_emb = input_feat["history_news"]
        user_emb = self.transform_user(self.user_embedding(input_feat["uid"]))
        y = self.user_att_layer(news_emb, user_emb)
        # y = self.user_att_layer(news_emb)[0]  # default use last hidden output
        return {"user_embed": y[0], "user_weight": y[1]}
