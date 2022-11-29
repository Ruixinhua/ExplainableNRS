import torch
import torch.nn as nn

from modules.models.general.topics import TopicLayer
from modules.models.nrs.rs_base import MindNRSBase


class BATMRSModel(MindNRSBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.variant_name = kwargs.pop("variant_name", "base_gru")
        self.topic_layer = TopicLayer(**kwargs)
        topic_dim = self.head_num * self.head_dim
        # the structure of basic model
        if self.variant_name == "base_gru":
            self.user_encode_layer = nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        elif self.variant_name == "bi_batm":
            self.user_encode_layer = nn.Sequential(nn.Linear(self.embedding_dim, topic_dim), nn.Tanh(),
                                                   nn.Linear(topic_dim, self.head_num))
            self.user_final = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.dropout = nn.Dropout(self.dropout_rate)

    def extract_topic(self, input_feat):
        input_feat["news_embeddings"] = self.dropout(self.embedding_layer(input_feat))
        return self.topic_layer(input_feat)

    def news_encoder(self, input_feat):
        """input_feat: Size is [N * H, S]"""
        y, topic_weight = self.extract_topic(input_feat)
        y = self.dropouts(y)  # TODO dropout layer
        # add activation function
        y = self.news_att_layer(y)
        return {"news_embed": y[0], "news_weight": y[1], "topic_weight": topic_weight}

    def user_encoder(self, input_feat):
        y = input_feat["history_news"]
        if self.variant_name == "base_gru":
            y = self.user_encode_layer(y)[0]
            y = self.user_att_layer(y)  # additive attention layer
        elif self.variant_name == "bi_batm":
            user_weight = self.user_encode_layer(y).transpose(1, 2)
            # mask = input_feat["news_mask"].expand(self.head_num, y.size(0), -1).transpose(0, 1) == 0
            # user_weight = torch.softmax(user_weight.masked_fill(mask, 0), dim=-1)  # fill zero entry with zero weight
            user_vec = self.user_final(torch.matmul(user_weight, y))
            y = self.user_att_layer(user_vec)  # additive attention layer
        elif self.variant_name == "base_att":
            y = self.user_att_layer(y)  # additive attention layer
        return {"user_embed": y[0], "user_weight": y[1]}
