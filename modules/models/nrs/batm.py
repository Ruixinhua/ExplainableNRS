import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.models.general import TopicLayer, AttLayer
from modules.models.nrs.rs_base import MindNRSBase


class BATMRSModel(MindNRSBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.news_encoder_name = kwargs.pop("news_encoder_name", "base")
        self.user_encoder_name = kwargs.pop("user_encoder_name", "base")
        self.topic_layer = TopicLayer(**kwargs)
        topic_dim = self.head_num * self.head_dim
        # the structure of basic model
        if self.user_encoder_name == "gru":
            self.user_encode_layer = nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True)
        elif self.user_encoder_name == "batm":
            self.user_encode_layer = nn.Sequential(nn.Linear(self.embedding_dim, topic_dim), nn.Tanh(),
                                                   nn.Linear(topic_dim, self.head_num))
            self.user_final = nn.Linear(self.embedding_dim, self.embedding_dim)
        if self.news_encoder_name == "multi_view":
            self.topic_att = AttLayer(self.embedding_dim * 2, self.attention_hidden_dim)
            self.topic_affine = nn.Linear(self.embedding_dim * 2, self.embedding_dim)
            # self.multi_att = AttLayer(self.embedding_dim, self.attention_hidden_dim)

    def extract_topic(self, input_feat):
        input_feat["news_embeddings"] = self.dropouts(self.embedding_layer(**input_feat))
        return self.topic_layer(**input_feat)

    def news_encoder(self, input_feat):
        """input_feat: Size is [N * H, S]"""
        if self.news_encoder_name == "multi_view":
            title = self.dropouts(self.embedding_layer(news=input_feat["title"], news_mask=input_feat["title_mask"]))
            body = self.dropouts(self.embedding_layer(news=input_feat["body"], news_mask=input_feat["body_mask"]))
            title_vector, title_topics = self.topic_layer(news_embeddings=title, news_mask=input_feat["title_mask"])
            body_vector, body_topics = self.topic_layer(news_embeddings=body, news_mask=input_feat["body_mask"])
            topic_weight = torch.cat([title_topics, body_topics], dim=-1)
            topic_vector, news_weight = self.topic_att(torch.cat([title_vector, body_vector], dim=-1))
            news_vector = F.relu(self.topic_affine(topic_vector), inplace=True)
        else:
            topic_vector, topic_weight = self.extract_topic(input_feat)
            # add activation function
            news_vector, news_weight = self.news_att_layer(self.dropouts(topic_vector))
        return {"news_embed": news_vector, "news_weight": news_weight.squeeze(-1), "topic_weight": topic_weight}

    def user_encoder(self, input_feat):
        history_news = input_feat["history_news"]
        if self.user_encoder_name == "gru":
            y = self.user_encode_layer(history_news)[0]
            user_vector, user_weight = self.user_att_layer(y)  # additive attention layer
        elif self.user_encoder_name == "batm":
            user_weight = self.user_encode_layer(history_news).transpose(1, 2)
            # mask = input_feat["news_mask"].expand(self.head_num, y.size(0), -1).transpose(0, 1) == 0
            # user_weight = torch.softmax(user_weight.masked_fill(mask, 0), dim=-1)  # fill zero entry with zero weight
            user_vec = self.user_final(torch.matmul(user_weight, history_news))
            user_vector, user_weight = self.user_att_layer(user_vec)  # additive attention layer
        else:
            user_vector, user_weight = self.user_att_layer(history_news)  # additive attention layer
        return {"user_embed": user_vector, "user_weight": user_weight}
