import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence

from modules.models.general import TopicLayer, AttLayer
from modules.models.nrs.rs_base import MindNRSBase


class BATMRSModel(MindNRSBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.news_encoder_name = kwargs.pop("news_encoder_name", "base")
        self.user_encoder_name = kwargs.pop("user_encoder_name", "base")
        self.user_history_connect = kwargs.get("user_history_connect", "stack")  # limited in ["concat", "stack"]
        # concat means concatenate user history clicked news(list of words) , stack means stack user history(matrix)
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
            title_topics = self.topic_layer(news_embeddings=title, news_mask=input_feat["title_mask"])
            topic_dict = self.topic_layer(news_embeddings=body, news_mask=input_feat["body_mask"])
            topic_weight = torch.cat([title_topics["topic_weight"], topic_dict["topic_weight"]], dim=-1)
            all_vec = torch.cat([title_topics["topic_vec"], topic_dict["topic_vec"]], dim=-1)
            topic_vector, news_weight = self.topic_att(all_vec)
            news_vector = F.relu(self.topic_affine(topic_vector), inplace=True)
        else:
            if input_feat.get("run_name", None) == "history" and self.user_history_connect == "concat":
                news_shape = input_feat.get("batch_size", 64), -1, input_feat["news"].size(-1)
                input_feat["news"] = input_feat["news"].reshape(news_shape).reshape(news_shape[0], -1)
                input_feat["news_mask"] = input_feat["news_mask"].reshape(news_shape).reshape(news_shape[0], -1).bool()
            topic_dict = self.extract_topic(input_feat)
            topic_weight = topic_dict["topic_weight"]
            # add activation function
            news_vector, news_weight = self.news_att_layer(self.dropouts(topic_dict["topic_vec"]))
        out_dict = {"news_embed": news_vector, "news_weight": news_weight.squeeze(-1), "topic_weight": topic_weight}
        if self.topic_variant == "variational_topic":
            out_dict["kl_divergence"] = topic_dict["kl_divergence"]
        return out_dict

    def user_encoder(self, input_feat):
        history_news = input_feat["history_news"]
        if self.user_history_connect == "concat":
            return {"user_embed": history_news, "user_weight": None}
        if self.user_encoder_name == "gru":
            history_length = input_feat["history_length"].cpu()
            packed_y = pack_padded_sequence(history_news, history_length, batch_first=True, enforce_sorted=False)
            user_vector = self.user_encode_layer(packed_y)[1].squeeze(dim=0)
            user_weight = None
            # y = self.user_encode_layer(history_news)[0]
            # user_vector, user_weight = self.user_att_layer(y)  # additive attention layer
        elif self.user_encoder_name == "batm":
            user_weight = self.user_encode_layer(history_news).transpose(1, 2)
            # mask = input_feat["news_mask"].expand(self.head_num, y.size(0), -1).transpose(0, 1) == 0
            # user_weight = torch.softmax(user_weight.masked_fill(mask, 0), dim=-1)  # fill zero entry with zero weight
            user_vec = self.user_final(torch.matmul(user_weight, history_news))
            user_vector, user_weight = self.user_att_layer(user_vec)  # additive attention layer
        else:
            user_vector, user_weight = self.user_att_layer(history_news)  # additive attention layer
        return {"user_embed": user_vector, "user_weight": user_weight}
