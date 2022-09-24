from typing import Dict

import torch
import numpy as np
from torch import nn

from news_recommendation.models.general.layers import MultiHeadedAttention


class TopicLayer(nn.Module):
    def __init__(self, **kwargs):
        super(TopicLayer, self).__init__()
        self.variant_name = kwargs.get("topic_variant", "base")
        self.head_num, self.head_dim = kwargs.get("head_num", 50), kwargs.get("head_dim", 20)
        topic_dim = self.head_num * self.head_dim
        self.embedding_dim = kwargs.get("embedding_dim", 300)
        self.word_dict = kwargs.get("word_dict", None)
        self.final = nn.Linear(self.embedding_dim, self.embedding_dim)
        if self.variant_name == "base":  # default using two linear layers
            self.topic_layer = nn.Sequential(nn.Linear(self.embedding_dim, topic_dim), nn.Tanh(),
                                             nn.Linear(topic_dim, self.head_num))
        elif self.variant_name == "raw":  # a simpler model using only one linear layer to encode topic weights
            self.topic_layer = nn.Linear(self.embedding_dim, self.head_num)  # can not focus on specific words
        elif self.variant_name == "topic_embed":
            self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
            self.topic_embed_path = kwargs.get("topic_embed_path", None)
            if self.topic_embed_path is not None:
                self.freeze_topic = kwargs.get("freeze_topic", True)
                topic_embeds = torch.FloatTensor(np.load(self.topic_embed_path))
                self.topic_layer = self.topic_layer.from_pretrained(topic_embeds, freeze=self.freeze_topic)
        elif self.variant_name == "MHA":
            self.sentence_encoder = MultiHeadedAttention(self.head_num, self.head_dim, self.embedding_dim)
            self.token_layer = kwargs.get("token_layer", "distribute_topic")
            if self.token_layer == "distribute_topic":
                self.token_weight = nn.Linear(self.head_dim * self.head_num, self.head_num)
            else:
                self.token_weight = nn.Linear(self.head_dim, 1)
        else:
            raise ValueError("Specify correct variant name!")

    def mha_topics(self, topic_vector, input_feat):
        if self.token_layer == "distribute_topic":
            topic_vector = self.token_weight(topic_vector)  # (N, S, H)
            topic_vector = topic_vector.transpose(1, 2)     # (N, H, S)
        else:
            topic_vector = topic_vector.view(-1, topic_vector.shape[1], self.head_num, self.head_dim)
            topic_vector = topic_vector.transpose(1, 2)  # (N, H, S, D)
            topic_vector = self.token_weight(topic_vector).squeeze(-1)  # (N, H, S)
        mask = input_feat["news_mask"].expand(self.head_num, topic_vector.size(0), -1).transpose(0, 1) == 0
        topic_weight = topic_vector.masked_fill(mask, -1e4)
        topic_weight = torch.softmax(topic_weight, dim=-1)  # (N, H, S)
        return topic_weight

    def forward(self, input_feat: Dict[str, torch.Tensor]) -> (torch.Tensor, torch.Tensor):
        """
        Topic forward pass, return topic vector and topic weights
        :param input_feat: Dict[str, torch.Tensor], input feature contains "news" or "news_embeddings"
        :return:
        """
        embedding = input_feat["news_embeddings"]
        if self.variant_name == "topic_embed":
            topic_weight = self.topic_layer(input_feat["news"]).transpose(1, 2)  # (N, H, S)
        elif self.variant_name == "MHA":
            hidden_score, _ = self.sentence_encoder(embedding, embedding, embedding)
            topic_weight = self.mha_topics(hidden_score, input_feat)
        else:
            topic_weight = self.topic_layer(embedding).transpose(1, 2)  # (N, H, S)
            # expand mask to the same size as topic weights
            mask = input_feat["news_mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
            # fill zero entry with -INF when doing softmax and fill in zeros after that
            topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e4), dim=-1).masked_fill(mask, 0)
            # topic_weight = torch.softmax(topic_weight, dim=1).masked_fill(mask, 0)  # external attention
            # topic_weight = topic_weight / torch.sum(topic_weight, dim=-1, keepdim=True)
        topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, E)
        return topic_vec, topic_weight
