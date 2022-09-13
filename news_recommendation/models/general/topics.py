from typing import Dict

import torch
import numpy as np
from torch import nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

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
        elif self.variant_name == "topic_embed":
            self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
            self.topic_embed_path = kwargs.get("topic_embed_path", None)
            if self.topic_embed_path is not None:
                self.freeze_topic = kwargs.get("freeze_topic", True)
                topic_embeds = torch.FloatTensor(np.load(self.topic_embed_path))
                self.topic_layer = self.topic_layer.from_pretrained(topic_embeds, freeze=self.freeze_topic)
        else:
            raise ValueError("Specify correct variant name!")
        if self.variant_name == "gru" or self.variant_name == "combined_gru":
            self.gru = nn.GRU(self.embed_dim, self.embed_dim, 2, batch_first=True)
        if self.variant_name == "weight_mha":
            head_dim = self.embed_dim // 12
            self.sentence_encoder = MultiHeadedAttention(12, head_dim, self.embed_dim)
        if self.variant_name == "combined_mha":
            self.query = nn.Linear(self.embed_dim, topic_dim)
            self.key = nn.Linear(self.embed_dim, topic_dim)

    def run_gru(self, embedding, length):
        try:
            embedding = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
        except RuntimeError:
            raise RuntimeError()
        y, _ = self.gru(embedding)  # extract interest from history behavior
        y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
        return y

    def forward(self, input_feat: Dict[str, torch.Tensor]) -> (torch.Tensor, torch.Tensor):
        """
        Topic forward pass, return topic vector and topic weights
        :param input_feat: Dict[str, torch.Tensor], input feature contains "news" or "news_embeddings"
        :return:
        """
        embedding = input_feat["news_embeddings"]
        if self.variant_name == "topic_embed":
            topic_weight = self.topic_layer(input_feat["news"]).transpose(1, 2)  # (N, H, S)
        else:
            topic_weight = self.topic_layer(embedding).transpose(1, 2)  # (N, H, S)
            # expand mask to the same size as topic weights
            mask = input_feat["news_mask"].expand(self.head_num, embedding.size(0), -1).transpose(0, 1) == 0
            # fill zero entry with -INF when doing softmax and fill in zeros after that
            topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e4), dim=-1).masked_fill(mask, 0)
            # topic_weight = torch.softmax(topic_weight, dim=1).masked_fill(mask, 0)  # external attention
            # topic_weight = topic_weight / torch.sum(topic_weight, dim=-1, keepdim=True)
        if self.variant_name == "combined_mha":
            # context_vec = torch.matmul(topic_weight, embedding)  # (N, H, E)
            query, key = [linear(embedding).view(embedding.size(0), -1, self.head_num, self.head_dim).transpose(1, 2)
                          for linear in (self.query, self.key)]
            # topic_vec, _ = self.mha(context_vec, context_vec, context_vec)  # (N, H, H*D)
            scores = torch.matmul(query, key.transpose(-2, -1)) / self.head_num ** 0.5  # (N, H, S, S)
            context_weight = torch.mean(scores, dim=-1)  # (N, H, S)
            topic_weight = context_weight * topic_weight  # (N, H, S)
        elif self.variant_name == "combined_gru":
            length = torch.sum(input_feat["mask"], dim=-1)
            embedding = self.run_gru(embedding, length)
        elif self.variant_name == "weight_mha":
            embedding = self.sentence_encoder(embedding, embedding, embedding)[0]
        topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, E)
        return topic_vec, topic_weight
