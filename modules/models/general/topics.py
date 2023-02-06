import torch
import numpy as np
from torch import nn

from modules.models.general.layers import MultiHeadedAttention, activation_layer


class TopicLayer(nn.Module):
    def __init__(self, **kwargs):
        super(TopicLayer, self).__init__()
        self.variant_name = kwargs.get("topic_variant", "base")
        self.head_num, self.head_dim = kwargs.get("head_num", 50), kwargs.get("head_dim", 20)
        topic_dim = self.head_num * self.head_dim
        self.embedding_dim = kwargs.get("embedding_dim", 300)
        self.hidden_dim = kwargs.get("hidden_dim", 100)
        self.word_dict = kwargs.get("word_dict", None)
        self.final = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.act_layer = activation_layer(kwargs.get("act_layer", "tanh"))  # default use tanh
        if self.variant_name == "base":  # default using two linear layers
            self.topic_layer = nn.Sequential(nn.Linear(self.embedding_dim, topic_dim), self.act_layer,
                                             nn.Linear(topic_dim, self.head_num))
        elif self.variant_name == "raw":  # a simpler model using only one linear layer to encode topic weights
            # can not extract any meaningful topics from documents
            self.topic_layer = nn.Sequential(nn.Linear(self.embedding_dim, self.head_num), self.act_layer)
        elif self.variant_name == "add_dense":  # add a dense layer to the topic layer
            self.topic_layer = nn.Sequential(nn.Linear(self.embedding_dim, self.hidden_dim), self.act_layer,
                                             nn.Linear(self.hidden_dim, self.head_num))
        elif self.variant_name == "topic_embed":
            self.topic_layer = nn.Embedding(len(self.word_dict), self.head_num)
            self.topic_embed_path = kwargs.get("topic_embed_path", None)
            if self.topic_embed_path is not None:
                self.freeze_topic = kwargs.get("freeze_topic", True)
                topic_embeds = torch.FloatTensor(np.load(self.topic_embed_path))
                self.topic_layer = self.topic_layer.from_pretrained(topic_embeds, freeze=self.freeze_topic)
        elif self.variant_name == "topic_matrix":
            self.transform = nn.Sequential(nn.Linear(self.embedding_dim, self.head_dim), self.act_layer)
            self.topic_matrix = nn.Embedding(self.head_num, self.head_dim)
        elif self.variant_name == "topic_dense":
            self.topic_layer = nn.Sequential(nn.Linear(self.embedding_dim, self.head_num), self.act_layer)
        elif self.variant_name == "MHA":
            self.sentence_encoder = MultiHeadedAttention(self.head_num, self.head_dim, self.embedding_dim)
            self.token_layer = kwargs.get("token_layer", "distribute_topic")
            if self.token_layer == "distribute_topic":
                self.token_weight = nn.Linear(self.head_dim * self.head_num, self.head_num)
            else:
                self.token_weight = nn.Linear(self.head_dim, 1)
        elif self.variant_name == "variational_topic":
            self.topic_layer = nn.Sequential(nn.Linear(self.embedding_dim, topic_dim, bias=True), self.act_layer,
                                             nn.Linear(topic_dim, self.head_num, bias=True))
            self.logsigma_q_theta = nn.Sequential(nn.Linear(self.embedding_dim, topic_dim, bias=True), self.act_layer,
                                                  nn.Linear(topic_dim, self.head_num, bias=True))
        else:
            raise ValueError("Specify correct variant name!")

    def mha_topics(self, topic_vector, mask):
        if self.token_layer == "distribute_topic":
            topic_vector = self.token_weight(topic_vector)  # (N, S, H)
            topic_vector = topic_vector.transpose(1, 2)     # (N, H, S)
        else:
            topic_vector = topic_vector.view(-1, topic_vector.shape[1], self.head_num, self.head_dim)
            topic_vector = topic_vector.transpose(1, 2)  # (N, H, S, D)
            topic_vector = self.token_weight(topic_vector).squeeze(-1)  # (N, H, S)
        topic_weight = topic_vector.masked_fill(mask, -1e4)
        topic_weight = torch.softmax(topic_weight, dim=-1)  # (N, H, S)
        return topic_weight

    def forward(self, news_embeddings, news_mask, **kwargs):
        """
        Topic forward pass, return topic vector and topic weights
        """
        mask = news_mask.expand(self.head_num, news_embeddings.size(0), -1).transpose(0, 1) == 0
        out_dict = {}
        if self.variant_name == "topic_embed":
            topic_weight = self.topic_layer(kwargs.get("news")).transpose(1, 2)  # (N, H, S)
        elif self.variant_name == "MHA":
            hidden_score, _ = self.sentence_encoder(news_embeddings, news_embeddings, news_embeddings)
            topic_weight = self.mha_topics(hidden_score, mask)
        elif self.variant_name == "topic_matrix":
            embeds_trans = self.transform(news_embeddings)  # (N, S, D)
            topic_vector = self.topic_matrix.weight.unsqueeze(dim=0).transpose(1, 2)  # (1, D, H)
            topic_weight = embeds_trans @ topic_vector  # (N, S, H)
            topic_weight = topic_weight.transpose(1, 2)  # (N, H, S)
            topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e4), dim=-1).masked_fill(mask, 0)
        elif self.variant_name == "variational_topic":
            topic_weight = self.topic_layer(news_embeddings)
            log_q_theta = self.logsigma_q_theta(news_embeddings)
            kl_divergence = -0.5 * torch.sum(1 + log_q_theta - topic_weight.pow(2) - log_q_theta.exp(), dim=-1).mean()
            out_dict["kl_divergence"] = kl_divergence
            if self.training:  # reparameterization topic weight in training
                std = torch.exp(0.5 * log_q_theta)
                eps = torch.randn_like(std)
                topic_weight = eps.mul_(std).add_(topic_weight)
            topic_weight = topic_weight.transpose(1, 2)
            topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e4), dim=-1).masked_fill(mask, 0)
        else:
            topic_weight = self.topic_layer(news_embeddings).transpose(1, 2)  # (N, H, S)
            # expand mask to the same size as topic weights
            # fill zero entry with -INF when doing softmax and fill in zeros after that
            topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e4), dim=-1).masked_fill(mask, 0)
            # topic_weight = torch.softmax(topic_weight, dim=1).masked_fill(mask, 0)  # external attention
            # topic_weight = topic_weight / torch.sum(topic_weight, dim=-1, keepdim=True)
        topic_vec = self.final(torch.matmul(topic_weight, news_embeddings))  # (N, H, E)
        out_dict.update({"topic_vec": topic_vec, "topic_weight": topic_weight})
        return out_dict
