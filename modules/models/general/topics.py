import torch
import numpy as np
import torch.nn.functional as F
from torch import nn

from modules.models.general.layers import activation_layer


class TopicLayer(nn.Module):
    def __init__(self, **kwargs):
        super(TopicLayer, self).__init__()
        self.variant_name = kwargs.get("topic_variant", "base")
        self.head_num, self.head_dim = kwargs.get("head_num", 50), kwargs.get("head_dim", 20)
        topic_dim = self.head_num * self.head_dim
        self.embedding_dim = kwargs.get("embedding_dim", 300)
        self.hidden_dim = kwargs.get("hidden_dim", 256)
        self.word_dict = kwargs.get("word_dict", None)
        self.evaluate_topic = kwargs.get("evaluate_topic", False)
        self.final = nn.Linear(self.embedding_dim, self.embedding_dim)
        self.act_layer = activation_layer(kwargs.get("act_layer", "tanh"))  # default use tanh
        if self.variant_name == "base":  # default using two linear layers
            self.topic_layer = nn.Sequential(nn.Linear(self.embedding_dim, topic_dim), self.act_layer,
                                             nn.Linear(topic_dim, self.head_num))
        elif self.variant_name == "base_adv":  # advanced base topic model
            self.topic_layer = nn.Sequential(
                nn.Linear(self.embedding_dim, self.head_num * self.hidden_dim), self.act_layer,  # map to hidden dim
                nn.Linear(self.head_num * self.hidden_dim, topic_dim), self.act_layer,  # map to topic dim
                nn.Linear(topic_dim, self.head_num), activation_layer("sigmoid"))  # map to topic num
        elif self.variant_name == "base_gate":  # add a gate to the topic layer
            self.topic_layer = nn.Sequential(nn.Linear(self.embedding_dim, topic_dim), self.act_layer,
                                             nn.Linear(topic_dim, self.head_num))
            # self.gate_layer = nn.Linear(self.head_num, self.head_num)
            self.gate_layer = nn.Linear(np.sum(kwargs.get("news_lengths", [100])).squeeze().astype(np.int), 1)
            self.gate_type = kwargs.get("gate_type", "close")
        elif self.variant_name == "base_topic_vector":
            self.topic_layer = nn.Linear(self.embedding_dim, self.head_num, bias=False)
        elif self.variant_name == "topic_embed":
            self.rho = nn.Linear(self.embedding_dim, len(self.word_dict), bias=False)
            self.topic_layer = nn.Linear(self.embedding_dim, self.head_num)
            self.topic_embed_path = kwargs.get("topic_embed_path", None)
            if self.topic_embed_path is not None:
                self.freeze_topic = kwargs.get("freeze_topic", True)
                topic_embeds = torch.FloatTensor(np.load(self.topic_embed_path))
                self.topic_layer = self.topic_layer.from_pretrained(topic_embeds, freeze=self.freeze_topic)
        elif self.variant_name == "topic_matrix":
            self.transform = nn.Sequential(nn.Linear(self.embedding_dim, self.head_dim), self.act_layer)
            self.topic_matrix = nn.Embedding(self.head_num, self.head_dim)
        elif self.variant_name == "variational_topic":
            self.topic_layer = nn.Sequential(nn.Linear(self.embedding_dim, topic_dim, bias=True), self.act_layer,
                                             nn.Linear(topic_dim, self.head_num, bias=True))
            self.logsigma_q_theta = nn.Sequential(nn.Linear(self.embedding_dim, topic_dim, bias=True), self.act_layer,
                                                  nn.Linear(topic_dim, self.head_num, bias=True))
        else:
            raise ValueError("Specify correct variant name!")

    def forward(self, news_embeddings, news_mask, **kwargs):
        """
        Topic forward pass, return topic vector and topic weights
        """
        mask = news_mask.expand(self.head_num, news_embeddings.size(0), -1).transpose(0, 1) == 0
        out_dict = {}
        if self.variant_name == "topic_embed":
            # topic_weight = self.topic_layer(kwargs.get("news")).transpose(1, 2)  # (N, H, S)
            weights = F.softmax(self.topic_layer(self.rho.weight), dim=0)
            topic_weight = weights[kwargs.get("news")].transpose(1, 2)  # (N, H, S)
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
        elif self.variant_name == "base_gate":
            topic_weight = self.topic_layer(news_embeddings).transpose(1, 2)  # (N, H, S)
            if not kwargs.get("evaluate_topic", False):
                # topic_entropy = torch.sum(-topic_weight * torch.log2(1e-9 + topic_weight), dim=-1)
                topic_reg = torch.sigmoid(self.gate_layer(topic_weight))
                if self.gate_type == "close":
                    topic_reg = torch.where(topic_reg > 0.5, 1.0, 0.0)
                topic_weight = topic_reg * topic_weight
            topic_weight = torch.softmax(topic_weight.masked_fill(mask, -1e4), dim=-1).masked_fill(mask, 0)
        elif self.variant_name == "base_topic_vector":
            topic_weight = (news_embeddings @ self.topic_layer.weight.transpose(0, 1)).transpose(1, 2)  # (N, H, S)
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
