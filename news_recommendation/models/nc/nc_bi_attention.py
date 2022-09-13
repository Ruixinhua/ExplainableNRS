import torch
import torch.nn as nn

from news_recommendation.models import TopicLayer, AttLayer
from news_recommendation.models.nc.nc_models import BaseClassifyModel


class BiAttentionClassifyModel(BaseClassifyModel):
    def __init__(self, **kwargs):
        super(BiAttentionClassifyModel, self).__init__(**kwargs)
        self.topic_layer = TopicLayer(**kwargs)
        # the structure of basic model
        self.final = nn.Linear(self.embed_dim, self.embed_dim)
        self.projection = AttLayer(self.embed_dim, 128)
        self.entropy_constraint = kwargs.get("entropy_constraint", False)
        self.calculate_entropy = kwargs.get("calculate_entropy", False)

    def extract_topic(self, input_feat):
        input_feat["news_embeddings"] = self.dropout(self.embedding_layer(input_feat))
        return self.topic_layer(input_feat)

    def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
        input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
        topic_vec, topic_weight = self.extract_topic(input_feat)
        doc_embedding, doc_topic = self.projection(topic_vec)  # (N, E), (N, H)
        output = self.classify_layer(doc_embedding, topic_weight, return_attention=return_attention)
        if self.entropy_constraint or self.calculate_entropy:
            entropy_sum = torch.sum(-topic_weight * torch.log(1e-6 + topic_weight)).squeeze() / self.head_num
            output = output + (entropy_sum,)
        return output
