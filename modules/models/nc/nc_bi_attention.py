import torch
import torch.nn as nn

from modules.models import TopicLayer, AttLayer
from modules.models.nc.nc_models import BaseClassifyModel


class BiAttentionClassifyModel(BaseClassifyModel):
    def __init__(self, **kwargs):
        super(BiAttentionClassifyModel, self).__init__(**kwargs)
        self.topic_layer = TopicLayer(**kwargs)
        # the structure of basic model
        self.final = nn.Linear(self.embed_dim, self.embed_dim)
        self.projection = AttLayer(self.embed_dim, 128)
        self.entropy_constraint = kwargs.get("with_entropy", False)
        self.calculate_entropy = kwargs.get("calculate_entropy", False)

    def extract_topic(self, input_feat):
        input_feat["news_embeddings"] = self.dropout(self.embedding_layer(**input_feat))
        return self.topic_layer(input_feat)

    def forward(self, input_feat, inputs_embeds=None, return_attention=False, **kwargs):
        input_feat["embedding"] = input_feat.get("embedding", inputs_embeds)
        topic_dict = self.extract_topic(input_feat)
        topic_weight = topic_dict["topic_weight"]
        doc_embedding, doc_topic = self.projection(topic_dict["topic_vec"])  # (N, E), (N, H)
        output = self.classify_layer(doc_embedding, topic_weight, return_attention=return_attention)
        if self.entropy_constraint or self.calculate_entropy:
            entropy_sum = torch.sum(-topic_weight * torch.log2(1e-9 + topic_weight)).squeeze() / self.head_num
            output["entropy"] = entropy_sum
        return output
