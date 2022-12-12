import torch
import torch.nn as nn
import numpy as np
from modules.models.nrs.rs_base import MindNRSBase


class LDARSModel(MindNRSBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.topic_embed_path = kwargs.get("topic_embed_path", None)
        topic_embeds = torch.FloatTensor(np.load(self.topic_embed_path).transpose((1, 0)))  # load lda topic embeddings
        self.topic_embedding = nn.Embedding.from_pretrained(topic_embeds, freeze=True)
        self.embedding_dim = kwargs.get("embedding_dim", 300)
        self.final = nn.Linear(self.embedding_dim, self.embedding_dim)

    def news_encoder(self, input_feat):
        """
        input_feat["news"]: [N * H, S],
        default only use title, and body is added after title.
        Dim S contains: title(30) + body(100) + document_embed(300/768) + entity_feature(4*entity_num)
        """
        embedding = self.dropouts(self.embedding_layer(**input_feat))
        topic_weight = self.topic_embedding(input_feat["news"]).transpose(1, 2)  # (N, H, S)
        topic_vec = self.final(torch.matmul(topic_weight, embedding))  # (N, H, E)
        y = self.news_att_layer(topic_vec)
        return {"news_embed": y[0], "news_weight": y[1], "topic_weight": topic_weight}
