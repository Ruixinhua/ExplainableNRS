import torch
import torch.nn as nn
from news_recommendation.base.base_model import BaseModel
from news_recommendation.utils.graph_untils import construct_entity_embedding, construct_adj


class KGAT(BaseModel):
    def __init__(self, **kwargs):
        super(KGAT, self).__init__()
        self.entity_neighbor_num = kwargs.get("entity_neighbor_num", 20)
        self.freeze_embedding = kwargs.get("freeze_embedding", True)
        self.entity_adj, self.relation_adj = construct_adj(**kwargs)
        entity_embedding, relation_embedding = construct_entity_embedding(**kwargs)
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding.cuda())
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding.cuda(), freeze=self.freeze_embedding)
        entity_embedding_dim, layer_dim = kwargs.get("entity_embedding_dim", 100), kwargs.get("layer_dim", 128)
        self.attention_layer = nn.Sequential(
            nn.Linear(3 * entity_embedding_dim, layer_dim), nn.ReLU(inplace=True),  # first layer
            nn.Linear(layer_dim, 1), nn.Softmax(dim=-1),  # second layer: acquire weight
        )
        self.convolve_layer = nn.Sequential(
            nn.Linear(2 * entity_embedding_dim, entity_embedding_dim), nn.ReLU(inplace=True)
        )

    def aggregate(self, entity_embedding, neighbor_embedding):
        concat_embedding = torch.cat([entity_embedding, neighbor_embedding], len(entity_embedding.shape)-1)
        aggregate_embedding = self.convolve_layer(concat_embedding)
        return aggregate_embedding

    def forward(self, entity_ids):
        neighbor_entities, neighbor_relations = self.entity_adj[entity_ids.cpu()], self.relation_adj[entity_ids.cpu()]

        neighbor_entity_embedding = self.entity_embedding(torch.tensor(neighbor_entities).cuda())  # (B, EN, NE, E)
        relation_embedding = self.relation_embedding(torch.tensor(neighbor_relations).cuda())  # same as above
        entity_embedding = self.entity_embedding(entity_ids)  # shape (B, EN, E)
        entity_embedding_expand = torch.unsqueeze(entity_embedding, 2).expand(  # shape (B, EN, 1, E)
            entity_embedding.shape[0], entity_embedding.shape[1], self.entity_neighbor_num, entity_embedding.shape[2]
        )
        # (B, EN, NE, 3*E)
        embedding_concat = torch.cat([entity_embedding_expand, neighbor_entity_embedding, relation_embedding], 3)
        neighbor_att_embedding = torch.sum(self.attention_layer(embedding_concat) * neighbor_entity_embedding, dim=2)
        kgat_embedding = self.aggregate(entity_embedding, neighbor_att_embedding)
        return kgat_embedding
