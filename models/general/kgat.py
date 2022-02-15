import torch
import torch.nn as nn
from base.base_model import BaseModel


class KGAT(BaseModel):

    def __init__(self, entity_embedding, relation_embedding, adj_entity, adj_relation, **kwargs):
        super(KGAT, self).__init__()
        self.entity_neighbor_num = kwargs.get("entity_neighbor_num", 20)
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding.cuda())
        self.relation_embedding = nn.Embedding.from_pretrained(relation_embedding.cuda())
        entity_embedding_dim, layer_dim = kwargs.get("entity_embedding_dim", 100), kwargs.get("layer_dim", 128)
        self.attention_layer1 = nn.Linear(3 * entity_embedding_dim, layer_dim)
        self.attention_layer2 = nn.Linear(layer_dim, 1)
        self.softmax = nn.Softmax(dim=-1)
        self.relu = nn.ReLU(inplace=True)
        self.convolve_layer = nn.Linear(2 * entity_embedding_dim, entity_embedding_dim)

    def aggregate(self, entity_embedding, neighbor_embedding):
        concat_embedding = torch.cat([entity_embedding, neighbor_embedding], len(entity_embedding.shape)-1)
        aggregate_embedding = self.relu(self.convolve_layer(concat_embedding))
        return aggregate_embedding

    def forward(self, entity_ids):
        neighbor_entities, neighbor_relations = self.adj_entity[entity_ids.cpu()], self.adj_relation[entity_ids.cpu()]

        neighbor_entity_embedding = self.entity_embedding(torch.tensor(neighbor_entities).cuda())  # (B, EN, NE, E)
        neighbor_relation_embedding = self.relation_embedding(torch.tensor(neighbor_relations).cuda())  # same as above
        entity_embedding = self.entity_embedding(entity_ids)  # shape (B, EN, E)
        entity_embedding_expand = torch.unsqueeze(entity_embedding, 2).expand(  # shape (B, EN, 1, E)
            entity_embedding.shape[0], entity_embedding.shape[1], self.entity_neighbor_num, entity_embedding.shape[2]
        )
        embedding_concat = torch.cat([entity_embedding_expand, neighbor_entity_embedding,
                                      neighbor_relation_embedding], 3)  # (B, EN, NE, 3*E)
        attention_value = self.softmax(self.attention_layer2(self.relu(self.attention_layer1(embedding_concat))))
        neighbor_att_embedding = torch.sum(attention_value * neighbor_entity_embedding, dim=2)
        kgat_embedding = self.aggregate(entity_embedding, neighbor_att_embedding)
        return kgat_embedding
