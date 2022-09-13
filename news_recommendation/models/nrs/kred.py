import torch
import torch.nn as nn

from news_recommendation.models.general import MultiHeadedAttention, AttLayer, DNNClickPredictor
from news_recommendation.models.nrs.rs_base import MindNRSBase
from news_recommendation.models.general.kgat import KGAT


class KREDRSModel(MindNRSBase):
    def __init__(self, **kwargs):
        self.head_num, self.head_dim = kwargs.get("head_num", 20), kwargs.get("head_dim", 20)
        self.document_embedding_dim = kwargs.get("document_embedding_dim", self.head_num * self.head_dim)
        super(KREDRSModel, self).__init__(**kwargs)
        # construct entity embedding and adjacent matrix
        self.kgat = KGAT(**kwargs)  # load knowledge graph attention network
        # initial some parameters
        self.entity_embedding_dim = kwargs.get("entity_embedding_dim", 100)
        max_frequency, max_entity_category = kwargs.get("max_frequency", 100), kwargs.get("max_entity_category", 100)
        self.news_entity_num, self.layer_dim = kwargs.get("news_entity_num", 10), kwargs.get("layer_dim", 128)
        self.news_encode_layer = MultiHeadedAttention(self.head_num, self.head_dim, self.embedding_dim)
        self.news_att_layer = AttLayer(self.head_num * self.head_dim, self.attention_hidden_dim)
        self.use_sent_embed = kwargs.get("sentence_embed_method", None) and self.document_embedding_dim

        self.news_final_layer = nn.Sequential(
            nn.Linear(self.document_embedding_dim + self.entity_embedding_dim, self.layer_dim), nn.ReLU(inplace=True),
            nn.Linear(self.layer_dim, self.entity_embedding_dim), nn.Tanh()
        )
        self.dropouts = nn.Dropout(self.dropout_rate)

        self.position_encoding = self.init_embedding(10)
        self.frequency_encoding = self.init_embedding(max_frequency)
        self.category_encoding = self.init_embedding(max_entity_category)

        self.news_attention_layer = nn.Sequential(
            nn.Linear(self.document_embedding_dim + self.entity_embedding_dim, self.layer_dim), nn.ReLU(inplace=True),
            nn.Linear(self.layer_dim, 1), nn.ReLU(inplace=True), nn.Softmax(dim=-2),  # output weight
        )
        self.user_att_layer = nn.Sequential(
            nn.Linear(self.entity_embedding_dim, self.layer_dim), nn.ReLU(inplace=True),  # first attention layer
            nn.Linear(self.layer_dim, 1), nn.ReLU(inplace=True), nn.Softmax(dim=0)  # second layer: output weight
        )
        self.user_layer = kwargs.get("user_layer", None)
        if self.user_layer == "mha":
            self.user_encode_layer = MultiHeadedAttention(self.head_num, self.entity_embedding_dim // self.head_num,
                                                          self.entity_embedding_dim)
            self.user_att_layer = AttLayer(self.entity_embedding_dim, self.attention_hidden_dim)
        elif self.user_layer == "gru":
            self.user_encode_layer = nn.GRU(self.entity_embedding_dim, self.entity_embedding_dim,
                                            batch_first=True, bidirectional=False)
        if self.out_layer == "mlp":
            self.click_predictor = DNNClickPredictor(self.entity_embedding_dim * 2, self.layer_dim)

    def init_embedding(self, num_embeddings):
        embedding = nn.Embedding(num_embeddings, self.entity_embedding_dim)
        weight = torch.FloatTensor(num_embeddings, self.entity_embedding_dim)
        # Use xavier initialization method to initialize embeddings of entities and relations
        nn.init.xavier_normal_(weight, gain=0.01)
        embedding.weight = nn.Parameter(weight)
        return embedding

    def news_attention(self, entity_embeddings, context_vec):
        """TODO: news attention layer"""
        context_vec = torch.unsqueeze(context_vec, -2)
        if len(entity_embeddings.shape) == 4:
            shape = [context_vec.shape[0], context_vec.shape[1], entity_embeddings.shape[2], context_vec.shape[3]]
        else:
            shape = [context_vec.shape[0], entity_embeddings.shape[1], context_vec.shape[2]]
        context_vec = context_vec.expand(shape)
        concat_embedding = torch.cat([entity_embeddings, context_vec], dim=-1)
        soft_att_value = self.news_attention_layer(concat_embedding)
        weighted_entity_embedding = soft_att_value * entity_embeddings
        weighted_entity_embedding_sum = torch.sum(weighted_entity_embedding, dim=-2)
        return weighted_entity_embedding_sum, soft_att_value

    def news_encoder(self, input_feat):
        # the order of news is title, body(option), document embed, entity feature
        if self.use_sent_embed:
            entity_index = self.title_len+self.document_embedding_dim
            context_vec = input_feat["news"][:, self.title_len:entity_index]
            entity = input_feat["news"][:, entity_index:]
        else:  # TODO: optimize input format
            title, entity = input_feat["news"][:, :self.title_len], input_feat["news"][:, self.title_len:]
            input_feat["news"] = title
            y = self.dropouts(self.embedding_layer(input_feat))  # encode document
            y = self.dropouts(self.news_encode_layer(y, y, y)[0])
            context_vec = self.news_att_layer(y)[0]
        entity = entity.reshape(-1, 4, self.news_entity_num)  # (B, 4, EN)
        entity_ids, entity_freq, entity_pos, entity_type = entity[:, 0], entity[:, 1], entity[:, 2], entity[:, 3]
        freq_embedding, pos_embedding = self.frequency_encoding(entity_freq), self.position_encoding(entity_pos)
        type_embedding, kgat_embeddings = self.category_encoding(entity_type), self.kgat(entity_ids)
        entity_embedding = kgat_embeddings + freq_embedding + pos_embedding + type_embedding
        aggregate_embedding, _ = self.news_attention(entity_embedding, context_vec)
        concat_embedding = torch.cat([aggregate_embedding, context_vec], len(aggregate_embedding.shape) - 1)
        news_embeddings = self.news_final_layer(concat_embedding)
        return news_embeddings

    def user_encoder(self, input_feat):
        y = input_feat["history_news"]
        if self.user_layer == "mha":
            y = self.user_encode_layer(y, y, y)[0]  # the MHA layer for user encoding
            y = self.user_att_layer(y)[0]
        elif self.user_layer == "gru":
            y = self.user_encode_layer(y)[0]
            y = torch.sum(y * self.user_att_layer(y), dim=1)
        else:
            y = torch.sum(y * self.user_att_layer(y), dim=1)
        return y
