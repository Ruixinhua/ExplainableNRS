import torch
import torch.nn as nn

from models import MultiHeadedAttention, AttLayer
from models.nrs.rs_base import MindNRSBase
from models.general.kgat import KGAT
from utils.graph_untils import construct_adj, construct_entity_embedding


class KREDRSModel(MindNRSBase):
    def __init__(self, **kwargs):
        super(KREDRSModel, self).__init__(**kwargs)
        # construct entity embedding and adjacent matrix
        self.entity_embedding, self.relation_embedding = construct_entity_embedding(**kwargs)
        self.entity_adj, self.relation_adj = construct_adj(**kwargs)
        self.kgat = KGAT(self.entity_embedding, self.relation_embedding, self.entity_adj, self.relation_adj, **kwargs)
        # self.sentence_encode = SentenceTransformer('distilbert-base-nli-stsb-mean-tokens')
        # initial some parameters
        self.entity_embedding_dim = kwargs.get("entity_embedding_dim", 100)
        max_frequency, max_entity_category = kwargs.get("max_frequency", 100), kwargs.get("max_entity_category", 100)
        self.news_entity_num, self.layer_dim = kwargs.get("news_entity_num", 10), kwargs.get("layer_dim", 128)

        self.news_encode_layer = MultiHeadedAttention(self.head_num, self.head_dim, self.embedding_dim)
        self.news_att_layer = AttLayer(self.head_num * self.head_dim, self.attention_hidden_dim)
        self.document_embedding_dim = kwargs.get("document_embedding_dim", self.head_num * self.head_dim)

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
            self.user_encode_layer = MultiHeadedAttention(self.head_num, self.entity_embedding_dim / self.head_num,
                                                          self.entity_embedding_dim)
            self.user_att_layer = AttLayer(self.entity_embedding_dim, self.attention_hidden_dim)
        elif self.user_layer == "gru":
            self.user_encode_layer = nn.GRU(self.entity_embedding_dim, self.entity_embedding_dim,
                                            batch_first=True, bidirectional=False)
        self.mlp_layer = nn.Sequential(
            nn.Linear(2 * self.entity_embedding_dim, self.layer_dim),  # first MLP layer
            nn.ReLU(inplace=True),  # use ReLU activation
            nn.Linear(self.layer_dim, 1),  # second MLP layer
            nn.Sigmoid(),  # Sigmoid activation for the final output
        )

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
        title, entity = input_feat["news"][:, :self.title_len], input_feat["news"][:, self.title_len:]
        y = self.dropouts(self.embedding_layer(title))  # encode document
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
        elif self.user_layer == "gru":
            y = self.user_encode_layer(y)[0]
        y = torch.sum(y * self.user_att_layer(y), dim=1)
        return y

    def predict(self, input_feat, **kwargs):
        candidate_news, user_embedding = input_feat["candidate_news"], input_feat["history_news"]
        evaluate = kwargs.get("evaluate", False)
        if self.out_layer == "mlp":
            if len(candidate_news.shape) != len(user_embedding.shape):
                user_embedding = torch.unsqueeze(user_embedding, 1)  # expand user embedding to candidate news shape
                user_embedding = user_embedding.expand([
                    user_embedding.shape[0], candidate_news.shape[1], user_embedding.shape[2]])
            u_n_embedding = torch.cat([user_embedding, candidate_news], dim=(len(user_embedding.shape) - 1))
            pred = self.mlp_layer(u_n_embedding).squeeze()
        else:
            pred = torch.sum(candidate_news * user_embedding.unsqueeze(1), dim=-1)
            if not evaluate:
                pred = torch.softmax(pred, dim=-1)
        return pred
