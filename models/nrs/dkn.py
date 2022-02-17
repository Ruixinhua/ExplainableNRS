import torch
import torch.nn as nn

from models.nrs.rs_base import MindNRSBase
from models.general import AttLayer, DNNClickPredictor
from utils.graph_untils import construct_entity_embedding


class DKNRSModel(MindNRSBase):
    def __init__(self, **kwargs):
        super(DKNRSModel, self).__init__(**kwargs)
        entity_embedding, _ = construct_entity_embedding(**kwargs)
        self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding.cuda())
        self.num_filters, self.layer_dim = kwargs.get("num_filters", 50), kwargs.get("layer_dim", 50)
        self.window_sizes = kwargs.get("window_sizes", [2, 3, 4])
        self.news_entity_num = kwargs.get("news_entity_num", 10)
        self.entity_embedding_dim = kwargs.get("entity_embedding_dim", 100)

        self.additive_attention = AttLayer(self.num_filters, self.layer_dim)
        self.trans_weight = nn.Parameter(torch.empty(self.entity_embedding_dim, self.embedding_dim).uniform_(-0.1, 0.1))
        self.trans_bias = nn.Parameter(torch.empty(self.embedding_dim).uniform_(-0.1, 0.1))
        self.conv_filters = nn.ModuleDict({
            str(x): nn.Conv2d(2, self.num_filters, (x, self.embedding_dim))
            for x in self.window_sizes
        })
        self.user_att = nn.Sequential(nn.Linear(len(self.window_sizes) * 2 * self.num_filters, 16), nn.Linear(16, 1))
        self.click_predictor = DNNClickPredictor(len(self.window_sizes) * 2 * self.num_filters, self.layer_dim)

    def kcnn(self, title_embed, entity_embed):
        """
        Knowledge-aware CNN (KCNN) based on Kim CNN.
        Input a news sentence (e.g. its title), produce its embedding vector.
        """
        # batch_size, num_words_title, word_embedding_dim
        entity_vector_trans = torch.tanh(torch.add(torch.matmul(entity_embed, self.trans_weight), self.trans_bias))
        # batch_size, 2, num_words_title, word_embedding_dim
        multi_channel_vector = torch.stack([title_embed, entity_vector_trans], dim=1)
        pooled_vectors = []
        for x in self.window_sizes:
            # batch_size, num_filters, num_words_title + 1 - x
            convoluted = self.conv_filters[str(x)](multi_channel_vector).squeeze(dim=3)
            # batch_size, num_filters, num_words_title + 1 - x
            activated = torch.relu(convoluted)
            # batch_size, num_filters
            # Here we use an additive attention module
            # instead of pooling in the paper
            pooled = self.additive_attention(activated.transpose(1, 2))[0]
            # pooled = activated.max(dim=-1)[0]
            # # or
            # # pooled = F.max_pool1d(activated, activated.size(2)).squeeze(dim=2)
            pooled_vectors.append(pooled)
        # batch_size, len(window_sizes) * num_filters
        final_vector = torch.cat(pooled_vectors, dim=1)
        return final_vector

    def user_attention(self, candidate_news_vector, clicked_news_vector):
        """
        Attention Net.
        Input embedding vectors (produced by KCNN) of a candidate news and all of user's clicked news,
        produce final user embedding vectors with respect to the candidate news.
        Args:
            candidate_news_vector: batch_size, len(window_sizes) * num_filters
            clicked_news_vector: batch_size, num_clicked_news_a_user, len(window_sizes) * num_filters
        Returns:
            user_vector: batch_size, len(window_sizes) * num_filters
        """
        # batch_size, num_clicked_news_a_user
        concat_vector = torch.cat((candidate_news_vector.expand(
            clicked_news_vector.shape[1], -1, -1).transpose(0, 1), clicked_news_vector), dim=2)
        clicked_news_weights = torch.softmax(self.user_att(concat_vector).squeeze(dim=2), dim=1)

        # batch_size, len(window_sizes) * num_filters
        user_vector = torch.bmm(clicked_news_weights.unsqueeze(dim=1), clicked_news_vector).squeeze(dim=1)
        return user_vector

    def news_encoder(self, input_feat):
        title, entity = input_feat["news"][:, :self.title_len], input_feat["news"][:, self.title_len:]
        entity_ids = entity.reshape(-1, 4, self.news_entity_num)[:, 0]  # (B, 4, EN)
        title_embed, entity_embed = self.embedding_layer(title), self.entity_embedding(entity_ids)
        news_vector = self.kcnn(title_embed, entity_embed)
        return news_vector

    def user_encoder(self, input_feat):
        clicked_news_vector, candidate_news_vector = input_feat["history_news"], input_feat["candidate_news"]
        # batch_size, 1 + K, len(window_sizes) * num_filters
        user_vector = torch.stack([self.user_attention(x, clicked_news_vector)
                                   for x in candidate_news_vector.transpose(0, 1)])
        return user_vector

    def predict(self, input_feat, **kwargs):
        """
        prediction logic: use MLP for prediction or Dot-product.
        :param input_feat: should include encoded candidate news and user representations (history_news)
        :return: softmax possibility of click candidate news
        """
        candidate_news, user_vector = input_feat["candidate_news"], input_feat["history_news"]
        shape = candidate_news.shape
        pred = self.click_predictor(
            candidate_news.view(shape[0]*shape[1], shape[2]), user_vector.view(shape[0]*shape[1], shape[2]))
        return pred.view(shape[0], shape[1])
