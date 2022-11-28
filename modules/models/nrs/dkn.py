import os
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path

from modules.models.nrs.rs_base import MindNRSBase
from modules.models.general import AttLayer, DNNClickPredictor, DotProduct
from modules.utils import get_project_root, download_resources
from modules.utils.graph_untils import construct_entity_embedding


class DKNRSModel(MindNRSBase):
    def __init__(self, **kwargs):
        super(DKNRSModel, self).__init__(**kwargs)
        self.num_filters, self.layer_dim = kwargs.get("num_filters", 128), kwargs.get("layer_dim", 50)
        self.window_sizes = kwargs.get("window_sizes", [2, 3, 4])
        self.predictor = kwargs.get("predictor", "dot_product")
        self.use_entity, self.use_context = kwargs.get("use_entity", None), kwargs.get("use_context", None)
        self.news_entity_num = self.title_len if self.use_entity else None
        self.entity_embedding_dim = kwargs.get("entity_embedding_dim", 100)  # default is 100
        self.use_dkn_utils = kwargs.get("dkn_utils", None)

        if self.use_entity:
            if self.use_dkn_utils:
                mind_type = kwargs.get("mind_type")
                utils_path = Path(get_project_root()) / "dataset/utils/dkn_utils" / f"mind-{mind_type}-dkn"
                os.makedirs(utils_path, exist_ok=True)
                yaml_file = utils_path / "dkn.yaml"
                if not yaml_file.exists():  # TODO: bug in this part
                    download_resources(r"https://recodatasets.z20.web.core.windows.net/deeprec/",
                                       str(utils_path.parent), f"mind-{mind_type}-dkn.zip")
                word_embed_file = utils_path / "word_embeddings_100.npy"
                entity_embed_file = utils_path / "TransE_entity2vec_100.npy"
                word_embedding = torch.FloatTensor(np.load(str(word_embed_file))).cuda()
                entity_embedding = torch.FloatTensor(np.load(str(entity_embed_file)))  # load entity embedding
                self.embedding_layer = nn.Embedding.from_pretrained(word_embedding, freeze=False)
                self.embedding_dim = 100
                if self.use_context:
                    context_embed_file = utils_path / "TransE_context2vec_100.npy"
                    context_embedding = torch.FloatTensor(np.load(str(context_embed_file))).cuda()
                    self.context_embedding = nn.Embedding.from_pretrained(context_embedding, freeze=False)
            else:
                entity_embedding, _ = construct_entity_embedding(**kwargs)  # TODO: modify entity embedding
            self.entity_embedding = nn.Embedding.from_pretrained(entity_embedding.cuda(), freeze=False)
            self.trans_weight = nn.Parameter(  # init weight of linear function
                torch.empty(self.entity_embedding_dim, self.embedding_dim).uniform_(-0.1, 0.1))
            self.trans_bias = nn.Parameter(torch.empty(self.embedding_dim).uniform_(-0.1, 0.1))
            self.conv_filters = nn.ModuleDict({
                str(x): nn.Conv2d(3 if self.use_context else 2, self.num_filters, (x, self.embedding_dim))
                for x in self.window_sizes
            })
        else:
            self.conv_filters = nn.ModuleDict({
                str(x): nn.Conv1d(self.embedding_dim, self.num_filters, x)
                for x in self.window_sizes
            })
        self.additive_attention = AttLayer(self.num_filters, self.layer_dim)
        news_embed_dim = len(self.window_sizes) * self.num_filters * 2
        self.user_att = nn.Sequential(nn.Linear(news_embed_dim, 16), nn.Linear(16, 1))
        if self.predictor == "dot_product":
            self.click_predictor = DotProduct()
        elif self.predictor == "dnn":
            self.click_predictor = DNNClickPredictor(news_embed_dim, self.layer_dim)
        self.news_att_layer, self.user_att_layer = None, None  # No attention for user layer and news layer

    def kcnn(self, title_embed, entity_embed=None):
        """
        Knowledge-aware CNN (KCNN) based on Kim CNN.
        Input a news sentence (e.g. its title), produce its embedding vector.
        """
        if entity_embed is not None:
            # batch_size, num_words_title, word_embedding_dim
            entity_vector_trans = torch.tanh(torch.add(torch.matmul(entity_embed, self.trans_weight), self.trans_bias))
            # batch_size, 2, num_words_title, word_embedding_dim
            news_vector = torch.stack([title_embed, entity_vector_trans], dim=1)
        else:
            # batch_size, word_embedding_dim, num_words_title
            news_vector = title_embed.transpose(1, 2)
        pooled_vectors = []
        for x in self.window_sizes:
            # batch_size, num_filters, num_words_title + 1 - x
            if self.news_entity_num:
                convoluted = self.conv_filters[str(x)](news_vector).squeeze(dim=3)
            else:
                convoluted = self.conv_filters[str(x)](news_vector)
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
        Input embedding vectors (produced by KCNN) of a candidate news and all of user"s clicked news,
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
        if self.news_entity_num:
            title, entity = input_feat["news"][:, :self.title_len], input_feat["news"][:, self.title_len:]
            entity_ids = entity.reshape(-1, 4, self.news_entity_num)[:, 0]  # (B, 4, EN)
            title_embed, entity_embed = self.embedding_layer(title), self.entity_embedding(entity_ids)
            news_vector = self.kcnn(title_embed, entity_embed)
        else:
            news_vector = self.kcnn(self.embedding_layer(input_feat))
        return {"news_embed": news_vector}

    def user_encoder(self, input_feat):
        history_news, candidate_news = input_feat["history_news"], input_feat["candidate_news"]
        # batch_size, 1 + K, len(window_sizes) * num_filters
        user_vector = torch.stack([self.user_attention(x, history_news) for x in candidate_news.transpose(0, 1)], dim=1)
        return {"user_embed": user_vector}

    def predict(self, input_feat, **kwargs):
        """
        prediction logic: use MLP for prediction or Dot-product.
        :param input_feat: should include encoded candidate news and user representations (history_news)
        :return: softmax possibility of click candidate news
        """
        candidate_news, user_embed = input_feat["candidate_news"], input_feat["user_embed"]
        s = candidate_news.shape
        if self.predictor == "dot_product":
            # batch_size, 1 + K
            return self.click_predictor(user_embed, candidate_news)
        elif self.predictor == "dnn":
            pred = self.click_predictor(candidate_news.view(s[0]*s[1], s[2]), user_embed.view(s[0]*s[1], s[2]))
            return pred.view(s[0], s[1])
