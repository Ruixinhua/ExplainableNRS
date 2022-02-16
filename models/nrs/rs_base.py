import numpy as np
import torch
import torch.nn as nn

from base.base_model import BaseModel
from models.general import AttLayer, DNN, DotProduct, DNNClickPredictor


class MindNRSBase(BaseModel):
    """Basic model for News Recommendation for MIND dataset"""
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        if self.embedding_type == "glove":
            self.glove_embedding = np.load(self.word_emb_file)
            self.embedding_layer = nn.Embedding(self.glove_embedding.shape[0], self.embedding_dim).from_pretrained(
                torch.FloatTensor(self.glove_embedding), freeze=False)
        self.title_len, self.body_len = kwargs.get("title", 30), kwargs.get("body", None)
        self.news_att_layer = AttLayer(self.embedding_dim, self.attention_hidden_dim)
        self.user_att_layer = AttLayer(self.embedding_dim, self.attention_hidden_dim)
        if self.out_layer == "product":
            self.click_predictor = DotProduct()
        else:
            self.click_predictor = DNNClickPredictor(self.document_embedding_dim * 2, self.attention_hidden_dim)

    def news_encoder(self, input_feat):
        """
        input_feat["news"]: [N * H, S],
        default only use title, and body is added after title.
        Dim S contains: title(30) + body(100) + document_embed(300/768) + entity_feature(4*entity_num)
        """
        y = self.embedding_layer(input_feat["news"])
        # add activation function
        # y = nn.ReLU()(y)  # [N * H, D]
        return self.news_att_layer(y)[0]

    def time_distributed(self, news_index, news_mask=None):
        # calculate news features across multiple input news: [N * H, S]
        x_shape = torch.Size([-1]) + news_index.size()[2:]
        news_reshape = news_index.contiguous().view(x_shape)  # [N * H, S]
        mask_reshape = news_mask.contiguous().view(x_shape) if news_mask is not None else news_mask
        y = self.news_encoder({"news": news_reshape, "news_mask": mask_reshape})
        y = y.contiguous().view(news_index.size(0), -1, y.size(-1))  # change size to (N, H, D)
        return y

    def user_encoder(self, input_feat):
        y = self.user_att_layer(input_feat["history_news"])[0]
        return y

    def predict(self, input_feat, **kwargs):
        """
        prediction logic: use MLP for prediction or Dot-product.
        :param input_feat: should include encoded candidate news and user representations (history_news)
        :return: softmax possibility of click candidate news
        """
        candidate_news, user_vector = input_feat["candidate_news"], input_feat["history_news"]
        if self.out_layer == "mlp" and len(candidate_news.shape) != len(user_vector.shape):
            user_vector = torch.unsqueeze(user_vector, 1).expand([user_vector.shape[0], candidate_news.shape[1], -1])
        pred = self.click_predictor(candidate_news, user_vector)
        return pred

    def forward(self, input_feat):
        """
        training logic: candidate news encoding->user modeling (history news)
        :param input_feat: should include two keys, candidate and history
        :return: prediction result in softmax possibility
        """
        # [N, C, E]
        input_feat["candidate_news"] = self.time_distributed(input_feat["candidate"], input_feat["candidate_mask"])
        # [N, H, E]
        input_feat["history_news"] = self.time_distributed(input_feat["history"], input_feat["history_mask"])
        input_feat["history_news"] = self.user_encoder(input_feat)  # user modeling: [N, E]
        return self.predict(input_feat)
