import numpy as np
import torch
import torch.nn as nn

from base.base_model import BaseModel
from models.general import AttLayer, DNN


class MindNRSBase(BaseModel):
    """Basic model for News Recommendation for MIND dataset"""
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        if self.embedding_type == "glove":
            self.glove_embedding = np.load(self.word_emb_file)
            self.embedding_layer = nn.Embedding(self.glove_embedding.shape[0], self.embedding_dim).from_pretrained(
                torch.FloatTensor(self.glove_embedding), freeze=False)
        self.news_att_layer = AttLayer(self.embedding_dim, self.attention_hidden_dim)
        self.user_att_layer = AttLayer(self.embedding_dim, self.attention_hidden_dim)
        if self.out_layer == "mlp":
            self.dnn = DNN(self.embedding_dim * 2, (256, 128), 'relu', 0, 0, False, init_std=self.init_std, seed=1024)
            self.final_layer = nn.Linear(128, 2)

    def news_encoder(self, input_feat):
        """input_feat: Size of  is [N * H, S]"""
        y = self.embedding_layer(input_feat["news"])
        # add activation function
        # y = nn.ReLU()(y)  # [N * H, D]
        return self.news_att_layer(y)[0]

    def time_distributed(self, news_index):
        # calculate news features across time series
        x_shape = torch.Size([-1]) + news_index.size()[2:]
        news_reshape = news_index.contiguous().view(x_shape)  # [N * H, S]
        y = self.news_encoder({"news": news_reshape})
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
        candidate_news, history_news = input_feat["candidate_news"], input_feat["history_news"]
        evaluate = kwargs.get("evaluate", False)
        if self.out_layer == "mlp":
            pred = self.dnn(torch.cat([candidate_news.squeeze(1), history_news], dim=-1))
            pred = self.final_layer(pred)
        else:
            pred = torch.sum(candidate_news * history_news.unsqueeze(1), dim=-1)
            if not evaluate:
                pred = torch.softmax(pred, dim=-1)
        return pred

    def forward(self, input_feat):
        """
        training logic: candidate news encoding->user modeling (history news)
        :param input_feat: should include two keys, candidate and history
        :return: prediction result in softmax possibility
        """
        input_feat["candidate_news"] = self.time_distributed(input_feat["candidate"])  # (B, C, E)
        input_feat["history_news"] = self.time_distributed(input_feat["history"])  # (B, H, E)
        input_feat["history_news"] = self.user_encoder(input_feat)  # user modeling: (B, E)
        return self.predict(input_feat)
