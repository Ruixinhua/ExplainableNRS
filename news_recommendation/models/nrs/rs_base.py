import numpy as np
import torch
import torch.nn as nn

from news_recommendation.base.base_model import BaseModel
from news_recommendation.models.general import AttLayer, DotProduct, DNNClickPredictor, NewsEmbedding


class MindNRSBase(BaseModel):
    """Basic model for News Recommendation for MIND dataset"""
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.attention_hidden_dim = kwargs.get("attention_hidden_dim", 200)
        self.embedding_layer = NewsEmbedding(**kwargs)
        self.embedding_dim = self.embedding_layer.embed_dim
        self.title_len, self.body_len = kwargs.get("title", 30), kwargs.get("body", None)
        self.news_att_layer = AttLayer(self.embedding_dim, self.attention_hidden_dim)
        self.user_att_layer = AttLayer(self.embedding_dim, self.attention_hidden_dim)
        self.dropouts = nn.Dropout(self.dropout_rate)
        if self.out_layer == "product":
            self.click_predictor = DotProduct()
        else:
            self.click_predictor = DNNClickPredictor(self.document_embedding_dim * 2, self.attention_hidden_dim)

    def load_news_feat(self, input_feat, **kwargs):
        """the order of news info: title(abstract), category, sub-category, sentence embedding, entity feature"""
        use_category, use_sent_embed = kwargs.get("use_category", 0), kwargs.get("use_sent_embed", 0)
        news_info = [input_feat["news"][:, :self.title_len]]  # default only use title
        if use_category:
            news_info.append(input_feat["news"][:, self.title_len:self.title_len + 2])  # add category and sub category
        return news_info

    def news_encoder(self, input_feat):
        """
        input_feat["news"]: [N * H, S],
        default only use title, and body is added after title.
        Dim S contains: title(30) + body(100) + document_embed(300/768) + entity_feature(4*entity_num)
        """
        y = self.embedding_layer(input_feat)
        # add activation function
        # y = nn.ReLU()(y)  # [N * H, D]
        return self.news_att_layer(y)[0]

    def time_distributed(self, news_index, news_mask=None, **kwargs):
        # calculate news features across multiple input news: [N * H, S]
        x_shape = torch.Size([-1]) + news_index.size()[2:]
        news_reshape = news_index.contiguous().view(x_shape)  # [N * H, S]
        mask_reshape = news_mask.contiguous().view(x_shape) if news_mask is not None else news_mask
        feat = {"news": news_reshape, "news_mask": mask_reshape}
        feat.update(kwargs)
        y = self.news_encoder(feat)
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
        candidate_news, user_embeds = input_feat["candidate_news"], input_feat["user_embeds"]
        if self.out_layer == "mlp" and len(candidate_news.shape) != len(user_embeds.shape):
            user_embeds = torch.unsqueeze(user_embeds, 1).expand([user_embeds.shape[0], candidate_news.shape[1], -1])
        pred = self.click_predictor(candidate_news, user_embeds)
        return pred

    def forward(self, input_feat):
        """
        training logic: candidate news encoding->user modeling (history news)
        :param input_feat: should include two keys, candidate and history
        :return: prediction result in softmax possibility
        """
        if "candidate_news" not in input_feat:  # pass news embeddings cache in evaluation stage
            input_feat["candidate_news"] = self.time_distributed(input_feat["candidate"], input_feat["candidate_mask"])
            # output candidate news embedding shape: [N, C, E], C is the number of candidate news
        if "history_news" not in input_feat:  # no cache found
            input_feat["history_news"] = self.time_distributed(input_feat["history"], input_feat["history_mask"])
            # output history news embedding shape: [N, H, E], H is the number of history news
        input_feat["user_embeds"] = self.user_encoder(input_feat)  # user embedding shape: [N, E]
        return self.predict(input_feat)
