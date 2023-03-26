import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.models.general import AttLayer, Conv1D
from modules.models.nrs.rs_base import MindNRSBase


class NAMLRSModel(MindNRSBase):
    def __init__(self, **kwargs):
        self.use_category = True  # use category and subvert for NAML model
        super(NAMLRSModel, self).__init__(**kwargs)
        self.num_filters, self.window_size = kwargs.get("num_filters", 300), kwargs.get("window_size", 3)
        self.news_infos = kwargs.get("news_info", ["title", "body"])
        if self.use_category:
            self.category_affine = nn.Linear(self.category_dim, self.num_filters, bias=True)
            self.subvert_affine = nn.Linear(self.category_dim, self.num_filters, bias=True)
        self.reshape_tensors = [f"candidate_{f}" for f in self.news_infos] + [f"history_{f}" for f in self.news_infos]

        self.title_cnn = Conv1D(self.embedding_dim, self.num_filters, self.window_size)
        self.title_att = self.news_att_layer
        if "body" in self.news_infos:
            self.body_cnn = Conv1D(self.embedding_dim, self.num_filters, self.window_size)
            self.body_att = AttLayer(self.num_filters, self.attention_hidden_dim)
        self.multi_att = AttLayer(self.num_filters, self.attention_hidden_dim)

    def news_encoder(self, input_feat):
        """input_feat contains: news title, body, category, and subvert"""
        # 1. word embedding: [batch_size * news_num, max_news_length, word_embedding_dim]
        title = self.dropouts(self.embedding_layer(news=input_feat["title"], news_mask=input_feat["title_mask"]))
        # 2. cnn extract features: [batch_size * news_num, max_news_length, kernel_num]
        title_feature = self.dropouts(self.title_cnn(title.transpose(1, 2)).transpose(1, 2))
        # 3. attention layer: [batch_size * news_num, kernel_num]
        title_vector = self.title_att(title_feature)[0]
        feature = [title_vector]
        if "body" in self.news_infos:
            body = self.dropouts(self.embedding_layer(news=input_feat["body"], news_mask=input_feat["body_mask"]))
            body_feature = self.dropouts(self.body_cnn(body.transpose(1, 2)).transpose(1, 2))
            body_vector = self.body_att(body_feature)[0]
            feature.append(body_vector)
        # 4. category and subcategory encoding: [batch_size * news_num, kernel_num]
        if self.use_category:  # append category and subvert feature
            feature.append(F.relu(self.category_affine(self.category_embedding(input_feat["category"])), inplace=True))
            feature.append(F.relu(self.subvert_affine(self.subvert_embedding(input_feat["subvert"])), inplace=True))
        feature = torch.stack(feature, dim=1)
        # 5. multi-view attention: [batch_size*news_num, n, kernel_num]
        news_vector, weight = self.multi_att(feature)
        return {"news_embed": news_vector, "news_weight": weight}
