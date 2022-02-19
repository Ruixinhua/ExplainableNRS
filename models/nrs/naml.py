import torch
import torch.nn as nn

from models.general import AttLayer
from models.nrs.rs_base import MindNRSBase


class NAMLRSModel(MindNRSBase):
    def __init__(self, **kwargs):
        self.document_embedding_dim, self.window_size = kwargs.get("num_filters", 300), kwargs.get("window_size", 3)
        self.category_num, self.category_dim = kwargs.get("category_num", 300), kwargs.get("category_dim", 100)
        self.use_category, self.use_sub = kwargs.get("use_category", 0), kwargs.get("use_subcategory", 0)
        super(NAMLRSModel, self).__init__(**kwargs)
        self.category_embedding = nn.Embedding(self.category_num, self.category_dim)  # max category number 300
        self.category_linear = nn.Sequential(
            self.category_embedding, nn.Linear(self.category_dim, self.document_embedding_dim), nn.ReLU(inplace=True)
        )
        self.text_cnn = nn.Conv2d(1, self.document_embedding_dim, (self.window_size, self.embedding_dim),
                                  padding=(int((self.window_size - 1) / 2), 0))
        self.final_attention = AttLayer(self.document_embedding_dim, self.attention_hidden_dim)
        self.dropouts = nn.Dropout(self.dropout_rate)

    def text_encode(self, news):
        y = self.dropouts(self.embedding_layer(news))
        y = self.text_cnn(y.unsqueeze(dim=1)).squeeze(dim=3)  # Text CNN layer
        y = self.news_att_layer(self.dropouts(torch.relu(y)).transpose(1, 2))[0]
        return y

    def news_encoder(self, input_feat):
        """input_feat contains: news title + news category"""
        if self.use_category or self.use_sub:
            news, cat = input_feat["news"][:, :self.title_len], input_feat["news"][:, self.title_len:self.title_len+2]
            news_embed, cat_embed = self.text_encode(news), self.category_linear(cat)  # encode text and category
            y = self.final_attention(torch.cat([torch.unsqueeze(news_embed, 1), cat_embed], dim=1))[0]
        else:
            y = self.text_encode(input_feat["news"])
        # add activation function
        return y
