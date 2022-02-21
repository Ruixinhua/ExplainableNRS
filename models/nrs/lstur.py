import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.general import AttLayer
from models.nrs.rs_base import MindNRSBase


class LSTURRSModel(MindNRSBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category_num, self.num_filters = kwargs.get("category_num", 300), kwargs.get("num_filters", 300)
        self.use_category, self.use_sub = kwargs.get("use_category", 0), kwargs.get("use_subcategory", 0)
        self.user_embed_method, self.user_num = kwargs.get("user_embed_method", None), kwargs.get("user_num", 500001)
        padding = (self.window_size - 1) // 2
        assert 2 * padding == self.window_size - 1, "Kernel size must be an odd number"
        self.news_encode_layer = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.num_filters, self.window_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        self.news_att_layer = AttLayer(self.num_filters, self.attention_hidden_dim)
        if self.use_category or self.use_sub:
            self.category_embedding = nn.Embedding(self.category_num, self.num_filters)
        input_dim = self.num_filters * 3 if self.use_category and self.use_sub else self.num_filters
        output_dim = self.num_filters
        if self.user_embed_method == "cat":
            output_dim = int(self.num_filters * 1.5)
        elif self.user_embed_method == "init" or (self.use_category and self.use_sub):
            output_dim = self.num_filters * 3
        if self.user_embed_method == "init" or self.user_embed_method == "cat":
            self.user_embedding = nn.Embedding(self.user_num, output_dim)
        self.user_encode_layer = nn.GRU(input_dim, output_dim, batch_first=True, bidirectional=False)
        self.user_att_layer = AttLayer(output_dim, self.attention_hidden_dim)
        self.dropouts = nn.Dropout(self.dropout_rate)

    def text_encode(self, news):
        y = self.dropouts(self.embedding_layer(news))
        y = self.news_encode_layer(y.transpose(1, 2)).transpose(1, 2)
        y = self.news_att_layer(self.dropouts(y))[0]
        return y

    def news_encoder(self, input_feat):
        """input_feat: Size is [N * H, S]"""
        if self.use_category or self.use_sub:
            news, cat = self.load_news_feat(input_feat, use_category=True)
            news_embed, cat_embed = self.text_encode(news), self.category_embedding(cat)
            y = torch.cat([torch.reshape(cat_embed, (cat_embed.shape[0], -1)), news_embed], dim=1)
        else:
            y = self.text_encode(input_feat["news"])
        return y

    def user_encoder(self, input_feat):
        y, user_ids = input_feat["history_news"], input_feat["uid"]
        packed_y = pack_padded_sequence(y, input_feat["history_length"].cpu(),
                                        batch_first=True, enforce_sorted=False)
        if self.user_embed_method == "init":
            user_embed = self.user_embedding(user_ids)
            _, y = self.user_encode_layer(packed_y, user_embed.unsqueeze(dim=0))
            y = y.squeeze(dim=0)
        elif self.user_embed_method == "cat":
            user_embed = self.user_embedding(user_ids)
            _, y = self.user_encode_layer(packed_y)
            y = torch.cat((y.squeeze(dim=0), user_embed), dim=1)
        elif self.user_embed_method == "att":
            y = self.user_encode_layer(packed_y)[0]
            y, _ = pad_packed_sequence(y, batch_first=True)
            y = self.user_att_layer(y)[0]  # additive attention layer
        else:  # default use last hidden output
            y = self.user_encode_layer(packed_y)[1].squeeze(dim=0)
            # y = self.user_att_layer(y)[0]  # additive attention layer
        return y
