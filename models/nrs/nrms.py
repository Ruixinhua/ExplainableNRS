import torch.nn as nn

from models.general import AttLayer
from models.general import MultiHeadedAttention
from models.nrs.rs_base import MindNRSBase


class NRMSRSModel(MindNRSBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.news_att_layer = AttLayer(self.head_num * self.head_dim, self.attention_hidden_dim)
        self.user_att_layer = AttLayer(self.head_num * self.head_dim, self.attention_hidden_dim)
        self.news_encode_layer = MultiHeadedAttention(self.head_num, self.head_dim, self.embedding_dim)
        if self.user_encoder == "mha":
            self.user_encode_layer = MultiHeadedAttention(self.head_num, self.head_dim, self.head_num * self.head_dim)
        elif self.user_encoder == "gru":
            self.user_encode_layer = nn.GRU(self.head_num * self.head_dim, self.head_num * self.head_dim,
                                            batch_first=True, bidirectional=False)
        self.dropouts = nn.Dropout(self.dropout_rate)

    def news_encoder(self, input_feat):
        """input_feat: Size is [N * H, S]"""
        y = self.dropouts(self.embedding_layer(input_feat["news"]))
        y = self.news_encode_layer(y, y, y)[0]  # the MHA layer for news encoding
        y = self.dropouts(y)  # TODO dropout layer
        # add activation function
        return self.news_att_layer(y)[0]

    def user_encoder(self, input_feat):
        y = input_feat["history_news"]
        if self.user_encoder == "mha":
            y = self.user_encode_layer(y, y, y)[0]  # the MHA layer for user encoding
        elif self.user_encoder == "gru":
            y = self.user_encode_layer(y)[0]
        y = self.user_att_layer(y)[0]  # additive attention layer
        return y
