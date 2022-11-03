import torch.nn as nn

from modules.models.general import AttLayer
from modules.models.general import MultiHeadedAttention
from modules.models.nrs.rs_base import MindNRSBase


class NRMSRSModel(MindNRSBase):
    """
    Implementation of NRMS model
    Wu, C., Wu, F., Ge, S., Qi, T., Huang, Y., & Xie, X. (2019).
    Neural News Recommendation with Multi-Head Self-Attention. EMNLP.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # define document embedding dim before inherit super class
        self.head_num, self.head_dim = kwargs.get("head_num", 20), kwargs.get("head_dim", 20)
        self.document_embedding_dim = kwargs.get("document_embedding_dim", self.head_num * self.head_dim)
        self.news_att_layer = AttLayer(self.document_embedding_dim, self.attention_hidden_dim)
        self.news_encode_layer = MultiHeadedAttention(self.head_num, self.head_dim, self.embedding_dim)
        self.user_layer = kwargs.get("user_layer", "mha")
        if self.user_layer == "mha":
            self.user_encode_layer = MultiHeadedAttention(self.head_num, self.head_dim, self.document_embedding_dim)
        elif self.user_layer == "gru":
            self.user_encode_layer = nn.GRU(self.document_embedding_dim, self.document_embedding_dim,
                                            batch_first=True, bidirectional=False)
        self.user_att_layer = AttLayer(self.document_embedding_dim, self.attention_hidden_dim)
        self.dropouts = nn.Dropout(self.dropout_rate)

    def news_encoder(self, input_feat):
        """input_feat: Size is [N * H, S]"""
        y = self.dropouts(self.embedding_layer(input_feat))
        y = self.news_encode_layer(y, y, y)[0]  # the MHA layer for news encoding
        y = self.dropouts(y)  # TODO dropout layer
        # add activation function
        return self.news_att_layer(y)[0]

    def user_encoder(self, input_feat):
        y = input_feat["history_news"]
        if self.user_layer == "mha":  # the methods used by NRMS original paper
            y = self.user_encode_layer(y, y, y)[0]  # the MHA layer for user encoding
        elif self.user_layer == "gru":
            y = self.user_encode_layer(y)[0]
        y = self.user_att_layer(y)[0]  # additive attention layer
        return y
