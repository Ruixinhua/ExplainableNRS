import torch.nn as nn

from models.nrs.rs_base import MindNRSBase


class LSTURRSModel(MindNRSBase):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        padding = (self.kernel_size - 1) // 2
        assert 2 * padding == self.kernel_size - 1, "Kernel size must be an odd number"
        self.news_encode_layer = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.embedding_dim, self.kernel_size, padding=padding),
            nn.ReLU()
        )
        self.user_encode_layer = nn.GRU(self.embedding_dim, self.embedding_dim, batch_first=True, bidirectional=False)
        self.dropouts = nn.Dropout(self.dropout_rate)

    def news_encoder(self, input_feat):
        """input_feat: Size is [N * H, S]"""
        y = self.dropouts(self.embedding_layer(input_feat["news"]))
        y = self.news_encode_layer(y.transpose(1, 2)).transpose(1, 2)
        y = self.dropouts(y)  # TODO dropout layer
        # add activation function
        return self.news_att_layer(y)[0]

    def user_encoder(self, input_feat):
        y = input_feat["history_news"]
        y = self.user_encode_layer(y)[0]
        y = self.user_att_layer(y)[0]  # additive attention layer
        return y
