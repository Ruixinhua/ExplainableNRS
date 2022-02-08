import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from models.general.layers import MultiHeadedAttention, AttLayer
from models.nc_models import BaseClassifyModel


class TextCNNClassifyModel(BaseClassifyModel):
    """Time-consuming and the performance is not good, score is about 0.67 in News26 with 1 CNN layer"""
    def __init__(self, **kwargs):
        super(TextCNNClassifyModel, self).__init__(**kwargs)
        self.num_filters, self.filter_sizes = kwargs.get("num_filters", 256), kwargs.get("filter_sizes", (2, 3, 4))
        self.conv_layers = nn.ModuleList(
            [nn.Conv2d(1, self.num_filters, (k, self.embed_dim)) for k in self.filter_sizes])
        self.classifier = nn.Linear(self.num_filters * len(self.filter_sizes), self.num_classes)

    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, input_feat, **kwargs):
        x = self.embedding_layer(input_feat).unsqueeze(1)
        x = torch.cat([self.conv_and_pool(x, conv) for conv in self.conv_layers], 1)
        x = nn.Dropout(self.dropout_rate)(x)
        return self.classify_layer(x)


class NRMSNewsEncoderModel(BaseClassifyModel):
    def __init__(self, **kwargs):
        super(NRMSNewsEncoderModel, self).__init__(**kwargs)
        head_num = self.embed_dim // 20
        self.mha_encoder = MultiHeadedAttention(head_num, 20, self.embed_dim)
        self.news_att = AttLayer(self.embed_dim, 128)

    def forward(self, input_feat, **kwargs):
        x = self.embedding_layer(input_feat)
        if self.variant_name == "one_att":
            x = self.news_att(x)[0]
        else:
            x = self.mha_encoder(x, x, x)[0]
            x = nn.Dropout(self.dropout_rate)(x)
            x = self.news_att(x)[0]
        return self.classify_layer(x)


class GRUAttClassifierModel(BaseClassifyModel):
    def __init__(self, **kwargs):
        super(GRUAttClassifierModel, self).__init__(**kwargs)
        if self.variant_name == "gru_att":
            self.gru = nn.GRU(self.embed_dim, self.embed_dim, 2, batch_first=True)
            self.news_att = AttLayer(self.embed_dim, 128)
        elif self.variant_name == "biLSTM_att":
            self.gru = nn.LSTM(self.embed_dim, self.embed_dim, batch_first=True, bidirectional=True)
            self.news_att = AttLayer(self.embed_dim * 2, 128)
            self.classifier = nn.Linear(self.embed_dim * 2, self.num_classes)

    def run_gru(self, embedding, length):
        embedding = pack_padded_sequence(embedding, lengths=length.cpu(), batch_first=True, enforce_sorted=False)
        y, _ = self.gru(embedding)  # extract interest from history behavior
        y, _ = pad_packed_sequence(y, batch_first=True, total_length=self.max_length)
        return y

    def forward(self, input_feat, **kwargs):
        x = self.embedding_layer(input_feat)
        length = torch.sum(input_feat["mask"], dim=-1)
        x = self.run_gru(x, length)
        x = nn.Dropout(self.dropout_rate)(x)
        x = self.news_att(x)[0]
        return self.classify_layer(x)
