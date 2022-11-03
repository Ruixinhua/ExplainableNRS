import torch
import torch.nn as nn

from modules.models import PersonalizedAttentivePooling
from modules.models.general import AttLayer
from modules.models.nrs.rs_base import MindNRSBase
from modules.utils import read_json


class NPARSModel(MindNRSBase):
    """
    Implementation of NPM model
    Wu, Chuhan et al. “NPA: Neural News Recommendation with Personalized Attention.”
    Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining (2019): n. pag.
    """
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.category_num, self.num_filters = kwargs.get("category_num", 300), kwargs.get("num_filters", 300)
        self.user_embed_method = kwargs.get("user_embed_method", None)
        self.user_emb_dim, self.window_size = kwargs.get("user_emb_dim", 100), kwargs.get("window_size", 3)
        padding = (self.window_size - 1) // 2
        assert 2 * padding == self.window_size - 1, "Kernel size must be an odd number"
        self.news_encode_layer = nn.Sequential(
            nn.Conv1d(self.embedding_dim, self.num_filters, self.window_size, padding=padding),
            nn.ReLU(inplace=True)
        )
        if self.user_embed_method == "init":  # NPA paper uses user id as initialization
            uid_path = kwargs.get("uid_path", None)
            if uid_path is None:
                raise ValueError("Must specify user id dictionary path if you want to use user id to initialize GRU")
            uid2index = read_json(uid_path)
            self.user_embedding = nn.Embedding(len(uid2index), self.user_emb_dim)
            self.user_transform = nn.Linear(self.user_emb_dim, self.attention_hidden_dim)
            self.news_att_layer = PersonalizedAttentivePooling(self.num_filters, self.attention_hidden_dim)
            self.user_att_layer = PersonalizedAttentivePooling(self.num_filters, self.attention_hidden_dim)
        else:
            self.news_att_layer = AttLayer(self.embedding_dim, self.attention_hidden_dim)
            self.user_att_layer = AttLayer(self.embedding_dim, self.attention_hidden_dim)

    def text_encode(self, input_feat):
        y = self.dropouts(self.embedding_layer(input_feat))
        y = self.dropouts(self.news_encode_layer(y.transpose(1, 2)).transpose(1, 2))
        return y

    def news_encoder(self, input_feat):
        """input_feat: Size is [N * H, S]"""
        news_emb = self.text_encode(input_feat)
        if self.user_embed_method == "init":
            user_emb = self.user_transform(self.user_embedding(input_feat["uid"]))
            y = self.news_att_layer(news_emb, user_emb)[0]
        else:
            y = self.news_att_layer(news_emb)[0]
        return y

    def user_encoder(self, input_feat):
        if "history_news" in input_feat:
            news_emb = input_feat["history_news"]
        else:
            news_emb = self.time_distributed(input_feat["history"], input_feat["history_mask"],
                                             uid=input_feat["uid"])  # [N, H, E]
        if self.user_embed_method == "init":
            user_emb = self.user_transform(self.user_embedding(input_feat["uid"]))
            y = self.user_att_layer(news_emb, user_emb)[0]
        else:  # default use last hidden output
            y = self.user_att_layer(news_emb)[0]
            # y = self.user_att_layer(y)[0]  # additive attention layer
        return y

    def time_distributed(self, news_index, news_mask=None, uid=None):
        # calculate news features across multiple input news: [N * H, S]
        x_shape = torch.Size([-1]) + news_index.size()[2:]
        news_reshape = news_index.contiguous().view(x_shape)  # [N * H, S]
        mask_reshape = news_mask.contiguous().view(x_shape) if news_mask is not None else news_mask
        uid = uid.unsqueeze(1).expand((-1, news_index.shape[1])).reshape(-1)
        feat = {"news": news_reshape, "news_mask": mask_reshape, "uid": uid}
        y = self.news_encoder(feat)
        y = y.contiguous().view(news_index.size(0), -1, y.size(-1))  # change size to (N, H, D)
        return y

    def forward(self, input_feat):
        """
        training logic: candidate news encoding->user modeling (history news)
        :param input_feat: should include two keys, candidate and history
        :return: prediction result in softmax possibility
        """
        if "candidate_news" not in input_feat:  # pass news embeddings cache in evaluation stage
            input_feat["candidate_news"] = self.time_distributed(input_feat["candidate"], input_feat["candidate_mask"],
                                                                 uid=input_feat["uid"])  # [N, C, E]
        if "history_news" not in input_feat:  # no cache found
            input_feat["history_news"] = self.time_distributed(input_feat["history"], input_feat["history_mask"],
                                                               uid=input_feat["uid"])  # [N, H, E]
        input_feat["user_embeds"] = self.user_encoder(input_feat)  # user modeling: [N, E]
        return self.predict(input_feat)
