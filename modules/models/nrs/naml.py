import torch
import torch.nn as nn
import torch.nn.functional as F

from modules.models.general import AttLayer, Conv1D
from modules.models.nrs.rs_base import MindNRSBase
from modules.utils import load_category_dict


class NAMLRSModel(MindNRSBase):
    def __init__(self, **kwargs):
        self.use_category = True  # use category and subvert for NAML model
        super(NAMLRSModel, self).__init__(**kwargs)
        self.num_filters, self.window_size = kwargs.get("num_filters", 300), kwargs.get("window_size", 3)
        self.news_infos = kwargs.get("news_infos", ["title", "body", "category", "subvert"])
        if "title" in self.news_infos:
            self.news_infos.append("title_mask")
        if "body" in self.news_infos:
            self.news_infos.append("body_mask")
        self.reshape_tensors = [f"candidate_{f}" for f in self.news_infos] + [f"history_{f}" for f in self.news_infos]

        self.title_cnn = Conv1D(self.embedding_dim, self.num_filters, self.window_size)
        self.body_cnn = Conv1D(self.embedding_dim, self.num_filters, self.window_size)
        self.title_att = self.news_att_layer
        self.body_att = AttLayer(self.num_filters, self.attention_hidden_dim)
        self.affine1 = nn.Linear(self.num_filters, self.attention_hidden_dim, bias=True)
        self.affine2 = nn.Linear(self.attention_hidden_dim, 1, bias=False)
        self.category_affine = nn.Linear(self.category_dim, self.num_filters, bias=True)
        self.subvert_affine = nn.Linear(self.category_dim, self.num_filters, bias=True)

    def organize_feat(self, input_feat, **kwargs):
        run_name = kwargs.get("run_name")
        feat = {k: input_feat[f"{run_name}_{k}"] for k in self.news_infos}
        return feat

    def news_encoder(self, input_feat):
        """input_feat contains: news title, body, category, and subvert"""
        # 1. word embedding: [batch_size * news_num, max_news_length, word_embedding_dim]
        title = self.dropouts(self.embedding_layer(news=input_feat["title"], news_mask=input_feat["title_mask"]))
        body = self.dropouts(self.embedding_layer(news=input_feat["body"], news_mask=input_feat["body_mask"]))
        # 2. cnn extract features: [batch_size * news_num, max_news_length, kernel_num]
        title_feature = self.dropouts(self.title_cnn(title.transpose(1, 2)).transpose(1, 2))
        body_feature = self.dropouts(self.body_cnn(body.transpose(1, 2)).transpose(1, 2))
        # 3. attention layer: [batch_size * news_num, kernel_num]
        title_vector = self.title_att(title_feature)[0]
        body_vector = self.body_att(body_feature)[0]
        # 4. category and subcategory encoding: [batch_size * news_num, kernel_num]
        category_vector = F.relu(self.category_affine(self.category_embedding(input_feat["category"])), inplace=True)
        subvert_vector = F.relu(self.subvert_affine(self.subvert_embedding(input_feat["subvert"])), inplace=True)
        # 5. multi-view attention: [batch_size*news_num, 4, kernel_num]
        feature = torch.stack([title_vector, body_vector, category_vector, subvert_vector], dim=1)
        alpha = F.softmax(self.affine2(torch.tanh(self.affine1(feature))), dim=-1)  # [batch_size*news_num, 4, 1]
        news_vector = (feature * alpha).sum(dim=1, keepdim=False)  # [batch_size*news_num, kernel_num]
        return {"news_embed": news_vector, "news_weight": alpha}
