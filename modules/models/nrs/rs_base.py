import torch
import torch.nn as nn

from modules.base.base_model import BaseModel
from modules.models.general import AttLayer, DotProduct, DNNClickPredictor, NewsEmbedding
from modules.utils import reshape_tensor, load_category, get_news_info


class MindNRSBase(BaseModel):
    """Basic model for News Recommendation for MIND dataset"""
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.attention_hidden_dim = kwargs.get("attention_hidden_dim", 200)
        self.return_weight = kwargs.get("return_weight", False)
        self.with_entropy = kwargs.get("with_entropy", True if kwargs.get("alpha", 0) > 0 else False)
        self.show_entropy = kwargs.get("show_entropy", False)
        self.topic_variant = kwargs.get("topic_variant", "base")
        self.use_uid = kwargs.get("use_uid", False)
        self.reshape_tensors = []
        self.use_category, self.use_subvert = kwargs.get("use_category", False), kwargs.get("use_subvert", False)
        self.news_info = get_news_info(**kwargs)
        for attr in self.news_info:
            if attr not in ["title", "abstract", "body", "use_all"]:
                self.news_info.remove(attr)
        if "title" in self.news_info:
            self.news_info.append("title_mask")
        if "body" in self.news_info:
            self.news_info.append("body_mask")
        self.category_dim = kwargs.get("category_dim", 100)
        if self.use_category:
            self.news_info.append("category")
            category2id = load_category(cat_type="category", **kwargs),
            self.category_embedding = nn.Embedding(num_embeddings=len(category2id) + 1, embedding_dim=self.category_dim)
        if self.use_subvert:
            self.news_info.append("subvert")
            subvert2id = load_category(cat_type="subvert", **kwargs)
            self.subvert_embedding = nn.Embedding(num_embeddings=len(subvert2id) + 1, embedding_dim=self.category_dim)
        for feature in self.news_info:
            if feature == "use_all":
                self.reshape_tensors.extend(["candidate", "candidate_mask", "history", "history_mask"])
            else:
                self.reshape_tensors.extend([f"candidate_{feature}", f"history_{feature}"])
        # initialize model components
        self.embedding_layer = NewsEmbedding(**kwargs)
        self.embedding_dim = self.embedding_layer.embed_dim
        self.news_att_layer = AttLayer(self.embedding_dim, self.attention_hidden_dim)
        self.user_att_layer = AttLayer(self.embedding_dim, self.attention_hidden_dim)
        self.dropouts = nn.Dropout(self.dropout_rate)
        if self.out_layer == "product":
            self.click_predictor = DotProduct()
        else:
            self.click_predictor = DNNClickPredictor(self.document_embedding_dim * 2, self.attention_hidden_dim)

    def news_encoder(self, input_feat):
        """
        input_feat["news"]: [N * H, S],
        default only use title, and body is added after title.
        Dim S contains: title(30) + body(100) + document_embed(300/768) + entity_feature(4*entity_num)
        """
        y = self.embedding_layer(**input_feat)
        y = self.news_att_layer(y)
        return {"news_embed": y[0], "news_weight": y[1]}

    def user_encoder(self, input_feat):
        y = self.user_att_layer(input_feat["history_news"])
        return {"user_embed": y[0], "user_weight": y[1]}

    def predict(self, input_feat, **kwargs):
        """
        prediction logic: use MLP for prediction or Dot-product.
        :param input_feat: should include encoded candidate news and user representations (history_news)
        :return: softmax possibility of click candidate news
        """
        candidate_news, user_embed = input_feat["candidate_news"], input_feat["user_embed"]
        if self.out_layer == "mlp" and len(candidate_news.shape) != len(user_embed.shape):
            user_embed = torch.unsqueeze(user_embed, 1).expand([user_embed.shape[0], candidate_news.shape[1], -1])
        pred = self.click_predictor(candidate_news, user_embed)
        return pred

    def organize_feat(self, input_feat, **kwargs):
        """Fetch features from reshaped input_feat and pass them to the news encoder"""
        run_name = kwargs.get("run_name")
        feat = {"run_name": run_name, "batch_size": input_feat["uid"].shape[0]}
        for feature in self.news_info:
            if feature == "use_all":
                feat.update({"news": input_feat[run_name], "news_mask": input_feat[f"{run_name}_mask"]})
            else:
                feat[feature] = input_feat[f"{run_name}_{feature}"]
        if "news" not in feat:
            feat.update({"news": feat[self.news_info[0]], "news_mask": feat[f"{self.news_info[1]}"]})
        if self.use_uid:
            news_num = int(feat["news"].shape[0] / input_feat["uid"].shape[0])
            feat["uid"] = input_feat["uid"].unsqueeze(1).expand((-1, news_num)).reshape(-1)
        if self.use_category:
            feat["category"] = input_feat[f"{run_name}_category"]
            feat["subvert"] = input_feat[f"{run_name}_subvert"]
        return feat

    def run_news_encoder(self, input_feat, run_name, **kwargs):
        feat = self.organize_feat(input_feat, run_name=run_name)
        news_dict = self.news_encoder(feat)
        batch_size = kwargs.get("batch_size", input_feat["label"].size(0))
        news_shape = kwargs.get("news_shape", (batch_size, -1, news_dict["news_embed"].size(-1)))
        input_feat[f"{run_name}_news"] = reshape_tensor(news_dict["news_embed"], output_shape=news_shape)
        if "topic_weight" in news_dict:
            shape = (batch_size, -1, news_dict["topic_weight"].size(-2), news_dict["topic_weight"].size(-1))
            news_dict["topic_weight"] = reshape_tensor(news_dict["topic_weight"], output_shape=shape)
        if self.return_weight:
            weight_shape = kwargs.get("weight_shape", (batch_size, -1, news_dict["news_weight"].size(1)))
            input_feat[f"{run_name}_weight"] = reshape_tensor(news_dict["news_weight"], output_shape=weight_shape)
            if "topic_weight" in news_dict:
                input_feat[f"{run_name}_topic_weight"] = news_dict["topic_weight"]
        if self.with_entropy or self.show_entropy:
            topic_weight = news_dict["topic_weight"]
            entropy = torch.mean(torch.sum(-topic_weight * torch.log2(1e-9 + topic_weight), dim=-1))
            if "entropy" not in input_feat:
                input_feat["entropy"] = entropy
                input_feat["entropy_num"] = 1
            else:
                input_feat["entropy"] = input_feat["entropy"] + entropy
                input_feat["entropy_num"] += 1
        if self.topic_variant == "variational_topic":
            if "kl_divergence" not in input_feat:
                input_feat["kl_divergence"] = news_dict["kl_divergence"]
            else:
                input_feat["kl_divergence"] += news_dict["kl_divergence"]
        return input_feat

    def acquire_news_dict(self, input_feat):
        for tensor_name in self.reshape_tensors:
            if tensor_name in input_feat:
                input_feat[tensor_name] = reshape_tensor(input_feat[tensor_name])
        if "candidate_news" not in input_feat or self.return_weight or self.with_entropy:
            # pass news embeddings cache or return weight
            input_feat = self.run_news_encoder(input_feat, "candidate")
            # output candidate news embedding shape: [N, C, E], C is the number of candidate news
        if "history_news" not in input_feat or self.return_weight or self.with_entropy:  # no cache found
            input_feat = self.run_news_encoder(input_feat, "history")
            # output history news embedding shape: [N, H, E], H is the number of history news
        return input_feat

    def forward(self, input_feat):
        """
        training logic: candidate news encoding->user modeling (history news)
        :param input_feat: should include two keys, candidate and history
        :return: prediction result in softmax possibility
        """
        input_feat = self.acquire_news_dict(input_feat)
        user_dict = self.user_encoder(input_feat)
        input_feat["user_embed"] = user_dict["user_embed"]
        out_dict = {"pred": self.predict(input_feat)}
        if self.return_weight:
            out_dict["user_weight"] = user_dict["user_weight"].squeeze(-1)
            for name, values in input_feat.items():
                if name.endswith("_weight"):
                    out_dict[name] = values
        if self.with_entropy or self.show_entropy:
            out_dict["entropy"] = input_feat["entropy"] / input_feat["entropy_num"] if "entropy" in input_feat else None
        if self.topic_variant == "variational_topic":
            out_dict["kl_divergence"] = input_feat["kl_divergence"]
        return out_dict
