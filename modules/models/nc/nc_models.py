import torch
import torch.nn as nn
from transformers import AutoConfig, AutoModelForSequenceClassification

from modules.config import LAYER_MAPPING, MAX_LAYERS
from modules.models.general import NewsEmbedding
from modules.base.base_model import BaseModel


class BaseClassifyModel(BaseModel):
    def __init__(self, **kwargs):
        super().__init__()
        self.__dict__.update(kwargs)
        self.att_weight = None
        self.dropout_rate = kwargs.get("dropout_rate", 0)  # default without using dropout
        self.dropout = nn.Dropout(self.dropout_rate)
        self.num_classes = kwargs.get("num_classes", 15)
        self.embedding_layer = NewsEmbedding(**kwargs)
        self.embed_dim = self.embedding_layer.embed_dim
        self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        self.return_attention = kwargs.get("return_attention", False)

    def classify_layer(self, latent, weight=None, **kwargs):
        return_attention = kwargs.get("return_attention", self.return_attention)
        output = {"predicted": self.classifier(latent)}
        if return_attention:
            output["attention"] = weight
        return output

    def forward(self, input_feat, **kwargs):
        input_feat["embedding"] = input_feat.get("embedding", kwargs.get("inputs_embeds"))
        embedding = self.dropout(self.embedding_layer(input_feat))
        if self.embedding_type == "glove" or self.embedding_type == "init":
            embedding = torch.mean(embedding, dim=1)
        else:
            embedding = embedding[0][:, 0]  # shape of last hidden: (N, L, D), take the CLS for classification
        return self.classify_layer(embedding, self.att_weight)


class PretrainedBaseline(BaseModel):
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        super(PretrainedBaseline, self).__init__()
        self.use_pretrained = kwargs.get("use_pretrained", True)
        config = AutoConfig.from_pretrained(self.embedding_type, num_labels=self.num_classes)
        n_layers = min(self.n_layers, MAX_LAYERS[self.embedding_type])
        if self.embedding_type == "allenai/longformer-base-4096":
            config.attention_window = config.attention_window[:n_layers]
        config.__dict__.update({LAYER_MAPPING[self.embedding_type]: n_layers, "pad_token_id": 0})
        if self.use_pretrained:
            self.model = AutoModelForSequenceClassification.from_pretrained(self.embedding_type, config=config)
        else:
            self.model = AutoModelForSequenceClassification.from_config(config=config)

    def forward(self, input_feat, **kwargs):
        feat_dict = {"input_ids": input_feat["news"], "attention_mask": input_feat["news_mask"]}
        if self.embedding_type == "transfo-xl-wt103":
            outputs = self.model(input_feat["news"])
        else:
            outputs = self.model(**feat_dict)
        outputs = (outputs.logits,)
        return outputs
