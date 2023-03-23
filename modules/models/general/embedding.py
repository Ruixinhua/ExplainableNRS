import torch
from torch import nn

from modules.utils import load_embeddings
from modules.config import LAYER_MAPPING


class NewsEmbedding(nn.Module):
    def __init__(self, **kwargs):
        super(NewsEmbedding, self).__init__()
        self.embedding_type = kwargs.get("embedding_type", "glove")
        self.embed_dim = kwargs.get("embed_dim", 300)
        if self.embedding_type == "glove":
            self.embeds = load_embeddings(**kwargs)
            self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(self.embeds), freeze=False)
            self.embed_dim = self.embeds.shape[1]
        elif self.embedding_type == "init":
            self.word_dict = kwargs.get("word_dict", None)
            assert self.word_dict is not None, "Please provide word dictionary for embedding initialization"
            self.embedding = nn.Embedding(len(self.word_dict), self.embed_dim, padding_idx=0)
        elif self.embedding_type in LAYER_MAPPING:
            # load weight and model from pretrained model
            from transformers import AutoConfig, AutoModel
            self.output_hidden_states = kwargs.get("output_hidden_states", True)
            self.return_attention = kwargs.get("output_attentions", True)
            self.n_layers = kwargs.get("n_layers", 1)
            self.num_classes = kwargs.get("num_classes", 15)
            self.embed_config = AutoConfig.from_pretrained(self.embedding_type, num_labels=self.num_classes,
                                                           output_hidden_states=self.output_hidden_states,
                                                           output_attentions=self.return_attention)
            add_weight = kwargs.get("add_weight", False)
            layer_name = LAYER_MAPPING.get(self.embedding_type, "n_layers")
            self.embed_config.__dict__.update({"add_weight": add_weight, layer_name: self.n_layers})
            if self.embedding_type == "allenai/longformer-base-4096":
                self.embed_config.attention_window = self.embed_config.attention_window[:self.n_layers]
            embedding = AutoModel.from_pretrained(self.embedding_type, config=self.embed_config)
            self.embedding = kwargs.get("bert")(self.embed_config) if "bert" in kwargs else embedding
            if hasattr(self.embed_config, "dim"):  # for roberta like language model
                self.embed_dim = self.embed_config.dim
            elif hasattr(self.embed_config, "hidden_size"):  # for bert like language model
                self.embed_dim = self.embed_config.hidden_size
            else:
                raise ValueError("Unsure the embedding dimension, please check the config of the model")
        else:
            raise ValueError("Unknown embedding type")

    def forward(self, news, news_mask, **kwargs):
        if self.embedding_type in ["glove", "init"]:
            embedding = self.embedding(news)
        else:  # for bert like language model
            embedding = kwargs.get("embedding", None)
            output = self.embedding(news, news_mask, inputs_embeds=embedding)
            embedding = output[0]
        return embedding
