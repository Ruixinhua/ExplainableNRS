# setup default configures for models
default_configs = {
    "PretrainedBaseline": {
        "n_layers": 1,
    },
    "TextCNNClassifyModel": {
        "num_filters": 100, "filter_sizes": (2,)
    },
    "NRMSNewsEncoderModel": {
        "variant_name": "base"
    },
    "GRUAttClassifierModel": {
        "variant_name": "gru_att"
    },
    "BiAttentionClassifyModel": {
        "head_num": None, "head_dim": 20, "entropy_constraint": False, "alpha": 0.01, "n_layers": 1,
        "variant_name": "base", "topic_embed": None, "calculate_entropy": True,
    },
    "TopicExtractorClassifyModel": {
        "head_num": None, "head_dim": 20, "entropy_constraint": False, "alpha": 0.01, "n_layers": 1
    },
    "FastformerClassifyModel": {
        "embedding_dim": 300, "n_layers": 2, "hidden_act": "gelu", "head_num": 15, "type_vocab_size": 2,
        "vocab_size": 100000, "layer_norm_eps": 1e-12, "initializer_range": 0.02, "pooler_type": "weightpooler",
        "enable_fp16": "False"
    },
    # MIND RS model
    "MindNRSBase": {
        "out_layer": "product",
    },
    "NRMSRSModel": {
        "out_layer": "product", "head_num": 20, "head_dim": 20, "user_layer": "mha"  # options: mha, gru
    },
    "LSTURRSModel": {
        "kernel_size": 3,
    },
    "BATMRSModel": {
        "variant_name": "base", "head_num": 10, "head_dim": 30
    },
    "KREDRSModel": {
        "head_num": 10, "head_dim": 30
    }
}

# setup default values
default_values = {
    "seeds": [42, 2020, 2021, 25, 4],
    "head_num": [10, 30, 50, 70, 100, 150, 180, 200],
    "embedding_type": ["distilbert-base-uncased", "bert-base-uncased", "roberta-base", "xlnet-base-cased",
                       "allenai/longformer-base-4096", "transfo-xl-wt103"],
    "bert_embedding": ["distilbert-base-uncased", "bert-base-uncased", "roberta-base", "xlnet-base-cased",
                       "allenai/longformer-base-4096", "transfo-xl-wt103"]
}


def arch_default_config(arch_type: str):
    arch_config = {"type": arch_type}
    if arch_type in default_configs:
        arch_config.update(default_configs[arch_type])
    return arch_config
