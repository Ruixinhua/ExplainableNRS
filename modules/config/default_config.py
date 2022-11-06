import logging
import os


DEFAULT_CONFIGS = {
    "n_gpu": 1, "embedding_type": "glove", "embedding_dim": 300, "max_length": 100, "loss": "cross_entropy",
    "metrics": ["accuracy", "macro_f"], "save_model": True, "resume_path": None, "project_name": "",
    "seed": 42, "arch_type": "BiAttentionClassifyModel", "dropout_rate": 0.2, "dataloader_type": "NewsDataLoader",
    "batch_size": 32, "num_workers": 1, "dataset_name": "News26/keep_all", "trainer_type": "NCTrainer",
    # Trainer parameters
    "epochs": 10, "early_stop": 3, "monitor": "max val_accuracy", "verbosity": 2, "tensorboard": False,
}

LOG_LEVELS = {0: logging.WARNING, 1: logging.INFO, 2: logging.DEBUG}  # configure logging module

LAYER_MAPPING = {"distilbert-base-uncased": "n_layers", "xlnet-base-cased": "n_layer",
                 "bert-base-uncased": "num_hidden_layers", "roberta-base": "num_hidden_layers",
                 "allenai/longformer-base-4096": "num_hidden_layers",
                 "transfo-xl-wt103": "n_layer"}
MAX_LAYERS = {"bert-base-uncased": 12, "distilbert-base-uncased": 6, "allenai/longformer-base-4096": 12,
              "xlnet-base-cased": 12, "roberta-base": 12}

# setup default architecture configures for models
ARCH_CONFIGS = {
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
        "head_num": 20, "head_dim": 20, "entropy_constraint": False, "alpha": 0.01, "n_layers": 1,
        "variant_name": "base", "topic_embed": None, "calculate_entropy": False,
    },
    "TopicExtractorClassifyModel": {
        "head_num": 20, "head_dim": 20, "entropy_constraint": False, "alpha": 0.01, "n_layers": 1
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
        "window_size": 3,
    },
    "BATMRSModel": {
        "variant_name": "base", "head_num": 10, "head_dim": 30
    },
    "KREDRSModel": {
        "head_num": 10, "head_dim": 30, "entity_embedding_dim": 100, "attention_hidden_dim": 64
    },
    "DKNRSModel": {
        "num_filters": 50, "layer_dim": 50, "window_sizes": [2, 3, 4], "entity_embedding_dim": 100,
        "dataset_type": "DKNRSDataset",
    },
    "NPARSModel": {

    }
}

# setup default values
TEST_CONFIGS = {
    "seeds": [42, 2020, 2021, 25, 4],
    "head_num": [10, 30, 50, 70, 100, 150, 180, 200],
    "bert_embedding": ["distilbert-base-uncased", "bert-base-uncased", "roberta-base", "xlnet-base-cased",
                       "allenai/longformer-base-4096", "transfo-xl-wt103"]
}


def arch_default_config(arch_type: str):
    return ARCH_CONFIGS[arch_type] if arch_type in ARCH_CONFIGS else {}
