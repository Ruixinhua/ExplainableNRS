import torch
# import torch.distributed
import news_recommendation.dataset as module_dataset

from collections import defaultdict
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from news_recommendation.config.configuration import Configuration
from news_recommendation.config.default_config import TEST_CONFIGS
from news_recommendation.dataset import NewsDataset, UserDataset, ImpressionDataset, MindRSDataset
from news_recommendation.utils import load_dict, Tokenizer, read_json, get_project_root
from news_recommendation.config.default_config import arch_default_config


def bert_collate_fn(data):
    input_feat = defaultdict(lambda: [])
    max_length = max([feat["history_length"] for feat in data])
    for feat in data:
        for k in feat.keys():
            if k in ["history", "history_mask", "padding"]:
                continue
            input_feat[k].append(feat[k])
        s, p_l = feat["history"].shape, len(feat["padding"])
        padding = torch.tensor([feat["padding"] + [0] * (s[1] - p_l)] * (max_length - s[0]), dtype=torch.long)
        padding_mask = torch.tensor([[1] * p_l + [0] * (s[1] - p_l)] * (max_length - s[0]), dtype=torch.long)
        input_feat["history"].append(torch.cat([feat["history"], padding]))
        input_feat["history_mask"].append(torch.cat([feat["history_mask"], padding_mask]))
    return pad_feat(input_feat)


def pad_feat(input_feat):
    input_pad = {}
    for k, v in input_feat.items():
        try:
            input_pad[k] = pad_sequence(v, batch_first=True)
        except IndexError:
            input_pad[k] = torch.stack(v)
        except AttributeError:
            pass
    return input_pad


def collate_fn(data):
    input_feat = defaultdict(lambda: [])
    for feat in data:
        for k, v in feat.items():
            input_feat[k].append(v)
    return pad_feat(input_feat)


class MindDataLoader:
    def __init__(self, **kwargs):
        # load word and user dictionary
        data_path = kwargs.get("data_path", Path(get_project_root()) / "dataset/MIND")
        self.word_dict = load_dict(Path(data_path) / "utils" / "word_dict.pkl")
        self.word_dict["UNK"] = 0
        uid2index = load_dict(Path(data_path) / "utils" / "uid2index.pkl")
        kwargs.update({"word_dict": self.word_dict, "uid2index": uid2index})
        module_dataset_name = kwargs["dataset_class"] if "dataset_class" in kwargs else "MindRSDataset"
        self.use_dkn_utils = kwargs.get("dkn_utils", None)
        # set tokenizer
        self.tokenizer = Tokenizer(**kwargs)
        bs, sampler = kwargs.get("batch_size", 64), None
        self.fn = bert_collate_fn if self.tokenizer.embedding_type in TEST_CONFIGS["bert_embedding"] else collate_fn
        self.train_set = getattr(module_dataset, module_dataset_name)(self.tokenizer, phase="train", **kwargs)
        # if torch.distributed.is_initialized():
        #     sampler = DistributedSampler(self.train_set)
        self.train_loader = DataLoader(self.train_set, bs, pin_memory=True, sampler=sampler, collate_fn=self.fn)
        # setup news and user dataset
        news_sampler = None
        self.valid_set = getattr(module_dataset, module_dataset_name)(self.tokenizer, phase="valid", **kwargs)
        news_set, user_set = NewsDataset(self.valid_set), UserDataset(self.valid_set)
        impression_set = ImpressionDataset(self.valid_set)
        self.valid_loader = DataLoader(impression_set, 1, pin_memory=True, sampler=sampler, collate_fn=self.fn)
        # if torch.distributed.is_initialized():
        #     news_sampler, sampler = DistributedSampler(news_set), DistributedSampler(user_set)
        self.news_loader = DataLoader(news_set, bs, pin_memory=True, sampler=news_sampler)
        self.user_loader = DataLoader(user_set, bs, pin_memory=True, sampler=sampler, collate_fn=self.fn)

    def set_dataset(self, phase, **kwargs):
        return MindRSDataset(self.tokenizer, phase=phase, **kwargs)


if __name__ == "__main__":
    config = Configuration()
    default_config = read_json(Path(get_project_root()) / "news_recommendation" / "config" / "mind_rs_default.json")
    config.update(default_config)
    config["batch_size"] = 10
    train_loader = MindDataLoader(**config.final_configs).train_loader
    for batch in train_loader:
        break
