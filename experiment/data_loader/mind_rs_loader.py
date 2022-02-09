import torch
import torch.distributed

from collections import defaultdict
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DistributedSampler
from torch.utils.data.dataloader import DataLoader
from base.mind_rs_dataset import MindRSDataset, NewsDataset, UserDataset, ImpressionDataset
from config.configuration import Configuration
from config.default_config import default_values
from utils import load_dict, Tokenizer, read_json, get_project_root, get_mind_file_path


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
    def __init__(self, data_path, mind_type, **kwargs):
        # load word and user dictionary
        word_dict = load_dict(Path(data_path) / "utils" / "word_dict.pkl")
        uid2index = load_dict(Path(data_path) / "utils" / "uid2index.pkl")
        kwargs.update({"word_dict": word_dict, "uid2index": uid2index})
        # set tokenizer
        self.tokenizer = Tokenizer(**kwargs)
        bs, sampler = kwargs.get("batch_size", 64), None
        fn = bert_collate_fn if self.tokenizer.embedding_type in default_values["bert_embedding"] else collate_fn
        train_news, train_behaviors = get_mind_file_path(data_path, mind_type, "train")
        self.train_set = self.set_dataset(train_news, train_behaviors, "train", **kwargs)
        if torch.distributed.is_initialized():
            sampler = DistributedSampler(self.train_set)
        self.train_loader = DataLoader(self.train_set, bs, pin_memory=True, sampler=sampler, collate_fn=fn)
        # setup news and user dataset
        news_sampler = None
        valid_news, valid_behaviors = get_mind_file_path(data_path, mind_type, "valid")
        self.valid_set = self.set_dataset(valid_news, valid_behaviors, "valid", **kwargs)
        news_set, user_set = NewsDataset(self.valid_set), UserDataset(self.valid_set)
        impression_set = ImpressionDataset(self.valid_set)
        self.valid_loader = DataLoader(impression_set, 1, pin_memory=True, sampler=sampler, collate_fn=fn)
        if torch.distributed.is_initialized():
            news_sampler, sampler = DistributedSampler(news_set), DistributedSampler(user_set)
        self.news_loader = DataLoader(news_set, bs, pin_memory=True, sampler=news_sampler)
        self.user_loader = DataLoader(user_set, bs, pin_memory=True, sampler=sampler, collate_fn=fn)

    def set_dataset(self, news_file, behaviors_file, phase, **kwargs):
        return MindRSDataset(news_file, behaviors_file, self.tokenizer, phase=phase, **kwargs)


if __name__ == "__main__":
    config = Configuration()
    default_config = read_json(Path(get_project_root()) / "config" / "mind_rs_default.json")
    config.update(default_config)
    config.data_config["batch_size"] = 10
    train_loader = MindDataLoader(**config.data_config).train_loader
    for batch in train_loader:
        break
