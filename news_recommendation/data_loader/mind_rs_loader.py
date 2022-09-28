import torch
import news_recommendation.dataset as module_dataset

from collections import defaultdict
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from news_recommendation.dataset import NewsDataset, UserDataset, ImpressionDataset, MindRSDataset
from news_recommendation.utils import Tokenizer, get_project_root, read_json


def pad_feat(input_feat):
    padded_list = {"news_index", "news_mask", "history_index", "history_mask", "candidate_index", "candidate_mask"}
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
        data_root = kwargs.get("data_root", Path(get_project_root()) / "dataset")  # get root of dataset
        uid2index = read_json(kwargs.get("uid_path", Path(data_root) / "utils/MIND_uid_small.json"))
        module_dataset_name = kwargs["dataset_class"] if "dataset_class" in kwargs else "MindRSDataset"
        self.use_dkn_utils = kwargs.get("dkn_utils", None)
        # set tokenizer
        self.tokenizer = Tokenizer(**kwargs)
        self.word_dict = self.tokenizer.word_dict
        kwargs.update({"word_dict": self.word_dict, "uid2index": uid2index})
        bs, sampler = kwargs.get("batch_size", 64), None
        self.fn = collate_fn
        self.train_set = getattr(module_dataset, module_dataset_name)(self.tokenizer, phase="train", **kwargs)
        self.train_loader = DataLoader(self.train_set, bs, pin_memory=True, sampler=sampler, collate_fn=self.fn)
        # setup news and user dataset
        news_sampler = None
        self.valid_set = getattr(module_dataset, module_dataset_name)(self.tokenizer, phase="valid", **kwargs)
        news_set, user_set = NewsDataset(self.valid_set), UserDataset(self.valid_set)
        impression_set = ImpressionDataset(self.valid_set)
        self.valid_loader = DataLoader(impression_set, 1, pin_memory=True, sampler=sampler, collate_fn=self.fn)
        news_batch_size = kwargs.get("news_batch_size", 2048)
        self.news_loader = DataLoader(news_set, news_batch_size, sampler=news_sampler)
        self.user_loader = DataLoader(user_set, bs, sampler=sampler, collate_fn=self.fn)

    def set_dataset(self, phase, **kwargs):
        return MindRSDataset(self.tokenizer, phase=phase, **kwargs)
