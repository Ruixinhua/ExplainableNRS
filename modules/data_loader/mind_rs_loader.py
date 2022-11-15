import torch
import modules.dataset as module_dataset

from collections import defaultdict
from pathlib import Path
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from modules.dataset import NewsDataset, UserDataset, ImpressionDataset
from modules.utils import Tokenizer, get_project_root, read_json


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
        # set tokenizer
        self.tokenizer = Tokenizer(**kwargs)
        self.word_dict = self.tokenizer.word_dict
        bs, sampler = kwargs.get("batch_size", 64), None
        self.fn = collate_fn
        module_dataset_name = kwargs.get("dataset_class", "MindRSDataset")
        self.train_set = getattr(module_dataset, module_dataset_name)(self.tokenizer, phase="train", **kwargs)
        self.train_loader = DataLoader(self.train_set, bs, pin_memory=True, sampler=sampler, collate_fn=self.fn)
        # setup news and user dataset
        news_sampler = None
        self.valid_set = getattr(module_dataset, module_dataset_name)(self.tokenizer, phase="valid", **kwargs)
        news_set, user_set = NewsDataset(self.valid_set), UserDataset(self.valid_set)
        impression_set = ImpressionDataset(self.valid_set)
        self.valid_loader = DataLoader(impression_set, 1, pin_memory=True, sampler=sampler, collate_fn=self.fn)
        self.news_loader = DataLoader(news_set, kwargs.get("news_batch_size", 2048), sampler=news_sampler)
        self.user_loader = DataLoader(user_set, bs, sampler=sampler, collate_fn=self.fn)
