import torch
import modules.dataset as module_dataset

from collections import defaultdict
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data.dataloader import DataLoader
from modules.dataset import NewsDataset, ImpressionDataset
from modules.utils import Tokenizer


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


class NRDataLoader:
    def __init__(self, **kwargs):
        # load word and user dictionary
        # set tokenizer
        self.tokenizer = Tokenizer(**kwargs)
        bs = kwargs.get("batch_size", 64)
        impression_bs = kwargs.get("impression_batch_size", 1)
        self.fn = collate_fn
        module_dataset_name = kwargs.get("dataset_class", "NewsRecDataset")
        self.train_set = getattr(module_dataset, module_dataset_name)(self.tokenizer, phase="train", **kwargs)
        self.train_loader = DataLoader(self.train_set, bs, pin_memory=True, collate_fn=self.fn)
        # setup news and user dataset
        self.valid_set = getattr(module_dataset, module_dataset_name)(self.tokenizer, phase="valid", **kwargs)
        self.test_set = getattr(module_dataset, module_dataset_name)(self.tokenizer, phase="test", **kwargs)
        news_set = NewsDataset(self.train_set)
        self.news_loader = DataLoader(news_set, kwargs.get("news_batch_size", 128))
        self.valid_loader = DataLoader(ImpressionDataset(self.valid_set), impression_bs, collate_fn=self.fn)
        self.word_dict = self.tokenizer.word_dict
