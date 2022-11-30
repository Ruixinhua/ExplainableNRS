import os

import torch

import numpy as np
import pandas as pd
from pathlib import Path

from torch.utils.data.dataset import Dataset

from collections import OrderedDict
from utils import read_json, news_sampling, Tokenizer, get_mind_dir, check_mind_set, write_json, get_project_root


class MindRSDataset(Dataset):
    """Load dataset from file and organize data for training"""

    def __init__(self, tokenizer: Tokenizer, **kwargs):
        # setup some default configurations for MindRSDataset
        self.tokenizer = tokenizer
        self.flatten_article = kwargs.get("flatten_article", True)
        self.phase = kwargs.get("phase", "train")  # RS phase: train, valid, test
        self.history_size = kwargs.get("history_size", 50)
        self.neg_pos_ratio = kwargs.get("neg_pos_ratio", 4)  # negative sampling ratio, default is 20% positive ratio
        self.mind_type = kwargs.get("mind_type", "small")
        data_root = Path(kwargs.get("data_dir", os.path.join(get_project_root(), "dataset")))  # get root of dataset
        default_uid_path = Path(data_root) / f"utils/MIND_uid_{self.mind_type}.json"
        self.uid2index = read_json(kwargs.get("uid_path", default_uid_path))
        # TODO: keep_all news text for PLM
        self.news_features = OrderedDict({"article": [""]})  # default use title only, the first article is empty
        self.news_attr = {"article": kwargs.get("article_length", 30)}  # default only use title
        news_path = data_root / f"data/MIND_{self.mind_type}_original.csv"
        tokenized_news_path = Path(kwargs.get("tokenized_news_path", news_path))
        tokenized_news = pd.read_csv(tokenized_news_path)
        self.news_features["article"].extend(tokenized_news["tokenized_text"].tolist())
        default_nid_path = Path(data_root) / f"utils/MIND_nid_{kwargs.get('mind_type')}.json"
        nid_path = kwargs.get("nid_path", default_nid_path)  # get nid path from kwargs
        if nid_path is not None and os.path.exists(nid_path):
            self.nid2index = read_json(nid_path)
        else:
            self.nid2index = dict(zip(tokenized_news.news_id, range(1, len(tokenized_news) + 1)))
            write_json(self.nid2index, str(nid_path))
        self.train_strategy = kwargs.get("train_strategy", "pair_wise")  # pair wise uses negative sampling strategy
        self.category2id = kwargs.get("category2id", {})  # map category to index dictionary
        self.mind_dir = get_mind_dir(**kwargs)  # get directory of MIND dataset that stores news and behaviors
        # check_mind_set(**kwargs)  # check if MIND dataset is ready
        self._load_news_matrix(**kwargs)  # load news matrix
        self._load_behaviors(**kwargs)  # load behavior file, default use mind dataset

    def convert_category(self, cat):
        if cat not in self.category2id:
            self.category2id[cat] = len(self.category2id) + 1
        return self.category2id[cat]

    def _load_news_matrix(self, **kwargs):
        self.feature_matrix = OrderedDict({  # init news text matrix
            k: np.stack(self.tokenizer.tokenize(news_text, self.news_attr[k], return_tensors=False))
            for k, news_text in self.news_features.items()
        })  # the order of news info: title(abstract), category, sub-category, sentence embedding, entity feature

    def _load_behaviors(self, **kwargs):
        """"
        Create global variants for behaviors:
        behaviors: The behaviors of the user, includes: history clicked news, impression news, and labels
        positive_news: The index of news that clicked by users in impressions
        """
        behaviors_file = self.mind_dir / "behaviors.tsv"  # define behavior file path
        attributes = ["uid", "index", "impression_index", "history_news", "history_length", "candidate_news", "labels"]
        self.behaviors = {attr: [] for attr in attributes}
        self.positive_news = []
        with open(behaviors_file, "r", encoding="utf-8") as rd:
            index = 0
            for behavior in rd:
                # read line of behaviors file
                imp_index, uid, _, history, candidates = behavior.strip("\n").split("\t")
                # deal with behavior data
                history = [self.nid2index[i] for i in history.split() if i in self.nid2index] if len(history) else [0]
                his_length = min(len(history), self.history_size)
                # history = history[:self.history_size]
                history = history[:self.history_size] + [0] * (self.history_size - len(history))
                candidate_news = [self.nid2index[i.split("-")[0]] if i.split("-")[0] in self.nid2index else 0
                                  for i in candidates.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0
                # define attributes value
                behavior = [uindex, index, int(imp_index), history, his_length, candidate_news]
                if self.phase != "final_test":  # load labels for train and validation phase
                    labels = [int(i.split("-")[1]) for i in candidates.split()]
                    self.behaviors["labels"].append(labels)
                if self.phase == "train":  # negative sampling for training
                    pos_news = [(news, index) for label, news in zip(labels, candidate_news) if label]
                    if self.train_strategy == "pair_wise":
                        self.positive_news.extend(pos_news)
                # update to the dictionary
                for attr, value in zip(attributes[:-1], behavior):
                    self.behaviors[attr].append(value)
                index += 1
        rd.close()

    def load_news_index(self, indices, input_name):
        """

        :param indices: List or integer, index of news information
        :param input_name: corresponding saved name of feat
        :return: a default dictionary with keys called name.
        """
        # get the matrix of corresponding news features with index
        news = [self.feature_matrix[k][indices] for k in self.feature_matrix.keys()]
        input_feat = {
            input_name: torch.tensor(np.concatenate(news, axis=-1), dtype=torch.long),
            f"{input_name}_index": torch.tensor(indices, dtype=torch.long),
        }
        # pass news mask to the model
        mask = np.where(input_feat[input_name] == self.tokenizer.pad_id, 0, 1)
        input_feat[f"{input_name}_mask"] = torch.tensor(mask, dtype=torch.int32)
        return input_feat

    def __getitem__(self, index):
        """
        Return the data that used for training
        :param index:
        :return: A default dictionary includes
        - history information: index, length, news content, news mask
        - candidate information: index, length, news content, news mask
        - padding data
        - label of candidate news
        """
        can_news, imp_index = self.positive_news[index]
        user_index, history = self.behaviors["uid"][imp_index], self.behaviors["history_news"][imp_index]
        if self.train_strategy == "pair_wise":  # negative sampling
            labels, candidate_news = self.behaviors["labels"][imp_index], self.behaviors["candidate_news"][imp_index]
            negs = [news for news, click in zip(candidate_news, labels) if not click]
            label = [1] + [0] * self.neg_pos_ratio
            candidate = [can_news] + news_sampling(negs, self.neg_pos_ratio)
        else:
            candidate = [can_news]
            label = [1]
        # define input feat
        input_feat = {
            "history_length": torch.tensor(self.behaviors["history_length"][imp_index], dtype=torch.int32),
            # TODO padding sentence
            # "padding": torch.tensor(self.tokenizer.tokenize("", self.self.news_attr["article"]), dtype=torch.long),
            # "pad_id": torch.tensor(self.tokenizer.pad_id, dtype=torch.int32),
            "label": torch.tensor(label, dtype=torch.long), "uid": torch.tensor(user_index, dtype=torch.long),
        }
        input_feat.update(self.load_news_index(history, "history"))
        input_feat.update(self.load_news_index(candidate, "candidate"))
        return input_feat

    def __len__(self):
        return len(self.positive_news)


class NewsDataset(Dataset):
    def __init__(self, dataset: MindRSDataset):
        self.dataset = dataset

    def __getitem__(self, i):
        input_feat = {"index": torch.tensor(i)}
        input_feat.update(self.dataset.load_news_index(i, "news"))
        return input_feat

    def __len__(self):
        return len(self.dataset.feature_matrix[list(self.dataset.feature_matrix.keys())[0]])


class ImpressionDataset(Dataset):
    def __init__(self, dataset: MindRSDataset, news_embeds=None):
        self.dataset = dataset
        self.behaviors = dataset.behaviors
        self.news_embeds = news_embeds  # news embeddings (numpy matrix)

    def __getitem__(self, index):
        candidate = self.behaviors["candidate_news"][index]
        history = self.behaviors["history_news"][index]
        input_feat = {
            "impression_index": torch.tensor(self.behaviors["impression_index"][index], dtype=torch.int32),  # index
            "candidate_length": torch.tensor(len(candidate), dtype=torch.int32),  # length of candidate news
            "history_length": torch.tensor(self.behaviors["history_length"][index], dtype=torch.int32),
            "uid": torch.tensor(self.behaviors["uid"][index], dtype=torch.long)
        }
        if self.dataset.phase != "final_test":
            input_feat.update({"label": torch.tensor(self.behaviors["labels"][index])})  # load true label of behaviors
        if self.news_embeds is not None:
            input_feat["candidate_news"] = torch.tensor(self.news_embeds[candidate])  # load news embed from cache
            input_feat["history_news"] = torch.tensor(self.news_embeds[history])  # load news embed from cache
        else:
            input_feat.update(self.dataset.load_news_index(candidate, "candidate"))  # load candidate news input
            input_feat.update(self.dataset.load_news_index(history, "history"))  # load candidate news input
        return input_feat

    def __len__(self):
        return len(self.behaviors["impression_index"])
