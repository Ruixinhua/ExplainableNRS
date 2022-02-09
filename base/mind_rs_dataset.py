import os
from collections import OrderedDict

import numpy as np
import torch
import torch.distributed
from torch.utils.data.dataset import Dataset
from pathlib import Path

from config.default_config import default_values
from utils import read_json, news_sampling, Tokenizer
from utils.mind_untils import load_entity


class MindRSDataset(Dataset):
    """Load dataset from file and organize data for training"""

    def __init__(self, news_file, behaviors_file, tokenizer: Tokenizer, **kwargs):
        # setup some default configurations for MindRSDataset
        self.tokenizer = tokenizer
        self.flatten_article = kwargs.get("flatten_article", True)
        self.phase = kwargs.get("phase", "train")  # RS phase: train, valid, test
        self.history_size = kwargs.get("history_size", 50)
        self.neg_pos_ratio = kwargs.get("neg_pos_ratio", 4)  # negative sampling ratio, default is 20% positive ratio
        self.uid2index = kwargs.get("uid2index", {})  # get user id to index dictionary
        self.train_strategy = kwargs.get("train_strategy", "pair_wise")  # pair wise uses negative sampling strategy
        self.news_attr = OrderedDict(kwargs.get("news_attr", {"title": 30}))  # shape of news attributes in order
        # initial data of corresponding news attributes, such as: title, entity, vert, subvert, abstract
        self.news_text = OrderedDict({attr: [""] for attr in self.news_attr.keys()})  # default use title only
        # load news articles text from a json file
        body_file = Path(os.path.dirname(news_file)) / "msn.json"
        self.use_body = "body" in self.news_text and body_file.exists()  # boolean value for use article or not
        self.news_articles = read_json(body_file) if self.use_body else None  # read articles from file
        if self.use_body and not self.flatten_article:
            self.news_text["body"] = [[""]]  # setup default article data format not using flatten articles
        self.nid2index = {}

        self._load_news(news_file)  # load news from file and save to news_text object
        self.news_matrix = OrderedDict({
            k: np.stack([self.tokenizer.tokenize(news, self.news_attr[k]) for news in news_text])
            for k, news_text in self.news_text.items()
        })
        self._load_behaviors(behaviors_file)

    def _load_news(self, news_file):
        """
        Load news from news file
        """
        with open(news_file, "r", encoding="utf-8") as rd:
            for text in rd:
                # news id, category, subcategory, title, abstract, url
                nid, vert, subvert, title, ab, url, title_entity, abs_entity = text.strip("\n").split("\t")
                entities = load_entity(title_entity) if len(title_entity) > 2 else load_entity(abs_entity)
                entities = entities if len(entities) else title
                news_dict = {"title": title, "entity": entities, "vert": vert, "subvert": subvert, "abstract": ab}
                if nid in self.nid2index:
                    continue
                if self.use_body:  # add news body
                    if nid not in self.news_articles or self.news_articles[nid] is None:  # article not found or none
                        article = "" if self.flatten_article else [""]
                    else:
                        article = " ".join(self.news_articles[nid]) if self.flatten_article else self.news_articles[nid]
                    news_dict["body"] = article
                # add news attribution
                for attr in self.news_text.keys():
                    if attr in news_dict:
                        self.news_text[attr].append(news_dict[attr])
                self.nid2index[nid] = len(self.nid2index) + 1
        rd.close()

    def _load_behaviors(self, behaviors_file, col_spl="\t"):
        """"
        Create global variants for behaviors:
        behaviors: The behaviors of the user, includes: history clicked news, impression news, and labels
        positive_news: The index of news that clicked by users in impressions
        """
        # initial behaviors attributes
        attributes = ["uid", "impression_index", "history_news", "history_length", "candidate_news", "labels"]
        self.behaviors = {attr: [] for attr in attributes}
        self.positive_news = []
        with open(behaviors_file, "r", encoding="utf-8") as rd:
            imp_index = 0
            for index in rd:
                # read line of behaviors file
                uid, time, history, candidates = index.strip("\n").split(col_spl)[-4:]
                # deal with behavior data
                history = [self.nid2index[i] for i in history.split()] if len(history) > 1 else [0]  # TODO history
                his_length = min(len(history), self.history_size)
                # history = history[:self.history_size]
                history = [0] * (self.history_size - len(history)) + history[:self.history_size]
                candidate_news = [self.nid2index[i.split("-")[0]] for i in candidates.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0
                # define attributes value
                behavior = [uindex, imp_index, history, his_length, candidate_news]
                if self.phase != "test":  # load labels for train and validation phase
                    labels = [int(i.split("-")[1]) for i in candidates.split()]
                    self.behaviors["labels"].append(labels)
                if self.phase == "train":  # negative sampling for training
                    pos_news = [(news, imp_index, label) for label, news in zip(labels, candidate_news) if label]
                    if self.train_strategy == "pair_wise":
                        self.positive_news.extend(pos_news)
                # update to the dictionary
                for attr, value in zip(attributes[:-1], behavior):
                    self.behaviors[attr].append(value)
                imp_index += 1
        rd.close()

    def load_news_feat(self, indices, name):
        """

        :param indices: index of news
        :param name: corresponding name of feat
        :return: a default dictionary with keys called name.
        """
        # get the matrix of corresponding news features with index
        news = [self.news_matrix[k][indices] for k in self.news_matrix.keys()]
        input_feat = {name: torch.tensor(np.concatenate(news, axis=-1), dtype=torch.long),
                      f"{name}_index": torch.tensor(indices, dtype=torch.long)}
        if self.tokenizer.embedding_type in default_values["bert_embedding"]:
            # pass news mask to the model
            input_feat[f"{name}_mask"] = torch.tensor(np.where(input_feat[name] == 0, 0, 1), dtype=torch.int8)
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
        can_news, imp_index, label = self.positive_news[index]
        user_index, history = self.behaviors["uid"][imp_index], self.behaviors["history_news"][imp_index]
        if self.train_strategy == "pair_wise":  # negative sampling
            labels, candidate_news = self.behaviors["labels"][imp_index], self.behaviors["candidate_news"][imp_index]
            negs = [news for news, click in zip(candidate_news, labels) if not click]
            label = [1] + [0] * self.neg_pos_ratio
            candidate = [can_news] + news_sampling(negs, self.neg_pos_ratio)
        else:
            candidate = [can_news]
        # define input feat
        input_feat = {
            "history_length": torch.tensor(len(history), dtype=torch.int8),
            # "padding": torch.tensor(self.tokenizer.tokenize("", 0), dtype=torch.int8),  # TODO padding sentence
            "label": torch.tensor(label, dtype=torch.long),
        }
        input_feat.update(self.load_news_feat(history, "history"))
        input_feat.update(self.load_news_feat(candidate, "candidate"))
        return input_feat

    def __len__(self):
        return len(self.positive_news)


class UserDataset(Dataset):
    def __init__(self, dataset: MindRSDataset):
        self.dataset = dataset
        self.behaviors = dataset.behaviors

    def __getitem__(self, i):
        # get the matrix of corresponding news features with index
        history_index = self.behaviors["history_news"][i]
        input_feat = self.dataset.load_news_feat(history_index, "history")
        input_feat.update({
            "impression_index": torch.tensor(i), "uid": torch.tensor(self.behaviors["uid"][i]),
            # "padding": self.tokenizer.pad_id,  # TODO: pad sentence
            "history_length": torch.tensor(len(history_index))
        })
        return input_feat

    def __len__(self):
        return len(self.behaviors["impression_index"])


class NewsDataset(Dataset):
    def __init__(self, dataset: MindRSDataset):
        self.dataset = dataset

    def __getitem__(self, i):
        input_feat = {"index": torch.tensor(i)}
        input_feat.update(self.dataset.load_news_feat(i, "news"))
        return input_feat

    def __len__(self):
        return len(self.dataset.news_text[list(self.dataset.news_text.keys())[0]])


class ImpressionDataset(Dataset):
    def __init__(self, dataset: MindRSDataset):
        self.dataset = dataset
        self.behaviors = dataset.behaviors

    def __getitem__(self, index):
        candidate = self.behaviors["candidate_news"][index]
        input_feat = {"impression_index": torch.tensor(index), "candidate_index": torch.tensor(candidate)}
        if self.dataset.phase != "test":
            input_feat.update({"label": torch.tensor(self.behaviors["labels"][index])})
        return input_feat

    def __len__(self):
        return len(self.behaviors["impression_index"])
