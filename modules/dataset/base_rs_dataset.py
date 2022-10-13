import torch

import numpy as np
import pandas as pd
from pathlib import Path

from torch.utils.data.dataset import Dataset

from collections import OrderedDict
from modules.utils import read_json, news_sampling, Tokenizer, get_project_root, get_mind_dir, check_mind_set
from utils import clean_df


class MindRSDataset(Dataset):
    """Load dataset from file and organize data for training"""

    def __init__(self, tokenizer: Tokenizer, **kwargs):
        # setup some default configurations for MindRSDataset
        self.tokenizer = tokenizer
        self.flatten_article = kwargs.get("flatten_article", True)
        self.phase = kwargs.get("phase", "train")  # RS phase: train, valid, test
        self.history_size = kwargs.get("history_size", 50)
        self.neg_pos_ratio = kwargs.get("neg_pos_ratio", 4)  # negative sampling ratio, default is 20% positive ratio
        self.uid2index = kwargs.get("uid2index", {})  # map user id to index dictionary
        self.train_strategy = kwargs.get("train_strategy", "pair_wise")  # pair wise uses negative sampling strategy
        self.category2id = kwargs.get("category2id", {})  # map category to index dictionary
        self.mind_dir = get_mind_dir(**kwargs)  # get directory of MIND dataset that stores news and behaviors
        check_mind_set(**kwargs)  # check if MIND dataset is ready
        self._load_news_matrix(**kwargs)  # load news matrix
        self._load_behaviors(**kwargs)  # load behavior file, default use mind dataset

    def convert_category(self, cat):
        if cat not in self.category2id:
            self.category2id[cat] = len(self.category2id) + 1
        return self.category2id[cat]

    def _load_news_text(self, **kwargs):
        """
        Load news from news file
        """
        self.news_features = OrderedDict({"article": [""]})  # default use title only, the first article is empty
        self.news_attr = {"article": kwargs.get("article_length", 30)}  # default only use title
        news_file = self.mind_dir / "news.tsv"  # define news file path
        # initial data of corresponding news attributes, such as: title, entity, vert, subvert, abstract
        columns = ["news_id", "category", "subvert", "title", "abstract", "url", "entity", "ab_entity"]
        news_df = pd.read_table(news_file, header=None, names=columns)
        self.nid2index = kwargs.get("nid2index", {})  # map news id to index dictionary
        self.nid2index.update(dict(zip(news_df.news_id, range(1, len(news_df) + 1))))
        if kwargs.get("tokenized_method", "keep_all"):
            article_path = self.mind_dir / "msn.json"
            articles = read_json(article_path)
            news_df["body"] = news_df.news_id.apply(lambda nid: " ".join(articles[nid]) if nid in articles else "")
            news_df = clean_df(news_df)
            news_df["docs"] = news_df["title"] + " " + news_df["abstract"] + " " + news_df["body"]
            self.news_features["article"].extend(news_df.docs.tolist())
        else:
            default_path = Path(get_project_root()) / "dataset/data/MIND_31139.csv"
            tokenized_news_path = Path(kwargs.get("tokenized_news_path", default_path))
            tokenized_news = pd.read_csv(tokenized_news_path)
            tokenized_text = pd.merge(news_df, tokenized_news, on="news_id", how="left")["tokenized_text"].fillna("")
            self.news_features["article"].extend(tokenized_text.tolist())

    def _load_news_matrix(self, **kwargs):
        self._load_news_text(**kwargs)  # load news text from file first before made up news matrix
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
        attributes = ["uid", "impression_index", "history_news", "history_length", "candidate_news", "labels"]
        self.behaviors = {attr: [] for attr in attributes}
        self.positive_news = []
        with open(behaviors_file, "r", encoding="utf-8") as rd:
            imp_index = 0
            for index in rd:
                # read line of behaviors file
                uid, _, history, candidates = index.strip("\n").split("\t")[-4:]
                # deal with behavior data
                history = [self.nid2index[i] for i in history.split() if i in self.nid2index] if len(history) else [0]
                his_length = min(len(history), self.history_size)
                # history = history[:self.history_size]
                history = history[:self.history_size] + [0] * (self.history_size - len(history))
                candidate_news = [self.nid2index[i.split("-")[0]] if i.split("-")[0] in self.nid2index else 0
                                  for i in candidates.split()]
                uindex = self.uid2index[uid] if uid in self.uid2index else 0
                # define attributes value
                behavior = [uindex, imp_index, history, his_length, candidate_news]
                if self.phase != "test":  # load labels for train and validation phase
                    labels = [int(i.split("-")[1]) for i in candidates.split()]
                    self.behaviors["labels"].append(labels)
                if self.phase == "train":  # negative sampling for training
                    pos_news = [(news, imp_index) for label, news in zip(labels, candidate_news) if label]
                    if self.train_strategy == "pair_wise":
                        self.positive_news.extend(pos_news)
                # update to the dictionary
                for attr, value in zip(attributes[:-1], behavior):
                    self.behaviors[attr].append(value)
                imp_index += 1
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
        mask = np.where(input_feat[input_name] == self.tokenizer.pad_id, 0, 1)  # TODO: fix mask bug
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
            "history_length": torch.tensor(len(history), dtype=torch.int32),
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


class UserDataset(Dataset):
    def __init__(self, dataset: MindRSDataset, news_vectors=None):
        self.dataset = dataset
        self.behaviors = dataset.behaviors
        self.news_vectors = news_vectors  # news vectors are in the form of a numpy matrix

    def __getitem__(self, i):
        # get the matrix of corresponding news features with index
        history = self.behaviors["history_news"][i]
        input_feat = {
            "impression_index": torch.tensor(i), "uid": torch.tensor(self.behaviors["uid"][i]),
            # "padding": self.tokenizer.pad_id,  # TODO: pad sentence
            "history_length": torch.tensor(len(history)),
        }
        if self.news_vectors is not None:  # if news vectors are provided, use them
            input_feat["history_news"] = torch.tensor(self.news_vectors[history])
        else:  # otherwise, load from news feat
            input_feat.update(self.dataset.load_news_index(history, "history"))
        return input_feat

    def __len__(self):
        return len(self.behaviors["impression_index"])


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
            "impression_index": torch.tensor(index, dtype=torch.int32),  # index of impression and for user history
            "candidate_length": torch.tensor(len(candidate), dtype=torch.int32),  # length of candidate news
            "history_length": torch.tensor(len(history), dtype=torch.int32),  # length of user history news
            "uid": torch.tensor(self.behaviors["uid"][index], dtype=torch.long)
        }
        if self.dataset.phase != "test":
            input_feat.update({"label": torch.tensor(self.behaviors["labels"][index])})  # load true label of behaviors
        if self.news_embeds is not None:
            input_feat["candidate_news"] = torch.tensor(self.news_embeds[candidate])  # load news embed from cache
            input_feat["history_news"] = torch.tensor(self.news_embeds[history])  # load news embed from cache
        else:
            input_feat.update(self.dataset.load_news_index(candidate, "candidate"))  # load candidate news input
        return input_feat

    def __len__(self):
        return len(self.behaviors["impression_index"])
