import os

import torch

import numpy as np
import pandas as pd
from pathlib import Path

from torch.utils.data.dataset import Dataset

from collections import OrderedDict
from utils import read_json, news_sampling, Tokenizer, get_mind_dir, write_json, get_project_root, load_category_dict


class MindRSDataset(Dataset):
    """Load dataset from file and organize data for training"""

    def __init__(self, tokenizer: Tokenizer, **kwargs):
        # setup some default configurations for MindRSDataset
        self.tokenizer = tokenizer
        self._init_vars(**kwargs)
        self.news_df = pd.read_csv(self.data_root / f"MIND/{self.mind_type}/news.csv")
        self.news_df.fillna("", inplace=True)
        self.news_df["use_all"] = self.news_df["title"] + " " + self.news_df["abstract"] + " " + self.news_df["body"]
        default_nid_path = Path(self.data_root) / f"utils/MIND_nid_{kwargs.get('mind_type')}.json"
        nid_path = kwargs.get("nid_path", default_nid_path)  # get nid path from kwargs
        if nid_path is not None and os.path.exists(nid_path):
            self.nid2index = read_json(nid_path)
        else:
            self.nid2index = dict(zip(self.news_df.news_id, range(1, len(self.news_df) + 1)))
            write_json(self.nid2index, str(nid_path))
        self.news_df["index"] = self.news_df["news_id"].apply(lambda x: self.nid2index[x])
        self.news_features = OrderedDict({k: [""] * (len(self.nid2index)+1) for k in self.news_info})
        for attr in self.news_info:
            self.append_feature(attr)
        self.feature_matrix = OrderedDict({  # init news text matrix
            k: np.stack(self.tokenizer.tokenize(news_text, self.news_attr[k], return_tensors=False))
            for k, news_text in self.news_features.items()
        })  # tokenize news text (title, abstract, body, all) and convert to numpy array

        self._load_category(**kwargs)  # load category information
        self._load_behaviors(**kwargs)  # load behavior file, default use mind dataset

    def append_feature(self, attr):
        news_info = dict(zip(self.news_df["index"].values, self.news_df[attr].values))
        for i, value in news_info.items():
            self.news_features[attr][i] = value

    def _load_category(self, **kwargs):
        self.use_category = kwargs.get("use_category", False)  # use category or not
        if self.use_category:
            cat, subvert = self.news_df["category"].unique(), self.news_df["subvert"].unique()
            self.category2id, self.subvert2id = load_category_dict(category_set=cat, subvert_set=subvert, **kwargs)
            for attr in ["category", "subvert"]:
                self.news_features[attr] = [""] * (len(self.nid2index)+1)
                self.append_feature(attr)
            self.feature_matrix.update({
                "category": np.stack([self.convert_cat(c, "category") for c in self.news_features["category"]]),
                "subvert": np.stack([self.convert_cat(c, "subvert") for c in self.news_features["subvert"]])
            })

    def convert_cat(self, cat, dict_name):
        if dict_name == "category":
            return self.category2id[cat] if cat in self.category2id else 0
        elif dict_name == "subvert":
            return self.subvert2id[cat] if cat in self.subvert2id else 0

    def _init_vars(self, **kwargs):
        self.flatten_article = kwargs.get("flatten_article", True)
        self.phase = kwargs.get("phase", "train")  # RS phase: train, valid, test
        self.history_size = kwargs.get("history_size", 50)
        self.neg_pos_ratio = kwargs.get("neg_pos_ratio", 4)  # negative sampling ratio, default is 20% positive ratio
        self.mind_type = kwargs.get("mind_type", "small")
        self.train_strategy = kwargs.get("train_strategy", "pair_wise")  # pair wise uses negative sampling strategy
        self.category2id = kwargs.get("category2id", {})  # map category to index dictionary
        self.mind_dir = get_mind_dir(**kwargs)  # get directory of MIND dataset that stores news and behaviors
        self.news_info = kwargs.get("news_info", ["use_all"])  # limited in ["title", "abstract", "body", "use_all"]
        self.news_lengths = kwargs.get("news_lengths", [30])
        if isinstance(self.news_info, str):
            self.news_info = [self.news_info]
        if isinstance(self.news_lengths, int):
            self.news_lengths = [self.news_lengths]
        for attr in self.news_info:
            if attr not in ["title", "abstract", "body", "use_all"]:
                self.news_info.remove(attr)
        self.data_root = Path(kwargs.get("data_dir", os.path.join(get_project_root(), "dataset")))  # root directory
        default_uid_path = Path(self.data_root) / f"utils/MIND_uid_{self.mind_type}.json"
        self.uid2index = read_json(kwargs.get("uid_path", default_uid_path))
        self.news_attr = {k: length for k, length in zip(self.news_info, self.news_lengths)}

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
        input_feat = {f"{input_name}_index": torch.tensor(indices, dtype=torch.long)}
        for feature in self.feature_matrix.keys():
            if feature == "use_all":
                feature_name = input_name
            else:
                feature_name = f"{input_name}_{feature}"
            feature_vector = self.feature_matrix[feature][indices]
            input_feat[feature_name] = torch.tensor(feature_vector, dtype=torch.long)
            mask = np.where(feature_vector == self.tokenizer.pad_id, 0, 1)
            input_feat[f"{feature_name}_mask"] = torch.tensor(mask, dtype=torch.int32)
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
    def __init__(self, dataset: MindRSDataset, news_embeds=None, selected_imp=None):
        self.dataset = dataset
        self.behaviors = dataset.behaviors
        if selected_imp:
            imp_indices = self.behaviors["impression_index"]
            indices = [imp_indices.index(int(i)) for i in selected_imp if i in imp_indices]
            self.behaviors = {k: [v[i] for i in indices] for k, v in self.behaviors.items()}
            print(self.behaviors)
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
