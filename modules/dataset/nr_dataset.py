import os
import pandas as pd
import numpy as np

from pathlib import Path
from collections import OrderedDict

from modules.utils import get_subset_dir, read_json, write_json, load_category, get_project_root, Tokenizer


class NewsBehaviorSet:
    def __init__(self, **kwargs):
        # setup some default configurations for basic NewsBehaviorSet
        self.tokenizer = kwargs.get("tokenizer", Tokenizer(**kwargs))
        self.phase = kwargs.get("phase", "train")  # RS phase: train, valid, test
        self.history_size = kwargs.get("history_size", 50)
        self.train_strategy = kwargs.get("train_strategy", "pair_wise")  # pair wise uses negative sampling strategy
        self.dataset_name = kwargs.get("dataset_name", "MIND")  # dataset name
        self.subset_dir = get_subset_dir(**kwargs)  # get directory of MIND dataset that stores news and behaviors
        self.subset_type = kwargs.get("subset_type", "small")
        self.data_root = Path(kwargs.get("data_dir", os.path.join(get_project_root(), "dataset")))  # root directory
        self._init_news_features(**kwargs)
        self._load_category(cat_type="category", **kwargs)  # load category information
        self._load_category(cat_type="subvert", **kwargs)  # load subcategory information
        self._load_behaviors(**kwargs)  # load behaviors information

    def _load_behaviors(self, **kwargs):
        """"
        Create global variants for behaviors:
        behaviors: The behaviors of the user, includes: history clicked news, impression news, and labels
        positive_news: The index of news that clicked by users in impressions
        """
        behaviors_file = self.subset_dir / "behaviors.tsv"  # define behavior file path
        attributes = ["uid", "index", "impression_index", "history_news", "history_length", "candidate_news", "labels"]
        self.behaviors = {attr: [] for attr in attributes}
        default_uid_path = Path(self.data_root) / f"utils/{self.dataset_name}/uid_{self.subset_type}.json"
        uid_path = kwargs.get("uid_path", default_uid_path)
        if not uid_path.exists():
            self.uid2index = {}
        else:
            self.uid2index = read_json(uid_path)
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
                if uid not in self.uid2index:
                    self.uid2index[uid] = len(self.uid2index)
                uindex = self.uid2index[uid]
                try:
                    imp_index = int(imp_index)
                except ValueError:
                    imp_index = index
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
        # write uid2index to file
        os.makedirs(uid_path.parent, exist_ok=True)
        write_json(self.uid2index, uid_path)

    def _load_category(self, cat_type, **kwargs):
        setattr(self, f"use_{cat_type}", kwargs.get(f"use_{cat_type}", False))
        if getattr(self, f"use_{cat_type}") and cat_type in self.news_df.columns:
            setattr(self, f"{cat_type}2id", load_category(category_set=self.news_df[cat_type].unique(), **kwargs))
            self.append_feature(cat_type)
            categories = self.news_features[cat_type]
            self.feature_matrix[cat_type] = np.stack([self.convert_cat(c, cat_type) for c in categories])

    def convert_cat(self, cat, cat_type):
        cat_dict = getattr(self, f"{cat_type}2id")
        return cat_dict[cat] if cat in cat_dict else 0

    def append_feature(self, attr):
        news_info = dict(zip(self.news_df["index"].values, self.news_df[attr].values))
        if attr not in self.news_features:
            self.news_features[attr] = [""] * (len(self.nid2index) + 1)
        for i, value in news_info.items():
            self.news_features[attr][i] = value

    def _init_news_features(self, **kwargs):
        self.news_info = kwargs.get("news_info", ["use_all"])  # limited in ["title", "abstract", "body", "use_all"]
        self.news_lengths = kwargs.get("news_lengths", [30])
        if isinstance(self.news_info, str):
            self.news_info = [self.news_info]
        if isinstance(self.news_lengths, int):
            self.news_lengths = [self.news_lengths]
        for attr in self.news_info:
            if attr not in ["title", "abstract", "body", "use_all"]:
                self.news_info.remove(attr)
        self.news_attr = {k: length for k, length in zip(self.news_info, self.news_lengths)}
        default_nid_path = os.path.join(self.data_root, f"utils/{self.dataset_name}/nid_{self.subset_type}.json")
        nid_path = Path(kwargs.get("nid_path", default_nid_path))  # get nid path from kwargs
        self.news_df = pd.read_csv(self.data_root / f"{self.dataset_name}/{self.subset_type}/news.csv").fillna("")
        if nid_path.exists():
            self.nid2index = read_json(nid_path)
        else:
            self.nid2index = dict(zip(self.news_df.news_id, range(1, len(self.news_df) + 1)))
            os.makedirs(nid_path.parent, exist_ok=True)
            write_json(self.nid2index, nid_path)
        self.news_df["index"] = self.news_df["news_id"].apply(lambda x: self.nid2index[x])
        self.news_features = OrderedDict({k: [""] * (len(self.nid2index)+1) for k in self.news_info})
        for attr in self.news_info:
            if attr == "use_all":
                # use all information of news (title, abstract, body)
                self.news_df["use_all"] = self.news_df["title"]+" "+self.news_df["abstract"]+" "+self.news_df["body"]
            self.append_feature(attr)
        self.feature_matrix = OrderedDict({  # init news text matrix
            k: np.stack(self.tokenizer.tokenize(news_text, self.news_attr[k], return_tensors=False))
            for k, news_text in self.news_features.items()
        })  # tokenize news text (title, abstract, body, all) and convert to numpy array
