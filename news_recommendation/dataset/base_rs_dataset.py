import os
import torch

import numpy as np
import pandas as pd
from pathlib import Path

from torch.utils.data.dataset import Dataset

from collections import OrderedDict, defaultdict
from news_recommendation.utils import read_json, news_sampling, Tokenizer, get_project_root, get_mind_root_path
from news_recommendation.utils.graph_untils import load_entities, load_entity_feature


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
        self.nid2index = kwargs.get("nid2index", {})  # map news id to index dictionary
        self.train_strategy = kwargs.get("train_strategy", "pair_wise")  # pair wise uses negative sampling strategy

        self._load_news_matrix(**kwargs)  # load news matrix
        self._load_behaviors(**kwargs)  # load behavior file, default use mind dataset

    def convert_category(self, cat):
        if cat not in self.category2id:
            self.category2id[cat] = len(self.category2id) + 1
        return self.category2id[cat]

    def _load_news_from_file(self, news_file):
        head = ["news_id", "category", "subcategory", "title", "abstract", "url", "title_entity", "abstract_entity"]
        self.news_content = pd.read_csv(news_file, sep="\t", names=head)  # load news information from file
        if self.use_body:  # use news articles
            body_file = Path(os.path.dirname(news_file)) / "msn.json"
            self.news_articles = read_json(body_file) if body_file.exists() else None

    def _load_news_text(self, **kwargs):
        """
        Load news from news file
        """
        news_file = get_mind_root_path(**kwargs) / "news.tsv"  # define news file path
        self.news_attr = {"title": kwargs.get("title", 30), "body": kwargs.get("body", None)}  # default only use title
        # initial data of corresponding news attributes, such as: title, entity, vert, subvert, abstract
        self.news_text = OrderedDict({"title": [""]})  # default use title only
        self.news_entity_num = kwargs.get("news_entity_num", None)  # default not use entity
        self.use_category, self.use_sub_cat = kwargs.get("use_category", 0), kwargs.get("use_subcategory", 0)
        if self.news_entity_num:
            self.entity2id = load_entities(**kwargs)  # load wikidata id to entity id mapping
            self.news_entity_feature, self.entity_type_dict = [np.zeros(4 * self.news_entity_num)], defaultdict()
        if self.use_category or self.use_sub_cat:
            self.category2id = OrderedDict()
            self.category_index = [np.array([0, 0])] if self.use_category and self.use_sub_cat else [np.array([0])]
        # load news articles text from a json file
        body_file = Path(os.path.dirname(news_file)) / "msn.json"
        self.use_body = self.news_attr["body"] and body_file.exists()  # boolean value for use article or not
        self.news_articles = read_json(body_file) if self.use_body else None  # read articles from file
        if self.use_body:  # setup default article data format based on flatten option
            self.news_text["body"] = [] if self.flatten_article else [[""]]
        # use sentence embedding, must specify method and document embedding dim
        sentence_embed_method = kwargs.get("sentence_embed_method", None)
        document_embedding_dim = kwargs.get("document_embedding_dim", None)
        self.use_sent_embed = sentence_embed_method and document_embedding_dim
        if self.use_sent_embed:
            self.sentence_embed = [np.zeros(document_embedding_dim)]  # initial zero embedding
            embed_file = Path(get_project_root()) / f"dataset/utils/sentence_embed/{sentence_embed_method}.vec"
            if not embed_file.exists():
                raise FileNotFoundError("Sentence embedding file is not found")
            self.sentence_embed_dict = torch.load(embed_file)
        with open(news_file, "r", encoding="utf-8") as rd:
            for text in rd:
                # news id, category, subcategory, title, abstract, url
                nid, vert, subvert, title, abstract, url, title_entity, abs_entity = text.strip("\n").split("\t")
                if nid in self.nid2index:
                    continue
                category = []
                if self.use_category:
                    category.append(self.convert_category(vert))
                if self.use_sub_cat:
                    category.append(self.convert_category(subvert))
                if self.use_category or self.use_sub_cat:
                    self.category_index.append(np.array(category, dtype=np.int))
                if self.news_entity_num:
                    entity_feature = load_entity_feature(title_entity, abs_entity, self.entity_type_dict)
                    entities = [[self.entity2id[entity_id]] + feature   # [entity id, freq, pos, type]
                                for entity_id, feature in entity_feature.items() if entity_id in self.entity2id]
                    entities.append([0, 0, 0, 0])  # avoid zero array
                    pad_size = max(0, self.news_entity_num - len(entities))
                    entities = np.array(entities, dtype=np.int)[:self.news_entity_num]  # shape is (N, 4)
                    feature = np.pad(entities, [(0, pad_size), (0, 0)]) if pad_size else entities
                    self.news_entity_feature.append(feature.transpose().flatten())  # convert to a 4*N vector
                news_dict = {"title": title + " " + abstract}  # abstract is used as a part of title
                if self.use_body:  # add news body
                    if nid not in self.news_articles or self.news_articles[nid] is None:  # article not found or none
                        article = "" if self.flatten_article else [""]
                    else:
                        article = " ".join(self.news_articles[nid]) if self.flatten_article else self.news_articles[nid]
                    news_dict["body"] = article
                if self.use_sent_embed:
                    self.sentence_embed.append(self.sentence_embed_dict[nid])
                # add news attribution
                for attr in self.news_text.keys():
                    if attr in news_dict:
                        self.news_text[attr].append(news_dict[attr])
                self.nid2index[nid] = len(self.nid2index) + 1
        rd.close()

    def _load_news_matrix(self, **kwargs):
        self._load_news_text(**kwargs)  # load news text from file first before made up news matrix
        self.news_matrix = OrderedDict({  # init news text matrix
            k: np.stack([self.tokenizer.tokenize(news, self.news_attr[k]) for news in news_text])
            for k, news_text in self.news_text.items()
        })  # the order of news info: title(abstract), category, sub-category, sentence embedding, entity feature
        if self.use_category or self.use_sub_cat:
            self.news_matrix["category"] = np.array(self.category_index, dtype=np.int)
        if self.use_sent_embed:
            self.news_matrix["sentence_embed"] = np.array(self.sentence_embed, dtype=np.float)
        if self.news_entity_num:  # after load news from file
            self.news_matrix["entity_feature"] = np.array(self.news_entity_feature, dtype=np.int)

    def _load_behaviors(self, **kwargs):
        """"
        Create global variants for behaviors:
        behaviors: The behaviors of the user, includes: history clicked news, impression news, and labels
        positive_news: The index of news that clicked by users in impressions
        """
        behaviors_file = get_mind_root_path(**kwargs) / "behaviors.tsv"  # define behavior file path
        # initial behaviors attributes
        attributes = ["uid", "impression_index", "history_news", "history_length", "candidate_news", "labels"]
        self.behaviors = {attr: [] for attr in attributes}
        self.positive_news = []
        with open(behaviors_file, "r", encoding="utf-8") as rd:
            imp_index = 0
            for index in rd:
                # read line of behaviors file
                uid, time, history, candidates = index.strip("\n").split("\t")[-4:]
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

    def load_news_index(self, indices, input_name):
        """

        :param indices: index of news
        :param input_name: corresponding saved name of feat
        :return: a default dictionary with keys called name.
        """
        # get the matrix of corresponding news features with index
        news = [self.news_matrix[k][indices] for k in self.news_matrix.keys()]
        input_feat = {
            input_name: torch.tensor(np.concatenate(news, axis=-1), dtype=torch.long),
            f"{input_name}_index": torch.tensor(indices, dtype=torch.long),
        }
        # pass news mask to the model
        mask = np.where(input_feat[input_name] == self.tokenizer.pad_id, self.tokenizer.pad_id, 1)
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
            "history_length": torch.tensor(len(history), dtype=torch.int32),
            # "padding": torch.tensor(self.tokenizer.tokenize("", 0), dtype=torch.int32),  # TODO padding sentence
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
        return len(self.dataset.news_matrix[list(self.dataset.news_matrix.keys())[0]])


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
