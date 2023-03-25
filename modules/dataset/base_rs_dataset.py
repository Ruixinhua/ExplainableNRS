import os
import torch
import numpy as np
from pathlib import Path
from torch.utils.data.dataset import Dataset

from modules.dataset.nr_dataset import NewsBehaviorSet
from modules.utils import news_sampling, get_project_root


class NewsRecDataset(Dataset):
    """Load dataset from file and organize data for training"""

    def __init__(self, tokenizer, **kwargs):
        self.train_strategy = kwargs.get("train_strategy", "pair_wise")  # pair wise uses negative sampling strategy
        self.data_root = Path(kwargs.get("data_dir", os.path.join(get_project_root(), "dataset")))  # root directory
        self.dataset_name = kwargs.get("dataset_name", "MIND")  # dataset name
        self.subset_type = kwargs.get("subset_type", "small")
        self.phase = kwargs.get("phase", "train")  # RS phase: train, valid, test
        self.neg_pos_ratio = kwargs.get("neg_pos_ratio", 4)  # negative sampling ratio, default is 20% positive ratio
        self.load_object = kwargs.get("load_object", False)
        default_object_path = Path(self.data_root) / f"utils/{self.dataset_name}/{self.subset_type}_{self.phase}.pt"
        set_object_path = kwargs.get("set_object_path", default_object_path)
        if set_object_path.exists() and self.load_object:
            self.news_behavior = torch.load(set_object_path)
        else:
            self.news_behavior = NewsBehaviorSet(tokenizer=tokenizer, **kwargs)
            torch.save(self.news_behavior, set_object_path)
        self.behaviors = self.news_behavior.behaviors
        self.positive_news = self.news_behavior.positive_news
        self.feature_matrix = self.news_behavior.feature_matrix
        
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
            "label": torch.tensor(label, dtype=torch.long), "uid": torch.tensor(user_index, dtype=torch.long),
        }
        input_feat.update(self.load_news_index(history, "history"))
        input_feat.update(self.load_news_index(candidate, "candidate"))
        return input_feat

    def load_news_index(self, indices, input_name):
        """

        :param indices: List or integer, index of news information
        :param input_name: corresponding saved name of feat
        :return: a default dictionary with keys called name.
        """
        # get the matrix of corresponding news features with index
        input_feat = {f"{input_name}_index": torch.tensor(indices, dtype=torch.long)}
        for feature in self.feature_matrix.keys():
            if feature == "use_all" or input_name == "news":  # TODO: fix bug
                feature_name = input_name
            else:
                feature_name = f"{input_name}_{feature}"
            feature_vector = self.feature_matrix[feature][indices]
            input_feat[feature_name] = torch.tensor(feature_vector, dtype=torch.long)
            mask = np.where(feature_vector == self.news_behavior.tokenizer.pad_id, 0, 1)
            input_feat[f"{feature_name}_mask"] = torch.tensor(mask, dtype=torch.int32)
        return input_feat
    
    def __len__(self):
        return len(self.positive_news)


class NewsDataset(Dataset):
    def __init__(self, dataset: NewsRecDataset):
        self.dataset = dataset

    def __getitem__(self, i):
        input_feat = {"index": torch.tensor(i)}
        input_feat.update(self.dataset.load_news_index(i, "news"))
        return input_feat

    def __len__(self):
        return len(self.dataset.feature_matrix[list(self.dataset.feature_matrix.keys())[0]])


class ImpressionDataset(Dataset):
    def __init__(self, dataset: NewsRecDataset, news_embeds=None, selected_imp=None):
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
