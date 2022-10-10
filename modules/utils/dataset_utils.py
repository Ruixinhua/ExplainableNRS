import pandas as pd
import random
import string
import re
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from modules.utils.preprocess_utils import text2index
from modules.utils.general_utils import read_json, write_json, get_project_root


def clean_text(text):
    rule = string.punctuation + "0123456789"
    return re.sub(rf'([^{rule}a-zA-Z ])', r" ", text)


def clean_df(data_df):
    """
    clean the data frame, remove non-ascii characters, and remove empty news
    :param data_df: input data frame, should contain title, body, and category columns
    :return: cleaned data frame
    """
    data_df.dropna(subset=["title", "body"], inplace=True, how="all")
    data_df.fillna("", inplace=True)
    data_df["title"] = data_df.title.apply(lambda s: clean_text(s))
    if "abstract" in data_df.columns:
        data_df["abstract"] = data_df.abstract.apply(lambda s: clean_text(s))
    data_df["body"] = data_df.body.apply(lambda s: clean_text(s))
    return data_df


def split_df(df, split=0.1, split_test=False):
    indices = df.index.values
    random.Random(42).shuffle(indices)
    split_len = round(split * len(df))
    df.loc[indices[:split_len], "split"] = "valid"
    if split_test:
        df.loc[indices[split_len:split_len*2], "split"] = "test"
        df.loc[indices[split_len*2:], "split"] = "train"
    else:
        df.loc[indices[split_len:], "split"] = "train"
    return df


def load_set_by_type(dataset, set_type: str) -> pd.DataFrame:
    df = {k: [] for k in ["data", "category"]}
    for text, label in zip(dataset[set_type]["text"], dataset[set_type]["label"]):
        for c, v in zip(["data", "category"], [text, label]):
            df[c].append(v)
    df["split"] = set_type
    return pd.DataFrame(df)


def load_tokenized_text(df, **kwargs):
    tokenized_method = kwargs.get("tokenized_method", "keep_all")
    if tokenized_method == "use_tokenize":
        tokenized_news_path = kwargs.get("tokenized_news_path", None)
        if tokenized_news_path is None:
            tokenized_text = df["tokenized_text"]
        else:  # TODO: load tokenized text from file
            tokenized_news = pd.read_csv(tokenized_news_path)
            tokenized_text = pd.merge(df, tokenized_news, on="news_id", how="left")["tokenized_text"].fillna("")
        return tokenized_text
    else:
        df["data"] = df.title + "\n" + df.body
        if "abstract" in df.columns:
            df["data"] += "\n" + df.abstract
        return df["data"]


def load_dataset_df(dataset_name="MIND15", data_path=None, **kwargs):
    if dataset_name.lower().startswith("mind") or dataset_name.lower().startswith("news"):
        if data_path is None:  # TODO: if data is not exist, download it
            data_path = Path(get_project_root()) / "dataset" / "data" / f"{dataset_name}.csv"
        df = clean_df(pd.read_csv(data_path, encoding="utf-8"))
        df["data"] = load_tokenized_text(df, **kwargs)

    elif dataset_name in ["ag_news", "yelp_review_full", "imdb"]:
        # load corresponding dataset from datasets library
        dataset = load_dataset(dataset_name)
        train_set, test_set = split_df(load_set_by_type(dataset, "train")), load_set_by_type(dataset, "test")
        df = train_set.append(test_set)
    else:
        raise ValueError("dataset name should be in one of MIND15, IMDB, News26, and ag_news...")
    labels = df["category"].values.tolist()
    label_dict = dict(zip(sorted(set(labels)), range(len(set(labels)))))
    return df, label_dict


def load_word_dict(**kwargs):
    data_root = kwargs.get("data_dir", Path(get_project_root()) / "dataset")
    word_dict_file = kwargs.get("word_dict_file", "default.csv")  # will not save to default dictionary
    word_dict_path = kwargs.get("word_dict_path", os.path.join(data_root, "utils", "word_dict", word_dict_file))
    if os.path.exists(word_dict_path):
        word_dict = read_json(word_dict_path)
    else:
        word_dict = {"[UNK]": 0}
        dataset_name = kwargs.get("dataset_name", "MIND15")
        if dataset_name is None:
            raise ValueError("dataset name should be provided: MIND15 or News26")
        data_path = kwargs.get("data_path", Path(data_root) / "data" / f"{dataset_name}.csv")
        if not os.path.exists(data_path):
            raise ValueError("data path is not correct or MIND15.csv is not exist")
        df = kwargs.get("df", load_dataset_df(dataset_name, data_path)[0])
        tokenized_method = kwargs.get("tokenized_method", "use_all")
        df.data.apply(lambda s: text2index(s, word_dict, tokenized_method, False))
        word_dict_path = Path(word_dict_path)
        os.makedirs(word_dict_path.parent, exist_ok=True)
        write_json(word_dict, word_dict_path.parent / f"{dataset_name}_{len(word_dict)}.json")
    return word_dict


def load_embedding_from_path(path=None):
    if not path:
        path = "E:\\glove.840B.300d.txt"
    if path.endswith(".txt"):
        glove = pd.read_csv(path, sep=" ", quoting=3, header=None, index_col=0)
        embeddings = {key: val.values for key, val in glove.T.items()}
    elif path.endswith(".npy"):
        embeddings = np.load(str(path)).item()
    else:
        raise ValueError("embedding file should be in txt or npy format")
    return embeddings


def load_embedding_from_dict(embed_dict: dict, word_dict: dict, embed_method: str, embed_dim: int = 300):
    new_wd = {}
    embeddings, exclude_words = [], []
    # acquire the embedding values in the embedding dictionary
    for i, w in enumerate(word_dict.keys()):
        if w in embed_dict:
            embeddings.append(embed_dict[w])
            new_wd[w] = embed_dict[w]
        else:
            exclude_words.append(w)
    if embed_method == "use_all":
        # append a random value if all words are initialized with values
        mean, std = np.mean(embeddings), np.std(embeddings)
        for i, w in enumerate(exclude_words):
            # append random embedding
            random_embed = np.random.normal(loc=mean, scale=std, size=300)
            new_wd[w] = random_embed
            embeddings.append(random_embed)
    elif embed_method == "zero_padding":  # use zero padding for the words not in the embedding dictionary
        # add zero embedding
        for i, w in enumerate(exclude_words):
            new_wd[w] = np.zeros(embed_dim)
            embeddings.append(np.zeros(embed_dim))
    else:  # skip the words that are not in the embedding dictionary
        pass
    # get embeddings with original word dictionary
    for w, i in word_dict.items():
        embeddings[i] = new_wd[w]
    return np.array(embeddings)


def load_embeddings(**kwargs):
    """
    load embeddings from path or dict and save
    :param kwargs:
    :return:
    """
    embed_method = kwargs.get("embed_method", "use_all")
    dataset_name = kwargs.get("dataset_name", "MIND15")
    data_root = kwargs.get("data_dir", Path(get_project_root()) / "dataset")
    word_dict = kwargs.get("word_dict", load_word_dict(**kwargs))  # load word dictionary
    embed_file = kwargs.get("embed_file", f"{dataset_name}_{len(word_dict)}.npy")
    embed_path = Path(kwargs.get("embed_path", os.path.join(data_root, "utils", "embed_dict", embed_file)))
    if os.path.exists(embed_path):
        return np.load(str(embed_path))
    else:
        glove_path = kwargs.get("glove_path", None)
        embed_path = embed_path.parent / embed_file
        if os.path.exists(embed_path):
            return np.load(str(embed_path))
        embed_dict = load_embedding_from_path(glove_path)
        embeddings = load_embedding_from_dict(embed_dict, word_dict, embed_method)
        os.makedirs(embed_path.parent, exist_ok=True)
        if not os.path.exists(embed_path):
            np.save(str(embed_path), embeddings)
    return embeddings
