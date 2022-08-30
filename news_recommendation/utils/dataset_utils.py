import pandas as pd
import random
import os
import numpy as np
from pathlib import Path
from datasets import load_dataset
from news_recommendation.utils.preprocess_utils import clean_text, text2index, tokenize, lemmatize, add_bigram
from news_recommendation.utils.general_utils import read_json, write_json, get_project_root


def clean_df(data_df):
    data_df.dropna(subset=["title", "body"], inplace=True, how="all")
    data_df.fillna("empty", inplace=True)
    data_df["title"] = data_df.title.apply(lambda s: clean_text(s))
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


def load_dataset_df(dataset_name, data_path=None, **kwargs):
    if dataset_name in ["MIND15", "News26"]:
        if data_path is None:
            data_path = Path(get_project_root()) / "dataset" / "data" / f"{dataset_name}.csv"
        df = clean_df(pd.read_csv(data_path, encoding="utf-8"))
        tokenized_method = kwargs.get("tokenized_method", "keep_all")
        if tokenized_method == "use_tokenize":
            df["data"] = df["tokenized_text"]
        else:
            df["data"] = df.title + "\n" + df.body
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


def load_docs(name, method, do_lemma=False, add_bi=False, min_count=200):
    df, _ = load_dataset_df(name)
    docs = [tokenize(d, method) for d in df["data"].values]
    if do_lemma:
        docs = lemmatize(docs)
    if add_bi:
        docs = add_bigram(docs, min_count)
    return docs


def load_word_dict(data_root, dataset_name, tokenized_method, **kwargs):
    embed_method = kwargs.get("embed_method", "use_all")
    wd_path = Path(data_root) / "utils" / "word_dict" / f"{dataset_name}_{tokenized_method}_{embed_method}.json"
    if os.path.exists(wd_path):
        word_dict = read_json(wd_path)
    else:
        word_dict = {"[UNK]": 0}
        data_path = kwargs.get("data_path", Path(data_root) / "data" / f"{dataset_name}.csv")
        df = kwargs.get("df", load_dataset_df(dataset_name, data_path)[0])
        df.data.apply(lambda s: text2index(s, word_dict, tokenized_method, False))
        os.makedirs(wd_path.parent, exist_ok=True)
        write_json(word_dict, wd_path)
    return word_dict


def load_embedding_from_path(path=None):
    if not path:
        path = "E:\\glove.840B.300d.txt"
    glove = pd.read_csv(path, sep=" ", quoting=3, header=None, index_col=0)
    return {key: val.values for key, val in glove.T.items()}


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
    else:
        # add zero embedding
        for i, w in enumerate(exclude_words):
            new_wd[w] = np.zeros(embed_dim)
            embeddings.append(np.zeros(embed_dim))
    # get embeddings with original word dictionary
    for w, i in word_dict.items():
        embeddings[i] = new_wd[w]
    return np.array(embeddings)


def load_glove_embeddings(data_root, dataset_name, tokenized_method, word_dict=None, glove_path=None, **kwargs):
    embed_method = kwargs.get("embed_method", "use_all")
    embed_path = Path(data_root) / "utils" / "embed_dict" / f"{dataset_name}_{tokenized_method}_{embed_method}.npy"
    if os.path.exists(embed_path):
        embeddings = np.load(embed_path.__str__())
    else:
        embed_dict = load_embedding_from_path(glove_path)
        if word_dict is None:
            word_dict = load_word_dict(data_root, dataset_name, tokenized_method, embed_method=embed_method)
        embeddings = load_embedding_from_dict(embed_dict, word_dict, embed_method)
        os.makedirs(embed_path.parent, exist_ok=True)
        np.save(embed_path.__str__(), embeddings)
    return embeddings
