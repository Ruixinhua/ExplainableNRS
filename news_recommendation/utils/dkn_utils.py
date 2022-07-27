import os
import numpy as np
from pathlib import Path

from news_recommendation.utils import get_project_root, download_resources


def load_utils_file(**kwargs):
    mind_type = kwargs.get("mind_type")
    utils_path = Path(get_project_root()) / "dataset/utils/dkn_utils" / f"mind-{mind_type}-dkn"
    os.makedirs(utils_path, exist_ok=True)
    yaml_file = utils_path / "dkn.yaml"
    if not yaml_file.exists():
        download_resources(r"https://recodatasets.z20.web.core.windows.net/deeprec/",
                           str(utils_path.parent), f"mind-{mind_type}-dkn.zip")
    utils_filename = {
        "news_feature_file": utils_path / "doc_feature.txt",
        "word_embed_file": utils_path / "word_embeddings_100.npy",
        "entity_embed_file": utils_path / "TransE_entity2vec_100.npy",
        "context_embed_file": utils_path / "TransE_context2vec_100.npy"
    }
    return utils_filename


def load_news_feature(news_file, **kwargs):
    nid2idx = {}
    news_matrix = {
        "title": [np.zeros(kwargs.get("title", 10), dtype=np.int)],
        "entity": [np.zeros(kwargs.get("news_entity_num", 10), dtype=np.int)]
    }
    with open(news_file, "r", encoding="utf-8") as rd:
        for line in rd:
            nid, title, entity = line.strip().split()
            if nid in nid2idx:
                continue
            nid2idx[nid] = len(nid2idx) + 1
            news_matrix["title"].append(np.array([int(i) for i in title.split(",")], dtype=np.int))
            news_matrix["entity"].append(np.array([int(e) for e in entity.split(",")], dtype=np.int))
    news_matrix["title"] = np.stack(news_matrix["title"])
    news_matrix["entity"] = np.stack(news_matrix["entity"])
    return nid2idx, news_matrix
