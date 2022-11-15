import os

import numpy as np
import pandas as pd
from itertools import product
from sklearn.metrics.pairwise import cosine_similarity

from analysis.perform_stat import get_mean_std
from modules.config import load_cmd_line
from modules.utils import load_embedding_from_path, load_embedding_from_dict, read_json, write_to_file
from pathlib import Path


def read_topics(path, n):
    with open(path) as r:
        topics_scores = [next(r).split(":") for _ in range(n)]
        topic_list = [t[1].split() for t in topics_scores]
        scores = sorted([eval(score[0]) for score in topics_scores], reverse=True)
        return topic_list, scores


if __name__ == "__main__":
    cmd_args = load_cmd_line()
    glove_embeddings = load_embedding_from_path(path=cmd_args.get("glove_path", None))
    word_dict = read_json(cmd_args.get("word_dict_path", None))
    embeddings = load_embedding_from_dict(glove_embeddings, word_dict, "use_all")
    stat_topic_dir = Path(cmd_args.get("stat_topic_dir", "saved/stat/topics"))
    score_dir = Path(cmd_args.get("score_dir", "saved/stat/coherence_score/lda"))
    os.makedirs(score_dir, exist_ok=True)
    top_n = 10
    mean, std = np.mean(embeddings), np.std(embeddings)
    topics_dir = Path(cmd_args.get("topics_dir", None))
    name = topics_dir.name.split("/")[-1]
    lda_stats = []
    for num, seed in product([10, 30, 50, 70, 100, 150, 200, 300, 500], [2020, 2021, 25, 4, 42]):
        topics_path = topics_dir / f"{seed}/topic_list_lda_npmi_{num}.txt"
        if not os.path.exists(topics_path):
            print(f"File {topics_path} does not exist.")
            continue
        topics, npmi_scores = read_topics(topics_path, num)
        metrics = read_json(Path(topics_dir, str(seed), f"metrics_{num}.json"))
        topics_mat = [np.array([glove_embeddings[term] if term in glove_embeddings else np.random.normal(
            loc=mean, scale=std, size=300) for term in topic]) for topic in topics]
        missed_terms = sum([sum([1 for term in topic if term not in glove_embeddings]) for topic in topics])
        count = top_n * (top_n - 1) / 2
        w2v_scores = sorted([np.sum(np.triu(cosine_similarity(mat), 1)) / count for mat in topics_mat], reverse=True)
        w2v = np.round(np.mean(w2v_scores), 4)
        npmi_mean = np.round(np.mean(npmi_scores), 4)
        lda_stats.append([num, seed, npmi_mean, w2v, missed_terms])
        write_to_file(score_dir / f"npmi_{seed}_{num}_{npmi_mean}.txt", "\n".join([str(s) for s in npmi_scores]))
        write_to_file(score_dir / f"w2v_{seed}_{num}_{w2v}.txt", "\n".join([str(s) for s in w2v_scores]))
    topic_stat_df = pd.DataFrame.from_records(lda_stats, columns=["topic_number", "seed", "NPMI", "W2v", "Missed"])
    topic_stat_df.to_csv(stat_topic_dir / f"{name}_npmi_w2v.csv")
    topic_stat = []
    for topic_num, group in topic_stat_df.groupby("topic_number"):
        topic_stat.append([topic_num, get_mean_std(group["NPMI"].values, r=4), get_mean_std(group["W2v"].values, r=4)])
    mean_stat_df = pd.DataFrame.from_records(topic_stat, columns=["topic_number", "NPMI", "W2v"])
    mean_stat_df.to_csv(stat_topic_dir / f"{name}_w2v_mean.csv")
