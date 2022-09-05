import heapq
from typing import Union

import torch
import torch.distributed
import os
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from scipy.stats import entropy
from news_recommendation.utils.general_utils import write_to_file


def get_topic_dist(trainer, word_seq, voc_size=None):
    voc_size = len(word_seq) if voc_size is None else voc_size
    topic_dist = np.zeros((trainer.config.get("head_num", 20), voc_size))
    if torch.distributed.is_initialized():
        best_model = trainer.best_model.module
    else:
        best_model = trainer.best_model
    with torch.no_grad():
        bs = 512
        num = bs * (len(word_seq) // bs)
        word_feat = np.array(word_seq[:num]).reshape(-1, bs).tolist() + [word_seq[num:]]
        for words in word_feat:
            input_feat = {"news": torch.tensor(words).unsqueeze(0), "news_mask": torch.ones(len(words)).unsqueeze(0)}
            input_feat = trainer.load_batch_data(input_feat)
            _, topic_weight = best_model.extract_topic(input_feat)  # (B, H, N)
            topic_dist[:, words] = topic_weight.squeeze().cpu().data
        return topic_dist


def get_topic_list(matrix, top_n, reverse_dict):
    top_index = [heapq.nlargest(top_n, range(len(vec)), vec.take) for vec in matrix]
    topic_list = [[reverse_dict[i] for i in index] for index in top_index]
    return topic_list


def evaluate_topic(topic_list, texts, method="c_v", top_n=25):
    dictionary = Dictionary(texts)
    cm = CoherenceModel(topics=topic_list, texts=texts, dictionary=dictionary, coherence=method, topn=top_n)
    return cm.get_coherence_per_topic()


def evaluate_entropy(topic_dist):
    token_entropy, topic_entropy = np.mean(entropy(topic_dist, axis=0)),  np.mean(entropy(topic_dist, axis=1))
    return token_entropy, topic_entropy


def topic_evaluation(trainer, word_dict, path: Union[str, os.PathLike], ref_texts=None, top_n=25, voc_size=None):
    # statistic topic distribution of Topic Attention network
    reverse_dict = {v: k for k, v in word_dict.items()}
    topic_dist = get_topic_dist(trainer, list(word_dict.values()), voc_size)  # get distribution for the given words
    topic_list = get_topic_list(topic_dist, top_n, reverse_dict)  # convert to tokens list
    if ref_texts is None:
        from news_recommendation.utils.dataset_utils import load_docs
        ref_texts = load_docs("MIND15")
    os.makedirs(path, exist_ok=True)
    topic_result = save_topic_info(path, topic_list, ref_texts, top_n=top_n)
    token_entropy, topic_entropy = evaluate_entropy(topic_dist)
    topic_result.update({"token_entropy": token_entropy, "topic_entropy": topic_entropy, "top_n": top_n})
    return topic_result


def save_topic_info(path, topic_list, ref_texts, top_n=25, methods="c_v,c_npmi"):
    write_to_file(os.path.join(path, "topic_list.txt"), [" ".join(topics) for topics in topic_list])
    topic_scores = {m: evaluate_topic(topic_list, ref_texts, m, top_n) for m in methods.split(",")}
    topic_result = {m: np.round(np.mean(c), 4) for m, c in topic_scores.items()}
    for method, scores in topic_scores.items():
        topic_file = os.path.join(path, f"topic_list_{method}_{topic_result[method]}.txt")
        for topics, score in zip(topic_list, scores):
            write_to_file(topic_file, f"{np.round(score, 4)}: {' '.join(topics)}\n", "a+")
        write_to_file(topic_file, f"Average score: {topic_result[method]}\n", "a+")
    return topic_result
