import heapq
import torch
import os
import numpy as np
from gensim.corpora import Dictionary
from gensim.models import CoherenceModel
from scipy.stats import entropy
from news_recommendation.utils.general_utils import write_to_file
from typing import Dict, Union, List
from scipy import sparse


class NPMI:
    """
    Reference: https://github.com/ahoho/gensim-runner/blob/main/utils.py
    NPMI (Normalized Pointwise Mutual Information) is a measure of association between two words: usually used to 
    evaluate topic quality.
    """
    def __init__(
        self,
        bin_ref_counts: Union[np.ndarray, sparse.spmatrix],
        vocab: Dict[str, int] = None,
    ):
        assert bin_ref_counts.max() == 1
        self.bin_ref_counts = bin_ref_counts
        if sparse.issparse(self.bin_ref_counts):
            self.bin_ref_counts = self.bin_ref_counts.tocsc()
        self.npmi_cache = {} # calculating NPMI is somewhat expensive, so we cache results
        self.vocab = vocab

    def compute_npmi(
        self,
        beta: np.ndarray = None,
        topics: Union[np.ndarray, List] = None,
        vocab: Dict[str, int] = None,
        n: int = 10
    ) -> np.ndarray:
        """
        Compute NPMI for an estimated beta (topic-word distribution) parameter using
        binary co-occurence counts from a reference corpus

        Supply `vocab` if the topics contain terms that first need to be mapped to indices
        """
        if beta is not None and topics is not None:
            raise ValueError(
                "Supply one of either `beta` (topic-word distribution array) "
                "or `topics`, a list of index or word lists"
            )
        if vocab is None and any([isinstance(idx, str) for idx in topics[0][:n]]):
            raise ValueError(
                "If `topics` contains terms, not indices, you must supply a `vocab`"
            )

        if beta is not None:
            topics = np.flip(beta.argsort(-1), -1)[:, :n]
        if topics is not None:
            topics = [topic[:n] for topic in topics]
        if vocab is not None:
            assert(len(vocab) == self.bin_ref_counts.shape[1])
            topics = [[vocab[w] for w in topic[:n]] for topic in topics]

        num_docs = self.bin_ref_counts.shape[0]
        npmi_means = []
        for indices in topics:
            npmi_vals = []
            for i, idx_i in enumerate(indices):
                for idx_j in indices[i+1:]:
                    ij = frozenset([idx_i, idx_j])
                    try:
                        npmi = self.npmi_cache[ij]
                    except KeyError:
                        col_i = self.bin_ref_counts[:, idx_i]
                        col_j = self.bin_ref_counts[:, idx_j]
                        c_i = col_i.sum()
                        c_j = col_j.sum()
                        if sparse.issparse(self.bin_ref_counts):
                            c_ij = col_i.multiply(col_j).sum()
                        else:
                            c_ij = (col_i * col_j).sum()
                        if c_ij == 0:
                            npmi = 0.0
                        else:
                            npmi = (
                                (np.log(num_docs) + np.log(c_ij) - np.log(c_i) - np.log(c_j))
                                / (np.log(num_docs) - np.log(c_ij))
                            )
                        self.npmi_cache[ij] = npmi
                    npmi_vals.append(npmi)
            npmi_means.append(np.mean(npmi_vals))

        return np.array(npmi_means)
    
    
def load_sparse(input_file):
    return sparse.load_npz(input_file).tocsr()


def extract_topics(model, word_seq, device, topic_num=20, voc_size=None):
    voc_size = len(word_seq) if voc_size is None else voc_size
    topic_dist = np.zeros((topic_num, voc_size))
    model = model.to(device)
    with torch.no_grad():
        word_feat = {"news": torch.tensor(word_seq).unsqueeze(0).to(device),
                     "news_mask": torch.ones(len(word_seq)).unsqueeze(0).to(device)}
        try:
            _, topic_weight = model.extract_topic(word_feat)  # (B, H, N)
        except AttributeError:
            model = model.module  # for multi-GPU
            _, topic_weight = model.extract_topic(word_feat)  # (B, H, N)
        topic_dist[:, word_seq] = topic_weight.squeeze().cpu().data
    return topic_dist


def get_topic_dist(model, word_seq: list, topic_num, voc_size=None):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # only run on one GPU
    try:
        topic_dist = extract_topics(model, word_seq, device, topic_num, voc_size)
    except (RuntimeError, ):  # RuntimeError: CUDA out of memory, change to CPU
        device = torch.device("cpu")
        topic_dist = extract_topics(model, word_seq, device, topic_num, voc_size)
    return topic_dist


def get_topic_list(matrix, top_n, reverse_dict):
    top_index = [heapq.nlargest(top_n, range(len(vec)), vec.take) for vec in matrix]
    topic_list = [[reverse_dict[i] for i in index] for index in top_index]
    return topic_list


def compute_coherence(topic_list, texts, method="c_v", top_n=25):
    dictionary = Dictionary(texts)
    cm = CoherenceModel(topics=topic_list, texts=texts, dictionary=dictionary, coherence=method, topn=top_n)
    return cm.get_coherence_per_topic()


def evaluate_entropy(topic_dist):
    token_entropy, topic_entropy = np.mean(entropy(topic_dist, axis=0)),  np.mean(entropy(topic_dist, axis=1))
    return token_entropy, topic_entropy


def save_topic_info(path, topic_list, topic_scores):
    topic_result = {m: np.round(np.mean(c), 4) for m, c in topic_scores.items()}
    for method, scores in topic_scores.items():
        topic_file = os.path.join(path, f"topic_list_{method}_{topic_result[method]}.txt")
        for topics, score in zip(topic_list, scores):
            write_to_file(topic_file, f"{np.round(score, 4)}: {' '.join(topics)}\n", "a+")
        write_to_file(topic_file, f"Average score: {topic_result[method]}\n", "a+")
    return topic_result
