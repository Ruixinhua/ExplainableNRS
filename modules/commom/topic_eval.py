import os
import copy
import torch
import numpy as np

from pathlib import Path
from scipy.stats import entropy
from modules.utils import load_word_dict, get_topic_dist, get_topic_list, get_project_root, load_embeddings, read_json
from modules.utils import fast_npmi_eval, w2v_sim_eval, slow_topic_eval, write_to_file, cal_topic_diversity
from modules.utils import kl_divergence_rowwise


class TopicEval:
    def __init__(self, config, **kwargs):
        # load basic configurations
        self.config = config
        self.word_dict = kwargs.get("word_dict", load_word_dict(**config.final_configs))
        self.embeddings = load_embeddings(**config.final_configs)
        self.group_name = kwargs.get("group_name", "topic_eval/")
        self.reverse_dict = {v: k for k, v in self.word_dict.items()}
        self.topic_evaluation_method = config.get("topic_evaluation_method", None)
        self.top_n = config.get("top_n", 10)

    def _add_post_topic_dist(self, model):
        self.model = copy.deepcopy(model)
        default_post_dict_path = Path(get_project_root(), "dataset", "utils", "word_dict", "post_process")
        self.post_word_dict_dir = self.config.get("post_word_dict_dir", default_post_dict_path)
        self.topic_dist = get_topic_dist(self.model, self.word_dict)
        self.topic_dists = {"original": self.topic_dist}
        if os.path.exists(self.post_word_dict_dir):
            for path in os.scandir(self.post_word_dict_dir):
                if not path.name.endswith(".json"):
                    continue
                post_word_dict = read_json(path)
                removed_index = [v for k, v in self.word_dict.items() if k not in post_word_dict]
                topic_dist_copy = copy.deepcopy(self.topic_dist)  # copy original topical distribution
                topic_dist_copy[:, removed_index] = 0  # set removed terms to 0
                self.topic_dists[path.name.replace(".json", "")] = topic_dist_copy
        del self.model
        torch.cuda.empty_cache()

    def result(self, model, **kwargs):
        """
        Acquire topic quality scores for the given model
        :param model: the model to be evaluated
        :param kwargs: could set the values of middle_name (middle name of saved topic directory) / use_post_dict
        :return: a dictionary of topic quality scores
        """
        topic_result = {}
        saved_name = f"topics_{self.config.seed}_{self.config.head_num}"
        middle_name = kwargs.get("middle_name", None)
        if middle_name is not None:
            saved_name += f"/{middle_name}"
        if kwargs.get("use_post_dict", True):
            self._add_post_topic_dist(model)
        for key, dist in self.topic_dists.items():  # calculate topic quality for different Post processing methods
            topic_scores = {}
            prefix = f"{self.group_name}/{key}/"
            topic_list = get_topic_list(dist, self.top_n, self.reverse_dict)  # convert to tokens list
            if "fast_npmi" in self.topic_evaluation_method:
                self.config.set("ref_data_path", os.path.join(get_project_root(), "dataset/utils/wiki.dtm.npz"))
                # convert to index list: minus 1 because the index starts from 0 (0 is for padding)
                topic_scores[f"{prefix}/NPMI"] = fast_npmi_eval(self.config, topic_list, self.word_dict)
            if "slow_eval" in self.topic_evaluation_method:
                scores = slow_topic_eval(self.config, topic_list)
                topic_scores.update({f"{prefix}/{k}": v for k, v in scores.items()})
            if "w2v_sim" in self.topic_evaluation_method:  # compute word embedding similarity
                topic_scores[f"{prefix}/W2V"] = w2v_sim_eval(self.config, self.embeddings, topic_list, self.word_dict)
            entropy_scores = np.array(entropy(dist, axis=1))  # calculate entropy for each topic
            # calculate average score for each topic quality method
            topic_result.update({m: np.round(np.mean(c), 4) for m, c in topic_scores.items()})
            topic_result[f"{prefix}/Diversity"] = np.round(cal_topic_diversity(topic_list), 4)  # calculate diversity
            topic_result[f"{prefix}/Entropy"] = np.round(np.mean(entropy_scores), 4)  # calculate entropy
            topic_result[f"{prefix}/KL_Div"] = np.round(kl_divergence_rowwise(dist), 4)  # calculate KL divergence

            def save2file(sort_scores=True):
                # save topic quality scores to file
                model_dir = self.config.model_dir
                topics_dir = Path(model_dir, "topics" if sort_scores else "topics_sorted", saved_name)
                os.makedirs(topics_dir, exist_ok=True)
                for method, score in topic_scores.items():
                    topic_file = os.path.join(topics_dir, f"{method}_{topic_result[method]}.txt")
                    entropy_file = os.path.join(topics_dir, f"{method}_{topic_result[method]}_entropy.txt")
                    scores_list = zip(score, topic_list, entropy_scores, range(len(score)))
                    if sort_scores:
                        scores_list = sorted(scores_list, reverse=True, key=lambda x: x[0])
                    for s, ts, es, i in scores_list:
                        word_weights = [f"{word}({round(dist[i, self.word_dict[word]], 5)})" for word in ts]
                        entropy_str = f"{np.round(s, 4)}({np.round(es, 4)}): {' '.join(word_weights)}\n"
                        write_to_file(topic_file, f"{np.round(s, 4)}: {' '.join(ts)}\n", "a+")
                        write_to_file(entropy_file, entropy_str, "a+")
                    write_to_file(topic_file, f"Average score: {topic_result[method]}\n", "a+")
                write_to_file(os.path.join(topics_dir, "topic_list.txt"), [" ".join(topics) for topics in topic_list])

            save2file(sort_scores=True)
            save2file(sort_scores=False)
        return topic_result
