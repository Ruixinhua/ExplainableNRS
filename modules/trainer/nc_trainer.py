import copy
import os
import numpy as np
import pandas as pd
import torch
import torch.distributed
from pathlib import Path
from scipy.stats import entropy
from modules.base.base_trainer import BaseTrainer
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm
from modules.utils import get_topic_list, get_project_root, get_topic_dist, load_sparse, calc_topic_diversity, \
    read_json, NPMI, compute_coherence, write_to_file, MetricTracker, load_batch_data, word_tokenize


class NCTrainer(BaseTrainer):
    """
    Trainer class
    """
    def __init__(self, model, config, data_loader, **kwargs):
        super().__init__(model, config)
        self.config = config
        self.data_loader = data_loader
        self.train_loader = data_loader.train_loader
        self.entropy_mode = config.get("entropy_mode", "static")
        self.alpha = config.get("alpha", 0.001)
        self.with_entropy = config.get("with_entropy", True if self.alpha > 0 else False)
        self.show_entropy = config.get("show_entropy", False)
        self.calculate_entropy = config.get("calculate_entropy", self.with_entropy)
        self.beta = config.get("beta", 0.1)
        self.len_epoch = len(self.train_loader)
        self.valid_loader = data_loader.valid_loader
        self.do_validation = self.valid_loader is not None
        self.log_step = int(np.sqrt(self.train_loader.batch_size))
        self.train_metrics = MetricTracker(*self.metric_funcs, writer=self.writer)
        self.valid_metrics = MetricTracker(*self.metric_funcs, writer=self.writer)
        self.model, self.optimizer, self.train_loader, self.lr_scheduler = self.accelerator.prepare(
            self.model, self.optimizer, self.train_loader, self.lr_scheduler)

    def run_model(self, batch_dict, model=None, multi_gpu=True):
        """
        run model with the batch data
        :param multi_gpu: default use multi-gpu training
        :param batch_dict: the dictionary of data with format like {"news": Tensor(), "label": Tensor()}
        :param model: by default we use the self model
        :return: the output of running, label used for evaluation, and loss item
        """
        batch_dict = load_batch_data(batch_dict, self.device, multi_gpu)
        output = model(batch_dict) if model is not None else self.model(batch_dict)
        loss = self.criterion(output["pred"], batch_dict["label"])
        if self.with_entropy:
            loss += self.alpha * output["entropy"]
        out_dict = {"label": batch_dict["label"], "loss": loss, "predict": output["pred"]}
        if self.calculate_entropy:
            out_dict.update({"attention_weight": output["attention"], "entropy": output["entropy"]})
        return out_dict

    def update_metrics(self, metrics=None, out_dict=None, predicts=None, labels=None):
        if predicts is not None and labels is not None:
            for met in self.metric_funcs:  # run metric functions
                metrics.update(met.__name__, met(predicts, labels), n=len(labels))
        else:
            n = len(out_dict["label"])
            metrics.update("loss", out_dict["loss"].item(), n=n)  # update metrix
            if self.calculate_entropy:
                metrics.update("doc_entropy", out_dict["entropy"].item() / n, n=n)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        bar = tqdm(enumerate(self.train_loader), total=len(self.train_loader))
        labels, predicts = [], []
        for batch_idx, batch_dict in bar:
            self.optimizer.zero_grad()  # setup gradient to zero
            out_dict = self.run_model(batch_dict, self.model)  # run model
            self.accelerator.backward(out_dict["loss"])  # backpropagation
            self.optimizer.step()  # gradient descent
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx, "train")
            self.update_metrics(self.train_metrics, out_dict)
            labels.extend(out_dict["label"].cpu().tolist())
            predicts.extend(torch.argmax(out_dict["predict"], dim=1).cpu().tolist())
            if batch_idx % self.log_step == 0:  # set bar
                bar.set_description(f"Train Epoch: {epoch} Loss: {out_dict['loss'].item()}")
            if batch_idx == self.len_epoch:
                break
        self.update_metrics(self.train_metrics, predicts=predicts, labels=labels)
        log = self.train_metrics.result()
        if self.do_validation:
            log.update(self.evaluate(self.valid_loader, self.model, epoch))  # update validation log

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def evaluate(self, loader, model, epoch=0, prefix="val"):
        model.eval()
        self.valid_metrics.reset()
        labels, predicts = [], []
        with torch.no_grad():
            for batch_idx, batch_dict in tqdm(enumerate(loader), total=len(loader)):
                out_dict = self.run_model(batch_dict, model, multi_gpu=False)
                self.writer.set_step((epoch - 1) * len(loader) + batch_idx, "evaluate")
                self.update_metrics(self.valid_metrics, out_dict)
                labels.extend(out_dict["label"].cpu().tolist())
                predicts.extend(torch.argmax(out_dict["predict"], dim=1).cpu().tolist())
        self.update_metrics(self.valid_metrics, predicts=predicts, labels=labels)
        for name, p in model.named_parameters():  # add histogram of model parameters to the tensorboard
            self.writer.add_histogram(name, p, bins='auto')
        log = {f"{prefix}_{k}": v for k, v in self.valid_metrics.result().items()}  # return log with prefix
        return log

    def topic_evaluation(self, model=None, word_dict=None, extra_str=None):
        """
        evaluate the topic quality of the BATM model using the topic coherence
        :param model: best model chosen from the training process
        :param word_dict: the word dictionary of the dataset
        :param extra_str: extra string to add to the file name
        :return: topic quality result of the best model
        """
        if model is None:
            model = self.model
        if word_dict is None:
            word_dict = self.data_loader.word_dict
        topic_evaluation_method = self.config.get("topic_evaluation_method", None)
        sort_score = self.config.get("sort_score", True)
        saved_name = f"topics_{self.config.seed}_{self.config.head_num}"
        if extra_str is not None:
            saved_name += f"_{extra_str}"
        if sort_score:
            saved_name += "_sorted"
        else:
            saved_name += "_unsorted"
        topics_dir = Path(self.config.model_dir, "topics", saved_name)
        coherence_dir = Path(self.config.model_dir, "coherence_score", saved_name)
        os.makedirs(coherence_dir, exist_ok=True)
        reverse_dict = {v: k for k, v in word_dict.items()}
        topic_dist = get_topic_dist(model, word_dict)
        self.model = self.model.to(self.device)
        top_n, methods = self.config.get("top_n", 10), self.config.get("coherence_method", ["c_npmi"])
        default_post_dict_path = Path(get_project_root(), "dataset", "utils", "word_dict", "post_process")
        post_word_dict_dir = self.config.get("post_word_dict_dir", default_post_dict_path)
        topic_dists = {"original": topic_dist}
        if post_word_dict_dir is not None and os.path.exists(post_word_dict_dir):
            for path in os.scandir(post_word_dict_dir):
                if not path.name.endswith(".json"):
                    continue
                post_word_dict = read_json(path)
                removed_index = [v for k, v in word_dict.items() if k not in post_word_dict]
                topic_dist_copy = copy.deepcopy(topic_dist)  # copy original topical distribution
                topic_dist_copy[:, removed_index] = 0  # set removed terms to 0
                topic_dists[path.name.replace(".json", "")] = topic_dist_copy
        topic_result = {}
        if torch.distributed.is_initialized():
            model = model.module
        for key, dist in topic_dists.items():  # calculate topic quality for different Post processing methods
            topic_scores = {}
            topic_list = get_topic_list(dist, top_n, reverse_dict)  # convert to tokens list
            ref_data_path = self.config.get("ref_data_path", Path(get_project_root()) / "dataset/data/MIND15.csv")
            if "fast_eval" in topic_evaluation_method:
                ref_texts = load_sparse(ref_data_path)
                scorer = NPMI((ref_texts > 0).astype(int))
                topic_index = [[word_dict[word] - 1 for word in topic] for topic in topic_list]
                # convert to index list: minus 1 because the index starts from 0 (0 is for padding)
                topic_scores[f"{key}_c_npmi"] = scorer.compute_npmi(topics=topic_index, n=top_n)
            if "slow_eval" in topic_evaluation_method:
                tokenized_method = self.config.get("tokenized_method", "keep_all")
                ws = self.config.get("window_size", 200)
                ps = self.config.get("processes", 35)
                tokenized_data_path = Path(get_project_root()) / f"dataset/data/MIND_tokenized.csv"
                ref_df = pd.read_csv(self.config.get("slow_ref_data_path", tokenized_data_path))
                ref_texts = [word_tokenize(doc, tokenized_method) for doc in ref_df["tokenized_text"].values]
                topic_scores.update({
                    f"{key}_{m}": compute_coherence(topic_list, ref_texts, coherence=m, topn=top_n, window_size=ws,
                                                    processes=ps) for m in methods
                })
            if "w2v_sim" in topic_evaluation_method:  # compute word embedding similarity of top-10 words for each topic
                embeddings = model.embedding_layer.embedding.weight.cpu().detach().numpy()
                # embeddings = load_embeddings(**self.config.final_configs)
                count = top_n * (top_n - 1) / 2
                topic_index = [[word_dict[word] for word in topic] for topic in topic_list]
                w2v_sim_list = [np.sum(np.triu(cosine_similarity(embeddings[i]), 1)) / count for i in topic_index]
                topic_scores[f"{key}_w2v_sim"] = w2v_sim_list
            # calculate average score for each topic quality method
            topic_result.update({m: np.round(np.mean(c), 4) for m, c in topic_scores.items()})
            if self.accelerator.is_main_process:  # save topic info
                os.makedirs(topics_dir, exist_ok=True)
                write_to_file(os.path.join(topics_dir, "topic_list.txt"), [" ".join(topics) for topics in topic_list])
                entropy_scores = np.array(entropy(dist, axis=1))
                topic_result[f"{key}_entropy"] = np.round(np.mean(entropy_scores), 4)
                topic_result[f"{key}_div"] = np.round(calc_topic_diversity(topic_list))  # calculate topic diversity
                for method, scores in topic_scores.items():
                    topic_file = os.path.join(topics_dir, f"{method}_{topic_result[method]}.txt")
                    coherence_file = os.path.join(coherence_dir, f"{method}_{topic_result[method]}.txt")
                    entropy_file = os.path.join(topics_dir, f"{method}_{topic_result[method]}_entropy.txt")
                    scores_list = zip(scores, topic_list, entropy_scores, range(len(scores)))
                    if sort_score:  # sort topics by scores
                        scores_list = sorted(scores_list, reverse=True, key=lambda x: x[0])
                    for score, topics, es, i in scores_list:
                        word_weights = [f"{word}({round(dist[i, word_dict[word]], 5)})" for word in topics]
                        entropy_str = f"{np.round(score, 4)}({np.round(es, 4)}): {' '.join(word_weights)}\n"
                        write_to_file(topic_file, f"{np.round(score, 4)}: {' '.join(topics)}\n", "a+")
                        write_to_file(entropy_file, entropy_str, "a+")
                        write_to_file(coherence_file, f"{np.round(score, 4)}\n", "a+")
                    write_to_file(topic_file, f"Average score: {topic_result[method]}\n", "a+")
        if not len(topic_result):
            raise ValueError("No correct topic evaluation method is specified!")
        return topic_result
