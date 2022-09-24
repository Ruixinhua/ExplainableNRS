import math
import torch
import torch.distributed
import pandas as pd
import numpy as np
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
from pathlib import Path

from config.configuration import Configuration
from dataset import ImpressionDataset
from trainer import NCTrainer
from utils import get_topic_list, get_project_root, get_topic_dist, gather_dict, convert_dict_to_numpy, \
    load_sparse, load_dataset_df, word_tokenize, NPMI, compute_coherence, write_to_file, save_topic_info


class MindRSTrainer(NCTrainer):
    """
    Trainer class
    """

    def __init__(self, model, config: Configuration, data_loader, **kwargs):
        super().__init__(model, config, data_loader, **kwargs)
        self.valid_interval = config.get("valid_interval", 0.1)
        self.fast_evaluation = config.get("fast_evaluation", True)
        self.train_strategy = config.get("train_strategy", "pair_wise")
        self.mind_loader = data_loader
        self.behaviors = data_loader.valid_set.behaviors

    def _validation(self, epoch, batch_idx, do_monitor=True):
        # do validation when reach the interval
        log = {"epoch/step": f"{epoch}/{batch_idx}"}
        log.update(**{"val_" + k: v for k, v in self._valid_epoch(extra_str=f"{epoch}_{batch_idx}").items()})
        if do_monitor:
            self._monitor(log, epoch)
        self.model.train()  # reset to training mode
        self.train_metrics.reset()
        return log

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch
        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        length = len(self.train_loader)
        bar = tqdm(enumerate(self.train_loader), total=length)
        # self._validation(epoch, 0)
        for batch_idx, batch_dict in bar:
            # load data to device
            batch_dict = self.load_batch_data(batch_dict)
            # setup model and train model
            self.optimizer.zero_grad()
            output = self.model(batch_dict)
            loss = self.criterion(output, batch_dict["label"])
            self.accelerator.backward(loss)
            self.optimizer.step()
            # record loss
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update("loss", loss.item())
            if batch_idx % self.log_step == 0:
                bar.set_description(f"Train Epoch: {epoch} Loss: {loss.item()}")
            if batch_idx == self.len_epoch:
                break
            if (batch_idx + 1) % math.ceil(length * self.valid_interval) == 0 and (batch_idx + 1) < length:
                self._validation(epoch, batch_idx)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return self._validation(epoch, length, False)

    def get_news_embeds(self, model, data_loader=None):
        """
        run news model and return news vectors (numpy matrix)
        :param model: target running model
        :param data_loader: data loader object used to load news data
        :return: numpy matrix of news vectors (each row is a news vector)
        """
        news_embeds = {}
        data_loader = self.mind_loader if data_loader is None else data_loader
        news_loader = self.accelerator.prepare_data_loader(data_loader.news_loader)
        for batch_dict in tqdm(news_loader, total=len(news_loader)):
            # load data to device
            batch_dict = self.load_batch_data(batch_dict)
            # run news encoder
            news_vec = model.news_encoder(batch_dict)
            # update news vectors
            news_embeds.update(dict(zip(batch_dict["index"].cpu().tolist(), news_vec.cpu().numpy())))
        return convert_dict_to_numpy(gather_dict(news_embeds))

    def _valid_epoch(self, model=None, data_loader=None, extra_str=None):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        """
        result_dict = {}
        impression_bs = self.config.get("impression_batch_size", 1024)
        if torch.distributed.is_initialized():
            model = self.model.module if model is None else model.module
        else:
            model = self.model if model is None else model
        model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            try:  # try to do fast evaluation: cache news embeddings 
                news_embeds = self.get_news_embeds(model, data_loader)
            except KeyError or RuntimeError:  # slow evaluation: re-calculate news embeddings every time
                news_embeds = None
            imp_set = ImpressionDataset(self.mind_loader.valid_set, news_embeds)
            valid_loader = DataLoader(imp_set, impression_bs, collate_fn=self.mind_loader.fn)
            valid_loader = self.accelerator.prepare_data_loader(valid_loader)
            for batch_dict in tqdm(valid_loader, total=len(valid_loader)):  # run model
                batch_dict = self.load_batch_data(batch_dict)
                label = batch_dict["label"].cpu().numpy()
                pred = model(batch_dict).cpu().numpy()
                can_len = batch_dict["candidate_length"].cpu().numpy()
                for i in range(len(label)):
                    index = batch_dict["impression_index"][i].cpu().tolist()  # record impression index
                    result_dict[index] = {m.__name__: m(label[i][:can_len[i]], pred[i][:can_len[i]])
                                          for m in self.metric_funcs}
            result_dict = gather_dict(result_dict)  # gather results
            eval_result = dict(np.round(pd.DataFrame.from_dict(result_dict, orient="index").mean(), 4))  # average
            if self.config.get("evaluate_topic_by_epoch", False) and self.config.get("topic_evaluation_method", None):
                eval_result.update(self.topic_evaluation(model, data_loader, extra_str=extra_str))
                self.accelerator.wait_for_everyone()
        # self.logger.info(f"validation time: {time.time() - start}")
        return eval_result

    def evaluate(self, loader, model, epoch=0, prefix="val"):
        """call this method after training"""
        model.eval()
        self.valid_metrics.reset()
        log = self._valid_epoch(model, loader)
        if not self.config.get("evaluate_topic_by_epoch", False) and self.config.get("topic_evaluation_method", None):
            log.update(self.topic_evaluation(model, loader, extra_str="best"))
        for name, p in model.named_parameters():  # add histogram of model parameters to the tensorboard
            self.writer.add_histogram(name, p, bins="auto")
        return {f"{prefix}_{k}": v for k, v in log.items()}  # return log with prefix

    def topic_evaluation(self, model=None, data_loader=None, extra_str=None):
        if model is None:
            model = self.model
        if data_loader is None:
            data_loader = self.mind_loader
        topic_evaluation_method = self.config.get("topic_evaluation_method", None)
        saved_name = f"topics_{self.config.seed}_{self.config.head_num}"
        if extra_str is not None:
            saved_name += f"_{extra_str}"
        topic_path = Path(self.config.model_dir, saved_name)
        reverse_dict = {v: k for k, v in data_loader.word_dict.items()}
        topic_dist = get_topic_dist(model, data_loader, self.config.get("topic_variant", "base"))
        self.model = self.model.to(self.device)
        top_n, methods = self.config.get("top_n", 10), self.config.get("coherence_method", "c_npmi")
        topic_list = get_topic_list(topic_dist, top_n, reverse_dict)  # convert to tokens list
        ref_data_path = self.config.get("ref_data_path", Path(get_project_root()) / "dataset/data/MIND15.csv")
        if self.config.get("save_topic_info", False) and self.accelerator.is_main_process:  # save topic info
            os.makedirs(topic_path, exist_ok=True)
            write_to_file(os.path.join(topic_path, "topic_list.txt"), [" ".join(topics) for topics in topic_list])
        if topic_evaluation_method == "fast_eval":
            ref_texts = load_sparse(ref_data_path)
            scorer = NPMI((ref_texts > 0).astype(int))
            topic_index = [[data_loader.word_dict[word] - 1 for word in topic] for topic in topic_list]
            topic_scores = {"c_npmi": scorer.compute_npmi(topics=topic_index, n=top_n)}
        else:
            dataset_name = self.config.get("dataset_name", "MIND15"),
            tokenized_method = self.config.get("tokenized_method", "use_tokenize")
            ref_df, _ = load_dataset_df(dataset_name, data_path=ref_data_path, tokenized_method=tokenized_method)
            ref_texts = [word_tokenize(doc, tokenized_method) for doc in ref_df["data"].values]
            topic_scores = {m: compute_coherence(topic_list, ref_texts, m, top_n) for m in methods.split(",")}
        topic_result = {m: np.round(np.mean(c), 4) for m, c in topic_scores.items()}
        if self.config.get("save_topic_info", False) and self.accelerator.is_main_process:
            # avoid duplicated saving
            sort_score = self.config.get("sort_score", True)
            topic_result = save_topic_info(topic_path, topic_list, topic_scores, sort_score)
        return topic_result
