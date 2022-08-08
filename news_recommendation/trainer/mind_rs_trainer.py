import math

import numpy as np
import torch
import torch.distributed
from tqdm import tqdm

from config.configuration import Configuration
from trainer import NCTrainer


class MindRSTrainer(NCTrainer):
    """
    Trainer class
    """

    def __init__(self, model, config: Configuration, data_loader, **kwargs):
        super().__init__(model, config, data_loader, **kwargs)
        self.valid_interval = config.get("valid_interval", 0.1)
        self.fast_evaluation = config.get("fast_evaluation", True)
        self.train_strategy = config.get("train_strategy", "pair_wise")
        self.loader = data_loader
        self.behaviors = data_loader.valid_set.behaviors

    def _validation(self, epoch, batch_idx, do_monitor=True):
        # do validation when reach the interval
        log = {"epoch/step": f"{epoch}/{batch_idx}"}
        log.update(**{'val_' + k: v for k, v in self._valid_epoch().items()})
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
        if torch.distributed.is_initialized():
            self.train_loader.sampler.set_epoch(epoch)
        bar = tqdm(enumerate(self.train_loader), total=length)
        # self._validation(epoch, 0)
        for batch_idx, batch_dict in bar:
            # load data to device
            batch_dict = self.load_batch_data(batch_dict)
            # setup model and train model
            self.optimizer.zero_grad()
            output = self.model(batch_dict)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
            loss = self.criterion(output, batch_dict["label"])
            if hasattr(self, "accelerator"):
                self.accelerator.backward(loss)
            else:
                loss.backward()
            self.optimizer.step()
            # record loss
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())
            if batch_idx % self.log_step == 0:
                bar.set_description(f"Train Epoch: {epoch} Loss: {loss.item()}")
            if batch_idx == self.len_epoch:
                break
            if (batch_idx + 1) % math.ceil(length * self.valid_interval) == 0 and (batch_idx + 1) < length:
                self._validation(epoch, batch_idx)
        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return self._validation(epoch, length, False)

    def _run_news_data(self, model, data_loader=None):
        news_vectors = {}
        data_loader = self.loader if data_loader is None else data_loader
        for batch_dict in tqdm(data_loader.news_loader, total=len(data_loader.news_loader)):
            # load data to device
            batch_dict = self.load_batch_data(batch_dict)
            # run news encoder
            news_vec = model.news_encoder(batch_dict)
            # update news vectors
            news_vectors.update(dict(zip(batch_dict["index"].cpu().tolist(), news_vec.cpu().numpy())))
        return news_vectors

    def _run_user_data(self, model, news_vectors, data_loader=None):
        """
        run user model and return user vectors
        :param model: a model with a user encoder
        :param news_vectors: cached news vectors dictionary
        :return: user vectors dictionary using impression index as the key of dictionary
        """
        user_vectors = {}
        data_loader = self.loader if data_loader is None else data_loader
        for batch_dict in tqdm(data_loader.user_loader, total=len(data_loader.user_loader)):
            # load data to device
            input_feat = {
                "history_news": torch.tensor(np.array(
                    [[news_vectors[i.tolist()] for i in history] for history in batch_dict["history_index"]]
                ))
            }
            batch_dict.update(input_feat)
            batch_dict = self.load_batch_data(batch_dict)
            # run news encoder
            user_vec = model.user_encoder(batch_dict)
            # update news vectors
            user_vectors.update(dict(zip(batch_dict["impression_index"].cpu().tolist(), user_vec.cpu().numpy())))
        return user_vectors

    def _valid_epoch(self, model=None, data_loader=None):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        """
        # used for distributed validation
        news_object, user_object = [{} for _ in range(2)], [{} for _ in range(2)]
        group_label, group_pred = [], []
        if torch.distributed.is_initialized():
            model = self.model.module if model is None else model.module
        else:
            model = self.model if model is None else model
        model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            news_vectors = self._run_news_data(model, data_loader)
            if torch.distributed.is_initialized():
                torch.distributed.barrier()
                torch.distributed.all_gather_object(news_object, news_vectors)
                for i in range(2):
                    news_vectors.update(news_object[i])
            if self.fast_evaluation:
                user_vectors = self._run_user_data(model, news_vectors, data_loader)
                if torch.distributed.is_initialized():
                    torch.distributed.barrier()
                    torch.distributed.all_gather_object(user_object, user_vectors)
                    for i in range(2):
                        user_vectors.update(user_object[i])
                for batch_dict in tqdm(self.valid_loader, total=len(self.valid_loader)):
                    candidate_news = np.array([[
                            news_vectors[i.tolist()] for i in cans] for cans in batch_dict["candidate_index"]
                    ])
                    history_news = np.array(
                            [user_vectors[index.tolist()] for index in batch_dict["impression_index"]]
                    )
                    if self.config["out_layer"] == "product":
                        pred = np.dot(candidate_news.squeeze(0), history_news.squeeze(0))
                    else:
                        input_feat = {
                            "candidate_news": torch.tensor(candidate_news),
                            "history_news": torch.tensor(history_news)
                        }
                        batch_dict.update(input_feat)
                        batch_dict = self.load_batch_data(batch_dict)
                        pred = model.predict(batch_dict, evaluate=True).cpu().squeeze().tolist()
                    group_pred.append(pred)
                    group_label.append(batch_dict["label"].squeeze().cpu().tolist())
            else:
                behaviors = data_loader.valid_set.behaviors
                # TODO: slow evaluation
                behaviors = zip(*[behaviors[attr] for attr in ["history_news", "candidate_news", "labels"]])
                for history, candidate, label in tqdm(behaviors, total=len(behaviors["labels"])):
                    # setup input feat of history news and candidate news
                    input_feat = {
                        "history_news": torch.tensor(np.array([[news_vectors[i] for i in history]])),
                        "candidate_news": torch.tensor(np.array([[news_vectors[i] for i in candidate]])),
                        "history_length": torch.tensor([len(history)]),
                    }
                    input_feat = self.load_batch_data(input_feat)
                    input_feat["history_news"] = model.user_encoder(input_feat)
                    group_pred.append(model.predict(input_feat).squeeze().cpu().tolist())
                    group_label.append(label)
            for met in self.metric_ftns:
                self.valid_metrics.update(met.__name__, met(group_label, group_pred))
        return self.valid_metrics.result()

    def evaluate(self, loader, model, epoch=0, prefix="val"):
        model.eval()
        self.valid_metrics.reset()
        log = self._valid_epoch(model, loader)
        for name, p in model.named_parameters():  # add histogram of model parameters to the tensorboard
            self.writer.add_histogram(name, p, bins='auto')
        return {f"{prefix}_{k}": v for k, v in log.items()}  # return log with prefix
