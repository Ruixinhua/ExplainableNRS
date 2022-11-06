import math
import torch
import torch.distributed
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm

from config.configuration import Configuration
from dataset import ImpressionDataset
from trainer import NCTrainer
from utils import gather_dict, load_batch_data, get_news_embeds


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
            batch_dict = load_batch_data(batch_dict, self.device)
            # setup model and train model
            self.optimizer.zero_grad()
            output = self.model(batch_dict)
            loss = self.criterion(output, batch_dict["label"])
            if self.entropy_constraint:
                loss += self.alpha * output["entropy"]
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

    def _valid_epoch(self, model=None, data_loader=None, extra_str=None):
        """
        Validate after training an epoch
        :return: A log that contains information about validation
        """
        result_dict = {}
        impression_bs = self.config.get("impression_batch_size", 1024)
        valid_method = self.config.get("valid_method", "fast_evaluation")
        if torch.distributed.is_initialized():
            model = self.model.module if model is None else model.module
        else:
            model = self.model if model is None else model
        data_loader = self.mind_loader if data_loader is None else data_loader
        model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            try:  # try to do fast evaluation: cache news embeddings
                if valid_method == "fast_evaluation":
                    news_embeds = get_news_embeds(model, data_loader, device=self.device, accelerator=self.accelerator)
                else:
                    news_embeds = None
            except KeyError or RuntimeError:  # slow evaluation: re-calculate news embeddings every time
                news_embeds = None
            imp_set = ImpressionDataset(data_loader.valid_set, news_embeds)
            valid_loader = DataLoader(imp_set, impression_bs, collate_fn=data_loader.fn)
            valid_loader = self.accelerator.prepare_data_loader(valid_loader)
            for batch_dict in tqdm(valid_loader, total=len(valid_loader)):  # run model
                batch_dict = load_batch_data(batch_dict, self.device)
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
        for name, p in model.named_parameters():  # add histogram of model parameters to the tensorboard
            self.writer.add_histogram(name, p, bins="auto")
        return {f"{prefix}_{k}": v for k, v in log.items()}  # return log with prefix
