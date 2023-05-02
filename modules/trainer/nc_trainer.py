import copy
import torch
import torch.distributed
from tqdm import tqdm

from modules.base.base_trainer import BaseTrainer
from modules.utils import MetricTracker, load_batch_data
from modules.commom import TopicEval


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
        self.alpha = config.get("alpha", 0)
        self.with_entropy = config.get("with_entropy", True if self.alpha > 0 else False)
        self.show_entropy = config.get("show_entropy", False)
        self.calculate_entropy = config.get("calculate_entropy", self.with_entropy)
        self.beta = config.get("beta", 0.1)
        self.len_epoch = len(self.train_loader)
        self.valid_loader = data_loader.valid_loader
        self.do_validation = self.valid_loader is not None
        self.log_step = config.get("log_step", 100)
        self.train_metrics = MetricTracker(*self.metric_funcs, writer=self.writer)
        self.valid_metrics = MetricTracker(*self.metric_funcs, writer=self.writer)
        self.train_topic_evaluator = TopicEval(config, word_dict=data_loader.word_dict, group_name="train_topic_eval")
        self.valid_topic_evaluator = copy.deepcopy(self.train_topic_evaluator)
        self.valid_topic_evaluator.group_name = "valid_topic_eval"
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
        bar = tqdm(enumerate(self.train_loader), total=self.len_epoch)
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

    def topic_evaluation(self, model=None, middle_name=None):
        """
        evaluate the topic quality of the BATM model using the topic coherence
        :param model: best model chosen from the training process
        :param middle_name: extra string to add to the file name
        :return: topic quality result of the best model
        """
        if model is None:
            model = self.model
        if middle_name is None:
            middle_name = "final"
        topic_result = self.valid_topic_evaluator.result(model, middle_name=middle_name, use_post_dict=True)
        if not len(topic_result):
            raise ValueError("No correct topic evaluation method is specified!")
        return topic_result
