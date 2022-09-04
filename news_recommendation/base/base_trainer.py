import copy
import os
import torch
import pandas as pd
import news_recommendation.utils.metric_utils as module_metric

from pathlib import Path
from abc import abstractmethod
from numpy import inf
from accelerate import Accelerator
from news_recommendation import utils as module_loss
from news_recommendation.utils import prepare_device
from news_recommendation.logger import TensorboardWriter


class BaseTrainer:
    """
    Base class for all trainers
    """
    def __init__(self, model, config, **kwargs):
        self.config = config
        self.logger = config.get_logger("trainer", config["verbosity"])
        # prepare for (multi-device) GPU training
        # self.device, device_ids = prepare_device(config["n_gpu"])
        self.accelerator = Accelerator()  # setup accelerator for multi-GPU training
        self.device = self.accelerator.device  # get device for multi-GPU training
        self.logger.info(f"Device: {self.device}")
        self.model = model.to(self.device)
        self.logger.info(f"load device {self.device}")
        # set up model parameters
        self.best_model = copy.deepcopy(model)
        # get function handles of loss and metrics
        self.criterion = getattr(module_loss, config["loss"])
        self.metric_funcs = [getattr(module_metric, met) for met in config["metrics"]]
        # setup visualization writer instance
        self.writer = TensorboardWriter(config.log_dir, self.logger, config["tensorboard"])

        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        self.optimizer = self._build_optimizer()
        self.lr_scheduler = self._build_lr_scheduler()
        # set up trainer parameters
        self.last_best_path = None
        self.not_improved_count = 0

        # configuration to monitor model performance and save best
        if self.config.monitor == "off":
            self.mnt_mode = "off"
            self.mnt_best = 0
        else:
            self.mnt_mode, self.mnt_metric = self.config.monitor.split()
            assert self.mnt_mode in ["min", "max"]

            self.mnt_best = inf if self.mnt_mode == "min" else -inf
            self.early_stop = self.config.get("early_stop", inf)
            if self.early_stop <= 0:
                self.early_stop = inf

        self.start_epoch = 1
        self.checkpoint_dir = config.model_dir

        if config["resume"] is not None:
            self._resume_checkpoint(config["resume"])

    def _build_optimizer(self, **kwargs):
        # build optimizer according to specified optimizer config
        trainable_params = filter(lambda p: p.requires_grad, self.model.parameters())
        learner = kwargs.pop("learner", self.config.get("learner", "Adam"))
        learning_rate = kwargs.pop("learning_rate", self.config.get("learning_rate", 0.001))
        weight_decay = kwargs.pop("weight_decay", self.config.get("weight_decay", 0))
        optimizer = getattr(torch.optim, learner)(trainable_params, lr=learning_rate, weight_decay=weight_decay)
        return optimizer

    def _build_lr_scheduler(self, **kwargs):
        # build learning rate scheduler according to specified scheduler config
        scheduler = kwargs.pop("scheduler", self.config.get("scheduler", "StepLR"))
        step_size = kwargs.pop("step_size", self.config.get("step_size", 50))
        gamma = kwargs.pop("gamma", self.config.get("gamma", 0.1))
        return getattr(torch.optim.lr_scheduler, scheduler)(self.optimizer, step_size=step_size, gamma=gamma)

    @abstractmethod
    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Current epoch number
        """
        raise NotImplementedError

    def _log_info(self, log):
        # print logged information to the screen
        for key, value in log.items():
            self.logger.info("    {:15s}: {}".format(str(key), value))

    def save_log(self, log, **kwargs):
        log["seed"] = self.config["seed"]
        for key, item in self.config.cmd_args.items():  # record all command line arguments
            if isinstance(item, dict):
                log.update(item)
            elif isinstance(item, tuple) or isinstance(item, list):
                pass
            else:
                log[key] = item
        log["run_name"] = self.config["run_name"]
        saved_path = kwargs.get("saved_path", Path(self.checkpoint_dir) / "model_best.csv")
        log_df = pd.DataFrame(log, index=[0])
        if os.path.exists(saved_path):
            log_df = log_df.append(pd.read_csv(saved_path, float_precision="round_trip"), ignore_index=True)
        log_df = log_df.loc[:, ~log_df.columns.str.contains("^Unnamed")]
        log_df.drop_duplicates(inplace=True)
        log_df.to_csv(saved_path)

    def _monitor(self, log, epoch):
        # evaluate model performance according to configured metric, save best checkpoint as model_best with score
        if self.mnt_mode != "off":
            try:
                # check whether model performance improved or not, according to specified metric(mnt_metric)
                improved = (self.mnt_mode == "min" and log[self.mnt_metric] <= self.mnt_best) or \
                           (self.mnt_mode == "max" and log[self.mnt_metric] >= self.mnt_best)
            except KeyError:
                err_msg = f"Warning:Metric {self.mnt_metric} is not found.Model performance monitoring is disabled."
                self.logger.warning(err_msg)
                self.mnt_mode = "off"
                improved = False
            if improved:
                self.mnt_best = log[self.mnt_metric]
                self.not_improved_count = 0
                self.best_model = copy.deepcopy(self.model)
                if self.config.save_model:
                    self._save_checkpoint(epoch, log[self.mnt_metric])
            else:
                self.not_improved_count += 1
            log["monitor_best"] = self.mnt_best
            self.save_log(log)
            self._log_info(log)

    def fit(self):
        """
        Full training logic
        """
        for epoch in range(self.start_epoch, self.config.epochs + 1):
            result = self._train_epoch(epoch)

            # save logged information into log dict
            log = {"epoch": epoch}
            log.update(result)
            self._monitor(log, epoch)
            if self.not_improved_count > self.early_stop:
                self.logger.info(f"Validation performance did not improve for {self.early_stop} epochs. "
                                 "Training stops.")
                break

    def _save_checkpoint(self, epoch, score=0.0):
        """
        Saving checkpoints
        :param epoch: current epoch number
        :param score: current score of monitor metric
        """
        arch = type(self.model).__name__
        state = {
            "arch": arch,
            "epoch": epoch,
            "state_dict": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "monitor_best": self.mnt_best,
            "config": self.config
        }
        best_path = str(self.checkpoint_dir / f"{round(score, 4)}_model_best-epoch{epoch}.pth")
        if self.last_best_path:
            if os.path.exists(self.last_best_path):
                os.remove(self.last_best_path)
        torch.save(state, best_path)
        self.logger.info(f"Saving current best: {best_path}")
        self.last_best_path = best_path

    def _resume_checkpoint(self, resume_path):
        """
        Resume from saved checkpoints

        :param resume_path: Checkpoint path to be resumed
        """
        resume_path = str(resume_path)
        self.logger.info(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path)
        self.start_epoch = checkpoint["epoch"] + 1
        self.mnt_best = checkpoint["monitor_best"]

        # load architecture params from checkpoint.
        if checkpoint["config"] != self.config:
            self.logger.warning("Warning: Architecture configuration given in config file is different from that of "
                                "checkpoint. This may yield an exception while state_dict is being loaded.")
        self.model.load_state_dict(checkpoint["state_dict"])
        # load optimizer state from checkpoint only when optimizer type is not changed.
        if checkpoint["config"]["optimizer_config"] != self.config["optimizer_config"]:
            self.logger.warning("Warning: Optimizer type given in config file is different from that of checkpoint. "
                                "Optimizer parameters not being resumed.")
        else:
            self.optimizer.load_state_dict(checkpoint["optimizer"])

        self.logger.info(f"Checkpoint loaded. Resume training from epoch {self.start_epoch}")
