import os
from pathlib import Path

import torch
import torch.backends.cudnn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.parallel

import experiment.data_loader as module_data
import models as module_arch
from config.config_utils import init_args, custom_args, set_seed
from config.parse_config import ConfigParser
from experiment.trainer import MindRSTrainer
from utils import prepare_device, read_json, get_project_root


def run(config):
    # prepare for (multi-device) GPU training
    device, device_ids = prepare_device(config["n_gpu"])

    if len(device_ids) > 1 and config["n_gpu"] > 1:
        mp.spawn(train_dist, nprocs=len(device_ids), args=(len(device_ids), config))
    else:
        train(config)


def train(config):
    logger = config.get_logger("MIND RS training")
    # setup data_loader instances
    data_loader = config.init_obj("data_config", module_data)
    # build model architecture, then print to console
    model = config.init_obj("arch_config", module_arch)
    logger.info(model)

    # get function handles of loss and metrics
    trainer = MindRSTrainer(model, config, data_loader)
    trainer.train()


def train_dist(local_rank, nprocs, config):
    # TODO: Distributed Training
    logger = config.get_logger("mind train")
    cudnn.benchmark = True
    dist.init_process_group(backend='nccl', init_method='tcp://127.0.0.1:23457', world_size=nprocs, rank=local_rank)
    # When using a single GPU per process and per
    # DistributedDataParallel, we need to divide the batch size
    # ourselves based on the total number of GPUs we have
    bs = int(config["data_loader"]["args"]["batch_size"] / nprocs)
    # setup data_loader instances
    data_loader = config.init_obj("data_config", module_data, batch_size=bs)

    # build model architecture, then print to console
    model = config.init_obj("arch", module_arch).to(local_rank)
    logger.info(model)
    torch.cuda.set_device(local_rank)
    logger.info(f"Set Device: {local_rank}")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    logger.info("Set model")
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    logger.info("Training Begin")
    trainer = MindRSTrainer(model, config, data_loader)

    trainer.train()


if __name__ == "__main__":
    # multiprocessing options
    torch.multiprocessing.set_start_method("spawn")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # custom cli options to modify configuration from default values given in json file.
    cus_args = [
        {"flags": ["-mt", "--mind_type"], "type": str, "target": "data_config"},
    ]
    default_config = read_json(Path(get_project_root()) / "config" / "mind_rs_default.json")
    args, options = init_args(), custom_args(cus_args)
    main_config = ConfigParser.from_args(args, options, default_config=default_config)
    main_config.config.update(default_config)
    set_seed(42)
    run(main_config)
