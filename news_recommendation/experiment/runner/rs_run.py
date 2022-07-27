import os
from pathlib import Path

import torch
import torch.backends.cudnn
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.nn.parallel
from news_recommendation.experiment.quick_run import run
from news_recommendation.config import Configuration
from news_recommendation.config.config_utils import set_seed, init_data_loader, init_model_class
from news_recommendation.trainer import MindRSTrainer
from news_recommendation.utils import get_project_root


# def run(config_parser):
#     # prepare for (multi-device) GPU training
#     device, device_ids = prepare_device(config_parser["n_gpu"])
#
#     if len(device_ids) > 1 and config_parser["n_gpu"] > 1:
#         mp.spawn(train_dist, nprocs=len(device_ids), args=(len(device_ids), config_parser))
#     else:
#         train(config_parser)


def train(config: Configuration):
    logger = config.get_logger("MIND RS training")
    # setup data_loader instances
    data_loader = init_data_loader(config)
    # build model architecture, then print to console
    model = init_model_class(config)
    logger.info(model)

    # get function handles of loss and metrics
    trainer = MindRSTrainer(model, config, data_loader)
    trainer.fit()


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
    data_loader = init_data_loader(config, batch_size=bs)
    # build model architecture, then print to console
    model = init_model_class(config).to(local_rank)
    logger.info(model)
    torch.cuda.set_device(local_rank)
    logger.info(f"Set Device: {local_rank}")
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], find_unused_parameters=True)

    logger.info("Set model")
    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    logger.info("Training Begin")
    trainer = MindRSTrainer(model, config, data_loader)

    trainer.fit()


if __name__ == "__main__":
    # multiprocessing options
    torch.multiprocessing.set_start_method("spawn")
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    # custom cli options to modify configuration from default values given in json file.
    config_file = Path(get_project_root()) / "news_recommendation" / "config" / "mind_rs_default.json"
    main_config = Configuration(config_file=config_file)
    set_seed(main_config.seed)
    run(main_config)
