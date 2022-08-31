import argparse
import copy
import importlib
import os
import torch
import random
import sys
import numpy as np

from enum import Enum
from torch.backends import cudnn
from logging import getLogger
from news_recommendation.logger import setup_logging
from news_recommendation.utils import get_project_root


def setup_project_path(configs):
    """
    Setup the project path and the corresponding directories
    """
    # identifier of experiment, default is identified by dataset name and architecture type.
    configs["run_name"] = f"{configs['dataset_name']}/{configs['arch_type']}"
    # make directory for saving checkpoints and log
    configs["model_dir"] = os.path.join(configs["saved_dir"], "models", configs["run_name"])
    os.makedirs(configs["model_dir"], exist_ok=True)
    configs["log_dir"] = os.path.join(configs["saved_dir"], "logs", configs["run_name"])
    os.makedirs(configs["log_dir"], exist_ok=True)
    setup_logging(configs["log_dir"])
    return configs


def convert_config_dict(config_dict):
    r"""This function convert the str parameters to their original type.

    """
    for key in config_dict:
        param = config_dict[key]
        if not isinstance(param, str):
            continue
        try:
            value = eval(param)
            if value is not None and not isinstance(value, (str, int, float, list, tuple, dict, bool, Enum)):
                value = param
        except (NameError, SyntaxError, TypeError):
            if isinstance(param, str):
                if param.lower() == "true":
                    value = True
                elif param.lower() == "false":
                    value = False
                else:
                    value = param
            else:
                value = param
        config_dict[key] = value
    return config_dict


def load_cmd_line():
    """
    Load command line arguments
    """
    cmd_config_dict = {}
    unrecognized_args = []
    if "ipykernel_launcher" not in sys.argv[0]:
        for arg in sys.argv[1:]:
            if not arg.startswith("--") or len(arg[2:].split("=")) != 2:
                unrecognized_args.append(arg)
                continue
            cmd_arg_name, cmd_arg_value = arg[2:].split("=")
            if cmd_arg_name in cmd_config_dict and cmd_arg_value != cmd_config_dict[cmd_arg_name]:
                raise SyntaxError("There are duplicate commend arg '%s' with different value." % arg)
            else:
                cmd_config_dict[cmd_arg_name] = cmd_arg_value
    if len(unrecognized_args) > 0:
        logger = getLogger()
        logger.warning(f"Unrecognized command line arguments(correct is '--key=value'): {' '.join(unrecognized_args)}")
    cmd_config_dict = convert_config_dict(cmd_config_dict)
    cmd_config_dict["cmd_args"] = copy.deepcopy(cmd_config_dict)
    return cmd_config_dict


def set_seed(seed):
    # fix random seeds for reproducibility
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def init_args(cus_args=None):
    args = argparse.ArgumentParser(description="Define Argument")
    if cus_args is not None:
        for ca in cus_args:
            default = ca.get("default", None)
            args.add_argument(*ca["flags"], default=default, type=ca["type"])
    return args
