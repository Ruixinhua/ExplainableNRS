import copy
import os
from datetime import datetime
from typing import Dict

import torch
import random
import sys
import numpy as np

from enum import Enum
from torch.backends import cudnn
from logging import getLogger
from modules.logger import setup_logging
from modules.utils import get_project_root


def setup_project_path(config):
    """
    Set up the project path and the corresponding directories
    """
    # identifier of experiment, default is identified by dataset name, architecture type, and current time.
    saved_path = config.get("saved_filename", config.get('arch_type'))
    config["project_root"] = config.get("project_root", get_project_root())  # default project path
    config["data_dir"] = config.get("data_dir", os.path.join(config["project_root"], "dataset"))
    config["saved_dir"] = config.get("saved_dir", os.path.join(config["project_root"], "saved"))
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    days, times = timestamp.split("-")
    default_name = f"{config['dataset_name']}/{days}/{saved_path}_{times}"
    config["run_name"] = config.get("run_name", default_name)
    # make directory for saving checkpoints and log
    config["model_dir"] = os.path.join(config["saved_dir"], "models", config["run_name"])
    os.makedirs(config["model_dir"], exist_ok=True)
    setup_logging(config["model_dir"])
    return config


def convert_config_dict(config_dict):
    r"""This function convert the str parameters to their original type.

    """
    for key in config_dict:
        param = config_dict[key]
        if not isinstance(param, str):
            continue
        try:
            value = eval(param)  # convert str to int, float, list, tuple, dict, bool. use ',' to split integer values
            if value is not None and not isinstance(value, (str, int, float, list, tuple, dict, bool, Enum)):
                value = param
        except (NameError, SyntaxError, TypeError):
            if isinstance(param, str):
                if param.lower() == "true":
                    value = True
                elif param.lower() == "false":
                    value = False
                else:
                    value = param.split(",") if "," in param else param  # split by ',' if it is a string
            else:
                value = param
        config_dict[key] = value
    return config_dict


def load_cmd_line() -> Dict:
    """
    Load command line arguments
    :return: dict
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
