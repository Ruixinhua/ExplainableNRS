import json
import os
import pickle
import random
from collections import OrderedDict
from pathlib import Path
from typing import Union, Dict

import torch


def read_json(file: Union[str, os.PathLike]):
    """
    Read json from file
    :param file: the path to the json file
    :return: ordered dictionary content
    """
    file = Path(file)
    with file.open("rt") as handle:
        return json.load(handle, object_hook=OrderedDict)


def write_json(content: Dict, file: Union[str, os.PathLike]):
    """
    Write content to a json file
    :param content: the content dictionary
    :param file: the path to save json file
    """
    file = Path(file)
    with file.open("wt") as handle:
        json.dump(content, handle, indent=4, sort_keys=False)


def write_to_file(file: Union[str, os.PathLike], text: Union[str, list], mode: str = "w"):
    with open(file, mode, encoding="utf-8") as w:
        if isinstance(text, str):
            w.write(text)
        elif isinstance(text, list):
            w.write("\n".join(text))


def prepare_device(n_gpu_use):
    """
    setup GPU device if available. get gpu device indices which are used for DataParallel
    """
    n_gpu = torch.cuda.device_count()
    if n_gpu_use > 0 and n_gpu == 0:
        print("Warning: There\'s no GPU available on this machine,"
              "training will be performed on CPU.")
        n_gpu_use = 0
    if n_gpu_use > n_gpu:
        print(f"Warning: The number of GPU\'s configured to use is {n_gpu_use}, but only {n_gpu} are "
              "available on this machine.")
        n_gpu_use = n_gpu
    device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
    list_ids = list(range(n_gpu_use))
    return device, list_ids


def del_index_column(df):
    return df.loc[:, ~df.columns.str.contains("^Unnamed")]


def get_project_root(**kwargs):
    project_name = kwargs.pop("project_name", "bi_attention")
    file_parts = Path(os.getcwd()).parts
    abs_path = Path(f"{os.sep}".join(file_parts[:file_parts.index(project_name) + 1]))
    return os.path.relpath(abs_path, os.getcwd())


def load_dict(file_path):
    """ load pickle file
    Args:
        file path (str): file path

    Returns:
        (obj): pickle load obj
    """
    with open(file_path, "rb") as f:
        return pickle.load(f)


def news_sampling(news, ratio):
    """ Sample ratio samples from news list.
    If length of news is less than ratio, pad zeros.

    Args:
        news (list): packed_input news list
        ratio (int): sample number

    Returns:
        list: output of sample list.
    """
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)


def init_obj(module_name: str, module: object, *args, **kwargs):
    """
    Finds a function handle with the name given as 'type' in config, and returns the
    instance initialized with corresponding arguments given.

    `object = config.init_obj('trainer_config', module, a, b=1)`
    is equivalent to
    `object = module.module_name(a, b=1)`
    """
    return getattr(module, module_name)(*args, **kwargs)
