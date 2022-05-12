import json
import os
import zipfile
import math
import logging
import requests
from tqdm import tqdm
import torch.nn as nn
from pathlib import Path

from utils import get_project_root


def load_entity(entity: str):
    """
    load entity from mind dataset
    :param entity: entity string in json format
    :return: entities extracted from the input string
    """
    return " ".join([" ".join(e["SurfaceForms"]) for e in json.loads(entity)])


def get_mind_root_path(**kwargs):
    data_path = kwargs.get("data_path", Path(get_project_root()) / "dataset/MIND")
    mind_type, phase = kwargs.get("mind_type", "demo"), kwargs.get("phase")  # options are train, valid, test
    mind_root = Path(data_path) / mind_type / phase  # define mind root path
    if not mind_root.exists():
        mind_url, mind_train_dataset, mind_dev_dataset, mind_utils = get_mind_data_set(mind_type)
        download_resources(mind_url, data_path / "train", mind_train_dataset)  # download mind training files
        download_resources(mind_url, data_path / "valid", mind_dev_dataset)  # download mind validation files
        util_path = data_path / "utils"
        if not util_path.exists():
            # download utils files if not found in the path
            download_resources(r"https://recodatasets.z20.web.core.windows.net/newsrec/", util_path, mind_utils)
    return mind_root


def init_layer(layers, init_std):
    for name, tensor in layers.named_parameters():
        if 'weight' in name:
            nn.init.normal_(tensor, mean=0, std=init_std)
    return layers


def maybe_download(url, filename=None, work_directory=".", expected_bytes=None):
    """Download a file if it is not already downloaded.

    Args:
        filename (str): File name.
        work_directory (str): Working directory.
        url (str): URL of the file to download.
        expected_bytes (int): Expected file size in bytes.

    Returns:
        str: File path of the file downloaded.
    """
    if filename is None:
        filename = url.split("/")[-1]
    os.makedirs(work_directory, exist_ok=True)
    filepath = os.path.join(work_directory, filename)
    if not os.path.exists(filepath):

        r = requests.get(url, stream=True)
        total_size = int(r.headers.get("content-length", 0))
        block_size = 1024
        num_iterables = math.ceil(total_size / block_size)

        with open(filepath, "wb") as file:
            for data in tqdm(
                    r.iter_content(block_size),
                    total=num_iterables,
                    unit="KB",
                    unit_scale=True,
            ):
                file.write(data)
    else:
        logging.getLogger(__name__).info("File {} already downloaded".format(filepath))
    if expected_bytes is not None:
        statinfo = os.stat(filepath)
        if statinfo.st_size != expected_bytes:
            os.remove(filepath)
            raise IOError("Failed to verify {}".format(filepath))

    return filepath


def download_resources(download_url, data_path, remote_resource_name):
    """Download resources.

    Args:
        download_url (str): URL of Azure container.
        data_path (str): Path to download the resources.
        remote_resource_name (str): Name of the resource.
    """
    os.makedirs(data_path, exist_ok=True)
    remote_path = download_url + remote_resource_name
    maybe_download(remote_path, remote_resource_name, data_path)
    zip_ref = zipfile.ZipFile(os.path.join(data_path, remote_resource_name), "r")
    zip_ref.extractall(data_path)
    zip_ref.close()
    os.remove(os.path.join(data_path, remote_resource_name))


def get_mind_data_set(dataset_type):
    """ Get MIND dataset address

    Args:
        dataset_type (str): type of mind dataset, must be in ['large', 'small', 'demo']

    Returns:
        list: data url and train valid dataset name
    """
    assert dataset_type in ["large", "small", "demo"]

    if dataset_type == "large":
        return (
            "https://mind201910small.blob.core.windows.net/release/",
            "MINDlarge_train.zip",
            "MINDlarge_dev.zip",
            "MINDlarge_utils.zip",
        )

    elif dataset_type == "small":
        return (
            "https://mind201910small.blob.core.windows.net/release/",
            "MINDsmall_train.zip",
            "MINDsmall_dev.zip",
            "MINDsma_utils.zip",
        )

    elif dataset_type == "demo":
        return (
            "https://recodatasets.blob.core.windows.net/newsrec/",
            "MINDdemo_train.zip",
            "MINDdemo_dev.zip",
            "MINDdemo_utils.zip",
        )
