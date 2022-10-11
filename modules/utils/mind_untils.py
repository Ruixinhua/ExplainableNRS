import json
import os
import zipfile
import math
import logging
import requests
import numpy as np
from tqdm import tqdm
from pathlib import Path

from modules.utils import get_project_root, write_json


def load_entity(entity: str):
    """
    load entity from mind dataset
    :param entity: entity string in json format
    :return: entities extracted from the input string
    """
    return " ".join([" ".join(e["SurfaceForms"]) for e in json.loads(entity)])


def get_mind_dir(**kwargs):
    mind_dir = kwargs.get("mind_dir", None)
    if mind_dir is None:
        data_dir = kwargs.get("data_dir", None)
        if data_dir is None:
            data_dir = Path(get_project_root()) / "dataset/MIND"
        else:
            data_dir = Path(data_dir) / "MIND"
        mind_dir = data_dir / kwargs.get("mind_type", "demo")
        if kwargs.get("phase", None) is not None:   # options for phase are train, valid, test
            mind_dir = mind_dir / kwargs.get("phase")
        return mind_dir
    else:
        raise ValueError("Please specify the mind_dir or data_dir and mind_type")


def check_mind_set(**kwargs):
    mind_type, mind_dir = kwargs.get("mind_type", "small"), kwargs.get("mind_dir", None)
    data_dir = kwargs.get("data_dir", None)
    mind_url = get_mind_download_url()
    train_dir = get_mind_dir(phase="train", mind_dir=mind_dir, mind_type=mind_type, data_dir=data_dir)
    valid_dir = get_mind_dir(phase="valid", mind_dir=mind_dir, mind_type=mind_type, data_dir=data_dir)
    test_dir = get_mind_dir(phase="test", mind_dir=mind_dir, mind_type=mind_type, data_dir=data_dir)
    util_path = get_mind_dir(phase="utils", mind_dir=mind_dir, mind_type=mind_type, data_dir=data_dir)
    if not (Path(train_dir, "behaviors.tsv").exists() and Path(train_dir, "news.tsv").exists()):
        download_resources(mind_url, train_dir, f"MIND{mind_type}_train.zip")  # download mind training files
    if not (Path(valid_dir, "behaviors.tsv").exists() and Path(valid_dir, "news.tsv").exists()):
        download_resources(mind_url, valid_dir, f"MIND{mind_type}_dev.zip")  # download mind validation files
    if mind_type == "large" and not (Path(test_dir, "behaviors.tsv").exists() and Path(test_dir, "news.tsv").exists()):
        download_resources(mind_url, data_dir / "test", f"MIND{mind_type}_test.zip")
    if not util_path.exists():
        download_resources(mind_url, util_path, f"MIND{mind_type}_utils.zip")  # download utils files
    # if not os.path.exists(data_dir.parent / "utils/word_dict/MIND_41059.json"):
    #     rename_utils(util_path)
    # if not os.path.exists(data_dir.parent / "data/MIND15.csv"):
    #     pass  # TODO: Extract news from MIND dataset


def save_tojson(path, util_path):
    with open(path, "rb") as f:
        import pickle
        load_obj = pickle.load(f)
        load_obj["[UNK]"] = 0
    write_json(load_obj, util_path / f"word_dict/MIND_{len(load_obj)}.json")


def save_tonpy(path, util_path):
    load_obj = np.load(path)
    np.save(util_path / f"embed_dict/MIND_{len(load_obj)}.npy", load_obj)


def rename_utils(util_path):
    paths = [util_path / "word_dict.pkl", util_path / "word_dict_all.pkl", util_path / "embedding.npy",
             util_path / "embedding_all.npy"]
    funcs = [save_tojson, save_tojson, save_tonpy, save_tonpy]
    for path, func in zip(paths, funcs):
        func(path, util_path)


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
        data_path: Path to download the resources.
        remote_resource_name (str): Name of the resource.
    """
    os.makedirs(data_path, exist_ok=True)
    remote_path = download_url + remote_resource_name
    maybe_download(remote_path, remote_resource_name, data_path)
    zip_ref = zipfile.ZipFile(os.path.join(data_path, remote_resource_name), "r")
    zip_ref.extractall(data_path)
    zip_ref.close()
    os.remove(os.path.join(data_path, remote_resource_name))


def get_mind_download_url():
    """
    Get MIND dataset address
    Returns: url for downloading MIND dataset
    """
    return "https://mind201910small.blob.core.windows.net/release/"
