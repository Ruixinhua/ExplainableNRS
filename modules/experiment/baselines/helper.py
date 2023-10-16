import os
from pathlib import Path

from recommenders.datasets.mind import (
    download_mind,
    extract_mind,
)
from recommenders.datasets.download_utils import maybe_download
from recommenders.models.deeprec.deeprec_utils import download_deeprec_resources
from recommenders.models.deeprec.models import dkn
from recommenders.models.newsrec.models import lstur, naml, nrms, npa
from recommenders.models.deeprec.io import dkn_iterator
from recommenders.models.newsrec.io import mind_iterator, mind_all_iterator
from recommenders.models.newsrec.newsrec_utils import get_mind_data_set


def get_model_class(model_name):
    model_to_module = {
        "DKN": dkn,
        "LSTURModel": lstur,
        "NAMLModel": naml,
        "NRMSModel": nrms,
        "NPAModel": npa,
    }

    module = model_to_module.get(model_name)
    try:
        model_class = getattr(module, model_name)
    except AttributeError:
        raise ValueError(f"Model {model_name} not found!")
    return model_class


def get_iterator_class(model_name):
    model_to_module = {
        "DKN": (dkn_iterator, "DKNTextIterator"),
        "NAMLModel": (mind_all_iterator, "MINDAllIterator"),
    }
    # default using MINDIterator
    module, module_name = model_to_module.get(
        model_name, (mind_iterator, "MINDIterator")
    )

    try:
        iterator_class = getattr(module, module_name)
    except AttributeError:
        raise ValueError(f"Module {module_name} not found!")
    return iterator_class


def get_yaml_path(model_name, data_path: Path, mind_size="small"):
    model_to_yaml = {
        "DKN": f"dkn_{mind_size}.yaml",
        "LSTURModel": "lstur.yaml",
        "NAMLModel": "naml.yaml",
        "NRMSModel": "nrms.yaml",
        "NPAModel": "npa.yaml",
    }
    yaml_file = data_path / "utils" / model_to_yaml.get(model_name, None)
    if yaml_file and not yaml_file.exists():
        maybe_download(
            url=f"https://recodatasets.z20.web.core.windows.net/deeprec/deeprec/dkn/dkn_MIND{mind_size}.yaml",
            work_directory=str(yaml_file.parent),
        )
        download_deeprec_resources(
            r"https://recodatasets.z20.web.core.windows.net/newsrec/",
            os.path.join(data_path, "utils"),
            f"MIND{mind_size}_utils.zip",
        )
    return yaml_file


def get_train_valid_path(data_path: Path, mind_size="small"):
    train_zip, valid_zip = download_mind(size=mind_size, dest_path=str(data_path))
    train_path, valid_path = extract_mind(train_zip, valid_zip)
    return train_path, valid_path
