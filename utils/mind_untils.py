import json
import torch.nn as nn
from pathlib import Path


def load_entity(entity: str):
    """
    load entity from mind dataset
    :param entity: entity string in json format
    :return: entities extracted from the input string
    """
    return " ".join([" ".join(e["SurfaceForms"]) for e in json.loads(entity)])


def get_mind_file_path(data_path, mind_type, phase):
    mind_path = Path(data_path) / mind_type / phase
    return mind_path / "news.tsv", mind_path / "behaviors.tsv"


def init_layer(layers, init_std):
    for name, tensor in layers.named_parameters():
        if 'weight' in name:
            nn.init.normal_(tensor, mean=0, std=init_std)
    return layers
