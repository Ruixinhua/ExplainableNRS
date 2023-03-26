import os

from tqdm import tqdm
from pathlib import Path
from modules.utils import convert_dict_to_numpy, gather_dict, load_batch_data, gpu_stat, get_project_root


def get_news_embeds(model, news_loader, **kwargs):
    """
    run news model and return news vectors (numpy matrix)
    :param model: target running model
    :param news_loader: news loader with all news data
    :return: numpy matrix of news vectors (each row is a news vector)
    """
    news_embeds = {}
    assert news_loader is not None, "must specify news_loader"
    accelerator = kwargs.get("accelerator", None)
    device = kwargs.get("device")
    if accelerator:
        news_loader = accelerator.prepare_data_loader(news_loader)
    bar = tqdm(news_loader, total=len(news_loader))
    for batch_dict in bar:
        bar.set_description(f"Get news embeddings: {gpu_stat()}")
        # load data to device
        batch_dict = load_batch_data(batch_dict, device)
        # run news encoder
        news_vec = model.news_encoder(batch_dict)["news_embed"]
        # update news vectors
        news_embeds.update(dict(zip(batch_dict["index"].cpu().tolist(), news_vec.cpu().numpy())))
    return convert_dict_to_numpy(gather_dict(news_embeds))


def get_default_upath(**kwargs):
    """
    get default user id path
    :return: user path
    """
    data_root = Path(kwargs.get("data_dir", os.path.join(get_project_root(), "dataset")))  # root directory
    default_path = os.path.join(data_root, "utils", kwargs.get("dataset_name"), f"M{kwargs.get('subset_type')}.json")
    uid_path = kwargs.get("uid_path", default_path)
    if not os.path.exists(uid_path):
        raise ValueError("User ID dictionary is not found, please check your config file")
    return uid_path
