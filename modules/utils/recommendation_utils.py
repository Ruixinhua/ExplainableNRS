from tqdm import tqdm
from modules.utils import convert_dict_to_numpy, gather_dict, load_batch_data


def get_news_embeds(model, data_loader=None, news_loader=None, **kwargs):
    """
    run news model and return news vectors (numpy matrix)
    :param model: target running model
    :param data_loader: data loader object used to load news data
    :param news_loader
    :return: numpy matrix of news vectors (each row is a news vector)
    """
    news_embeds = {}
    assert data_loader is not None or news_loader is not None, "data_loader and news_loader can't be both None"
    news_loader = news_loader if news_loader else data_loader.news_loader
    accelerator = kwargs.get("accelerator", None)
    device = kwargs.get("device")
    if accelerator:
        news_loader = accelerator.prepare_data_loader(news_loader)
    for batch_dict in tqdm(news_loader, total=len(news_loader), desc="Get news embeddings"):
        # load data to device
        batch_dict = load_batch_data(batch_dict, device)
        # run news encoder
        news_vec = model.news_encoder(batch_dict)["news_embed"]
        # update news vectors
        news_embeds.update(dict(zip(batch_dict["index"].cpu().tolist(), news_vec.cpu().numpy())))
    return convert_dict_to_numpy(gather_dict(news_embeds))
