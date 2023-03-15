from tqdm import tqdm
from modules.utils import convert_dict_to_numpy, gather_dict, load_batch_data, gpu_stat


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
        # load data to device
        batch_dict = load_batch_data(batch_dict, device)
        # run news encoder
        news_vec = model.news_encoder(batch_dict)["news_embed"]
        bar.set_description(f"Get news embeddings: {gpu_stat()}")
        # update news vectors
        news_embeds.update(dict(zip(batch_dict["index"].cpu().tolist(), news_vec.cpu().numpy())))
    return convert_dict_to_numpy(gather_dict(news_embeds))
