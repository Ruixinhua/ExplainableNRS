import importlib

from news_recommendation.config.configuration import Configuration
from news_recommendation.utils import init_data_loader, init_model_class, init_obj


def run(config: Configuration, **kwargs):
    logger = config.get_logger("train")
    data_loader = kwargs.get("data_loader")
    model_params = {}
    if hasattr(data_loader, "label_dict") and hasattr(data_loader, "word_dict"):  # for news classification task
        model_params.update({"num_classes": len(data_loader.label_dict), "word_dict": data_loader.word_dict})
    if hasattr(data_loader, "embeds"):  # for extra embedding parameters
        model_params.update({"embeds": data_loader.embeds})
    model = init_model_class(config, **model_params)
    logger.info(model)
    module_trainer = importlib.import_module("news_recommendation.trainer")
    trainer = init_obj(config.trainer_type, config.final_configs, module_trainer, model, config, data_loader)
    trainer.fit()
    return trainer
