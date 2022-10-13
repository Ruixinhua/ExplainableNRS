import importlib

from modules.config.configuration import Configuration
from modules.utils import init_model_class, init_obj


def setup_trainer(config: Configuration, **kwargs):
    logger = config.get_logger("train", 2)
    data_loader = kwargs.get("data_loader")
    model_params = {}
    if hasattr(data_loader, "label_dict") and hasattr(data_loader, "word_dict"):  # for news classification task
        model_params.update({"num_classes": len(data_loader.label_dict), "word_dict": data_loader.word_dict})
    model = init_model_class(config, **model_params)
    logger.info(model)
    module_trainer = importlib.import_module("modules.trainer")
    trainer = init_obj(config.trainer_type, config.final_configs, module_trainer, model, config, data_loader)
    return trainer


def run(config: Configuration, **kwargs):
    trainer = setup_trainer(config, **kwargs)
    trainer.fit()
    return trainer
