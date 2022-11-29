import importlib
import os
import time
import zipfile
import numpy as np
import torch

import modules.dataset as module_dataset

from torch.utils.data import DataLoader
from datetime import datetime
from pathlib import Path
from tqdm import tqdm

from modules.data_loader import collate_fn
from modules.config.configuration import Configuration, DEFAULT_CONFIGS
from modules.config.config_utils import load_cmd_line
from modules.utils import Tokenizer, init_model_class, load_batch_data, get_news_embeds, init_data_loader, init_obj

if __name__ == "__main__":
    # setup arguments used to run baseline models
    start_time = time.time()
    cmd_args = load_cmd_line()
    timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
    saved_dir_name = cmd_args["saved_dir_name"] if "saved_dir_name" in cmd_args else "prediction"
    evaluate_dir = cmd_args.get("evaluate_dir")  # evaluation directory: must be the directory of the saved model
    config = Configuration(config_file=Path(evaluate_dir, "config.json"))  # load configurations
    saved_dir = Path(cmd_args.get("saved_dir", DEFAULT_CONFIGS["saved_dir"])) / saved_dir_name  # init saved directory
    saved_name = config.get("saved_name", "MIND_Test")  # specify a saved name
    os.makedirs(saved_dir / saved_name, exist_ok=True)  # create empty directory
    saved_filename = Path(evaluate_dir).name  # tag of evaluated model
    config.save_config(Path(saved_dir, saved_name), f"{saved_filename}_config.json")
    logger = config.get_logger(saved_name, verbosity=2)
    data_loader = init_data_loader(config)  # load
    module_dataset_name = config.get("dataset_class", "MindRSDataset")
    impression_bs = config.get("impression_batch_size", 1)
    tokenizer = Tokenizer(**config.final_configs)
    test_set = getattr(module_dataset, module_dataset_name)(tokenizer, phase="test", **config.final_configs)
    news_set = module_dataset.NewsDataset(test_set)
    model_params = {"word_dict": tokenizer.word_dict}
    model = init_model_class(config, **model_params)
    # trainer = BaseTrainer(model, config)
    # trainer.model = trainer.accelerator.prepare(trainer.model)  # use accelerator to prepare model
    module_trainer = importlib.import_module("modules.trainer")
    trainer = init_obj(config.trainer_type, config.final_configs, module_trainer, model, config, data_loader)
    trainer.resume_checkpoint(evaluate_dir)  # load the best model
    result_dict = {}
    trainer.model.eval()
    with torch.no_grad():
        valid_result = trainer.evaluate(data_loader, trainer.model, prefix="val")
        logger.info(valid_result)
        try:  # try to do fast evaluation: cache news embeddings
            news_loader = DataLoader(news_set, config.batch_size)
            news_embeds = get_news_embeds(trainer.model, news_loader=news_loader, device=trainer.device,
                                          accelerator=trainer.accelerator)
        except KeyError or RuntimeError:  # slow evaluation: re-calculate news embeddings every time
            news_embeds = None
        impression_set = module_dataset.ImpressionDataset(test_set, news_embeds)
        test_loader = DataLoader(impression_set, impression_bs, collate_fn=collate_fn)
        test_loader = trainer.accelerator.prepare_data_loader(test_loader)
        for batch_dict in tqdm(test_loader, total=len(test_loader), desc="Impressions-Test"):  # run model
            batch_dict = load_batch_data(batch_dict, trainer.device)
            pred = trainer.model(batch_dict)["pred"].cpu().numpy()
            can_len = batch_dict["candidate_length"].cpu().numpy()
            for i in range(len(pred)):
                index = batch_dict["impression_index"][i].cpu().tolist()  # record impression index
                result_dict[index] = pred[i][:can_len[i]]
    with open(Path(saved_dir, saved_name, f"{saved_filename}.txt"), 'w') as f:  # saved predictions as txt file
        for impr_index, preds in tqdm(result_dict.items(), total=len(result_dict), desc="Writing"):
            pred_rank = (np.argsort(np.argsort(preds)[::-1]) + 1).tolist()
            pred_rank = '[' + ','.join([str(i) for i in pred_rank]) + ']'
            f.write(' '.join([str(impr_index), pred_rank]) + '\n')
    f = zipfile.ZipFile(Path(saved_dir, saved_name, f"{saved_filename}.zip"), 'w', zipfile.ZIP_DEFLATED)
    f.write(Path(saved_dir, saved_name, f"{saved_filename}.txt"), arcname='prediction.txt')
    f.close()
    logger.info(f"saved log: {Path(saved_dir, saved_name, f'{saved_filename}.zip')} finished.")
    log = {"arch_type": config.arch_type, "#Voc": len(tokenizer.word_dict), "max_length": config.max_length,
           "seed": config.seed, "Total Time": time.time() - start_time}
    log.update(valid_result)
    trainer.save_log(log, saved_path=Path(saved_dir, saved_name, f"{saved_filename}_log.csv"))
