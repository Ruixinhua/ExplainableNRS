# -*- coding: utf-8 -*-
# @Organization  : UCD
# @Author        : Dairui Liu
# @Time          : 18/04/2023 00:44
# @Function      :
import importlib
import json
import os
import time
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from collections import defaultdict
import modules.dataset as module_dataset

from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
from accelerate import Accelerator
from sklearn.metrics.pairwise import cosine_similarity

from modules.config.configuration import Configuration, DEFAULT_CONFIGS
from modules.config.config_utils import load_cmd_line, set_seed
from modules.utils import Tokenizer, init_model_class, load_batch_data, get_news_embeds, init_data_loader, init_obj
from modules.dataset import ImpressionDataset


def top_n_indices(arr, n):
    return arr.argsort()[-n:][::-1]


def compute_semantic_similarity(embeddings, n):
    count = n * (n - 1) / 2
    similarity = cosine_similarity(embeddings)
    min_sim = np.min(1 - np.triu(similarity, 1))
    return 1 - np.sum(np.triu(similarity, 1)) / count, min_sim


# setup arguments used to run baseline models
start_time = time.time()
cmd_args = load_cmd_line()
# evaluation directory: must be the directory of the saved model
# evaluate_dir = Path(r"C:\Users\Rui\Documents\Explainable_AI\ExplainableNRS\saved\models\MIND\20230417\RS_Baselines_small_NRMSRSModel_194415\checkpoint\65.8824_42-3")
default = r"/home/dairui/ExplainableNRS/saved/models/MIND/20230417/RS_Baselines_small_NRMSRSModel_194415/checkpoint/65.8824_42-3"
evaluate_dir = Path(cmd_args.get("evaluate_dir", default))
saved_dir_name ="prediction"
config = Configuration(config_file=Path(evaluate_dir, "config.json"))  # load configurations

set_seed(config["seed"])
saved_dir = Path(config.get("saved_dir", DEFAULT_CONFIGS["saved_dir"])) / saved_dir_name  # init saved directory
saved_name = config.get("saved_name", "MIND_Test")  # specify a saved name
os.makedirs(saved_dir / saved_name, exist_ok=True)  # create empty directory
saved_filename = Path(evaluate_dir).name  # tag of evaluated model
# config.save_config(Path(saved_dir, saved_name), f"{saved_filename}_config.json")
logger = config.get_logger(saved_name, verbosity=2)
data_loader = init_data_loader(config)  # load
module_dataset_name = config.get("dataset_class", "NewsRecDataset")
impression_bs = config.get("impression_batch_size", 1)
tokenizer = Tokenizer(**config.final_configs)
test_set = getattr(module_dataset, module_dataset_name)(tokenizer, phase="test", **config.final_configs)
news_set = module_dataset.NewsDataset(test_set)
model_params = {"word_dict": tokenizer.word_dict}
model = init_model_class(config, **model_params)
module_trainer = importlib.import_module("modules.trainer")
trainer = init_obj(config.trainer_type, config.final_configs, module_trainer, model, config, data_loader)
trainer.accelerator = Accelerator(step_scheduler_with_optimizer=False)
trainer.resume_checkpoint(evaluate_dir)  # load the best model
result_dict = {}
trainer.model.eval()

with torch.no_grad():
    try:  # try to do fast evaluation: cache news embeddings
        news_loader = DataLoader(data_loader.valid_set, config.batch_size)
        news_embeds = get_news_embeds(trainer.model, news_loader=data_loader.news_loader, device=trainer.device,
                                      accelerator=trainer.accelerator, num_processes=config.get("num_processes", 2))
    except KeyError or RuntimeError:  # slow evaluation: re-calculate news embeddings every time
        news_embeds = None

news_texts = data_loader.valid_set.news_behavior.news_features["title"]
encoder = SentenceTransformer('paraphrase-mpnet-base-v2')
news_embeddings = encoder.encode(news_texts)

result_dict = defaultdict(list)
with torch.no_grad():
    imp_set = ImpressionDataset(data_loader.valid_set, news_embeds)
    valid_loader = DataLoader(imp_set, impression_bs, collate_fn=data_loader.fn)
    valid_loader = trainer.accelerator.prepare_data_loader(valid_loader)
    bar = tqdm(enumerate(valid_loader), total=len(valid_loader))
    for vi, batch_dict in bar:
        batch_dict = load_batch_data(batch_dict, trainer.device)
        label = batch_dict["label"].cpu().numpy()
        out_dict = model(batch_dict)    # run model
        pred = out_dict["pred"].cpu().numpy()
        can_len = batch_dict["candidate_length"].cpu().numpy()
        for i in range(len(pred)):
            if can_len[i] > 10:
                index = batch_dict["impression_index"][i].cpu().tolist()  # record impression index
                can_news = batch_dict["candidate_news"][i].cpu().numpy()
                can_index = batch_dict["candidate_index"][i].cpu().numpy()
                scores = pred[i][:can_len[i]]
                top10_cans_indices = top_n_indices(scores, 10)
                news = [news_embeddings[can_index[j]] for j in top10_cans_indices]
                ilad5, ilmd5 = compute_semantic_similarity(can_news[top10_cans_indices[:5]], 5)
                result_dict["ILAD@5"].append(ilad5)
                result_dict["ILMD@5"].append(ilmd5)
                ilad5, ilmd = compute_semantic_similarity(news[:5], 5)
                result_dict["ILAD@5_bert"].append(ilad5)
                result_dict["ILMD@5_bert"].append(ilmd)
                ilad10, ilmd10 = compute_semantic_similarity(can_news[top10_cans_indices], 10)
                result_dict["ILAD@10"].append(ilad10)
                result_dict["ILMD@10"].append(ilmd10)
                ilad10, ilmd10 = compute_semantic_similarity(news, 5)
                result_dict["ILAD@10_bert"].append(ilad10)
                result_dict["ILMD@10_bert"].append(ilmd10)

# average result_dict and round to 4 decimal places
for k, v in result_dict.items():
    result_dict[k] = round(sum(v) / len(v), 4)
print(result_dict)
# save result_dict to json file
with open(Path(saved_dir, saved_name, f"{saved_filename}_result.json"), "w") as f:
    json.dump(result_dict, f)