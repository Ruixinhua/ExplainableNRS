import os
import ast
import time
import numpy as np
from datetime import datetime

from pathlib import Path
from itertools import product

from news_recommendation.config.configuration import Configuration
from news_recommendation.config.default_config import TEST_CONFIGS
from news_recommendation.config.config_utils import set_seed, load_cmd_line
from news_recommendation.experiment.quick_run import run, evaluate
from news_recommendation.utils import get_topic_list, get_project_root, init_data_loader, get_topic_dist, \
    load_sparse, load_dataset_df, word_tokenize, NPMI, compute_coherence, write_to_file, save_topic_info


def evaluate_run():
    start_time = time.time()
    data_loader = init_data_loader(config)
    set_seed(config["seed"])
    trainer = run(config, data_loader=data_loader)
    trainer.resume_checkpoint()  # load the best model
    log["#Voc"] = len(data_loader.word_dict)
    if "nc" in cmd_args["task"].lower():
        log.update(evaluate(trainer, data_loader))
    else:
        log.update(trainer.evaluate(data_loader, trainer.model, prefix="val"))
    if topic_evaluation_method:
        topic_path = Path(config.saved_dir) / f"topics/{saved_name}/{seed}/{datetime.now().strftime(r'%m%d_%H%M%S')}"
        topic_num = config.get("head_num")  # the number of heads is the number of topics
        reverse_dict = {v: k for k, v in data_loader.word_dict.items()}
        topic_dist = get_topic_dist(trainer.model, list(data_loader.word_dict.values()), topic_num, log["#Voc"])
        top_n, methods = config.get("top_n", 10), config.get("coherence_method", "c_v,c_npmi")
        topic_list = get_topic_list(topic_dist, top_n, reverse_dict)  # convert to tokens list
        ref_data_path = config.get("ref_data_path", Path(get_project_root()) / "dataset/data/MIND15.csv")
        if config.get("save_topic_info", False):
            os.makedirs(topic_path, exist_ok=True)
            write_to_file(os.path.join(topic_path, "topic_list.txt"), [" ".join(topics) for topics in topic_list])
        if topic_evaluation_method == "fast_eval":
            ref_texts = load_sparse(ref_data_path)
            scorer = NPMI((ref_texts > 0).astype(int))
            topic_index = [[data_loader.word_dict[word] - 1 for word in topic] for topic in topic_list]
            topic_scores = {"c_npmi": scorer.compute_npmi(topics=topic_index, n=top_n)}
        else:
            dataset_name, method = config.get("dataset_name", "MIND15"), config.get("tokenized_method", "keep_all")
            ref_df, _ = load_dataset_df(dataset_name, data_path=ref_data_path, tokenized_method=method)
            ref_texts = [word_tokenize(doc, method) for doc in ref_df["data"].values]
            topic_scores = {m: compute_coherence(topic_list, ref_texts, m, top_n) for m in methods.split(",")}
        topic_result = {m: np.round(np.mean(c), 4) for m, c in topic_scores.items()}
        if config.get("save_topic_info", False):
            topic_result = save_topic_info(topic_path, topic_list, topic_scores)
        log.update(topic_result)
    log["Total Time"] = time.time() - start_time
    saved_path = saved_dir / saved_name / saved_filename
    os.makedirs(saved_path.parent, exist_ok=True)
    if trainer.accelerator.is_main_process:  # to avoid duplicated writing
        trainer.save_log(log, saved_path=saved_path)
        logger.info(f"saved log: {saved_path} finished.")


if __name__ == "__main__":
    # setup arguments used to run baseline models
    cmd_args = load_cmd_line()
    timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
    if "nc" in cmd_args["task"].lower():
        config = Configuration()
    else:
        config_file = Path(get_project_root()) / "news_recommendation" / "config" / "mind_rs_default.json"
        config = Configuration(config_file=config_file)
    saved_dir_name = cmd_args["saved_dir_name"] if "saved_dir_name" in cmd_args else "performance"
    saved_dir = Path(config.saved_dir) / saved_dir_name  # init saved directory
    os.makedirs(saved_dir, exist_ok=True)  # create empty directory
    arch_attr = config.get("arch_attr", None)  # test an architecture attribute
    topic_evaluation_method = config.get("topic_evaluation_method", None)
    saved_filename = config.get("saved_filename", None)
    if saved_filename:
        saved_filename = f"{saved_filename}_{timestamp}.csv"
    else:
        saved_filename = f"{config.get('arch_type')}_{timestamp}.csv"
    entropy_constraint = config.get("entropy_constraint", 0)
    default_saved_name = f'{cmd_args["task"]}/{arch_attr}/'
    if topic_evaluation_method:
        default_saved_name += f"{topic_evaluation_method}/"
    if entropy_constraint:
        default_saved_name += "entropy_constraint/"
    saved_name = config.get("saved_name", default_saved_name)
    logger = config.get_logger(saved_name)
    # acquires test values for a given arch attribute
    test_values = config.get("values", TEST_CONFIGS.get(arch_attr, None))
    seeds = [int(s) for s in config.get("seeds", TEST_CONFIGS.get("seeds"))]
    if arch_attr is None or test_values is None:
        for seed in seeds:
            log = {"arch_type": config.arch_type, "seed": config.seed}
            config.set("seed", seed)
            evaluate_run()
    else:
        for value, seed in product(test_values, seeds):
            try:
                config.set(arch_attr, ast.literal_eval(value))  # convert to int or float if it is a numerical value
            except ValueError:
                config.set(arch_attr, value)
            log = {"arch_type": config.arch_type, "seed": config.seed, arch_attr: value}
            config.set("seed", seed)
            evaluate_run()
