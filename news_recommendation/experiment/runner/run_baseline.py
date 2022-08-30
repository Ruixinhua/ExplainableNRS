import os
import ast
import time
from pathlib import Path
from itertools import product

from news_recommendation.config.configuration import Configuration
from news_recommendation.config.default_config import TEST_CONFIGS
from news_recommendation.config.config_utils import set_seed, load_cmd_line
from news_recommendation.experiment.quick_run import run, evaluate
from news_recommendation.utils import topic_evaluation, load_docs, filter_tokens, get_project_root, init_data_loader


def evaluate_run():
    start_time = time.time()
    data_loader = init_data_loader(config)
    set_seed(config["seed"])
    trainer = run(config, data_loader=data_loader)
    if cmd_args["task"].lower() == "nc":
        log["#Voc"] = len(data_loader.word_dict)
        log.update(evaluate(trainer, data_loader))
    else:
        log.update(trainer.evaluate(data_loader, trainer.best_model, prefix="val"))
    if evaluate_topic:
        topic_path = Path(config.saved_dir) / "topics" / saved_name / f"{value}_{seed}"
        dataset_name, method = config["dataset_name"].split("/")
        ref_texts = load_docs(dataset_name, method)
        topic_dict = filter_tokens(ref_texts, 20, 0.5)
        topic_dict = {token: data_loader.word_dict[token] for token in topic_dict.values()
                      if token in data_loader.word_dict}
        log["#Ref Voc"] = len(topic_dict)
        scores = topic_evaluation(trainer, topic_dict, topic_path, ref_texts, config.get("top_n", 25), log["#Voc"])
        log.update(scores)
    log["Total Time"] = time.time() - start_time
    trainer.save_log(log, saved_path=saved_dir / f'{saved_name}.csv')


if __name__ == "__main__":
    # setup arguments used to run baseline models
    cmd_args = load_cmd_line()
    if cmd_args["task"].lower() == "nc":
        config = Configuration()
    else:
        config_file = Path(get_project_root()) / "news_recommendation" / "config" / "mind_rs_default.json"
        config = Configuration(config_file=config_file)
    saved_dir_name = cmd_args["saved_dir_name"] if "saved_dir_name" in cmd_args else "performance"
    saved_dir = Path(config.saved_dir) / saved_dir_name  # init saved directory
    os.makedirs(saved_dir, exist_ok=True)  # create empty directory
    arch_attr = config.get("arch_attr", None)  # test an architecture attribute
    saved_name = f'{cmd_args["task"]}-{config["dataset_name"].replace("/", "-")}-{arch_attr}'
    evaluate_topic, entropy_constraint = config.get("evaluate_topic", 0), config.get("entropy_constraint", 0)
    if evaluate_topic:
        saved_name += f"-evaluate_topic"
    if entropy_constraint:
        saved_name += "-entropy_constraint"
    # acquires test values for a given arch attribute
    test_values = config.get("values") if hasattr(config, "values") else TEST_CONFIGS.get(arch_attr, None)
    seeds = [int(s) for s in config.seeds] if hasattr(config, "seeds") else TEST_CONFIGS.get("seeds")
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
