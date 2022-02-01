import os
import ast
from pathlib import Path
from itertools import product
from experiment.config import ConfigParser
from experiment.config import init_args, custom_args, set_seed
from experiment.runner.nc_base import run, test, init_data_loader
from utils import topic_evaluation, load_docs, filter_tokens

# setup default values
DEFAULT_VALUES = {
    "seeds": [42, 2020, 2021, 25, 4],
    "head_num": [10, 30, 50, 70, 100, 150, 180, 200],
    "embedding_type": ["distilbert-base-uncased", "bert-base-uncased", "roberta-base", "xlnet-base-cased",
                       "allenai/longformer-base-4096", "transfo-xl-wt103"]
}


if __name__ == "__main__":
    # setup arguments used to run baseline models
    baseline_args = [
        {"flags": ["-ss", "--seeds"], "type": str, "target": None},
        {"flags": ["-aa", "--arch_attr"], "type": str, "target": None},
        {"flags": ["-va", "--values"], "type": str, "target": None},
        {"flags": ["-tp", "--evaluate_topic"], "type": int, "target": None},
        {"flags": ["-tn", "--top_n"], "type": int, "target": None}
    ]
    args, options = init_args(), custom_args(baseline_args)
    config_parser = ConfigParser.from_args(args, options)
    config = config_parser.config
    saved_dir = Path(config.project_root) / "saved" / "performance"  # init saved directory
    os.makedirs(saved_dir, exist_ok=True)  # create empty directory
    arch_attr = config.get("arch_attr", "base")  # test an architecture attribute
    saved_name = f'{config.data_config["name"].replace("/", "-")}-{arch_attr}'
    evaluate_topic, entropy_constraint = config.get("evaluate_topic", 0), config.get("entropy_constraint", 0)
    if evaluate_topic:
        saved_name += f"-evaluate_topic"
    if entropy_constraint:
        saved_name += "-entropy_constraint"
    # acquires test values for a given arch attribute
    test_values = config.get("values").split(",") if hasattr(config, "values") else DEFAULT_VALUES.get(arch_attr, [0])
    seeds = [int(s) for s in config.seeds.split(",")] if hasattr(config, "seeds") else DEFAULT_VALUES.get("seeds")
    for value, seed in product(test_values, seeds):
        try:
            config.set(arch_attr, ast.literal_eval(value))  # convert to int or float if it is a numerical value
        except ValueError:
            config.set(arch_attr, value)
        config.set("seed", seed)
        data_loader = init_data_loader(config_parser)
        log = {"arch_type": config.arch_config["type"], "seed": config.seed, arch_attr: value,
               "variant_name": config.arch_config.get("variant_name", None), "#Voc": len(data_loader.word_dict)}
        set_seed(log["seed"])
        trainer = run(config_parser, data_loader)
        log.update(test(trainer, data_loader))
        if evaluate_topic:
            topic_path = Path(config.project_root) / "saved" / "topics" / saved_name / f"{value}_{seed}"
            dataset_name, method = config.data_config["name"].split("/")
            ref_texts = load_docs(dataset_name, method)
            topic_dict = filter_tokens(ref_texts, 0, 1)
            topic_dict = {token: data_loader.word_dict[token] for token in topic_dict.values()}
            log["#Ref Voc"] = len(topic_dict)
            scores = topic_evaluation(trainer, topic_dict, topic_path, ref_texts, config.get("top_n", 25), log["#Voc"])
            log.update(scores)
        trainer.save_log(log, saved_path=saved_dir / f'{saved_name}.csv')
