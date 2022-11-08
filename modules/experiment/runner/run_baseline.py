import os
import ast
import time
from datetime import datetime

from pathlib import Path
from itertools import product

from modules.config.configuration import Configuration
from modules.config.default_config import TEST_CONFIGS
from modules.config.config_utils import set_seed, load_cmd_line
from modules.experiment.quick_run import run
from modules.utils import get_project_root, init_data_loader


def evaluate_run():
    start_time = time.time()
    data_loader = init_data_loader(config)
    set_seed(config["seed"])
    trainer = run(config, data_loader=data_loader)
    trainer.resume_checkpoint()  # load the best model
    log["#Voc"] = len(data_loader.word_dict)
    if "nc" in cmd_args["task"].lower():
        # run validation
        log.update(trainer.evaluate(trainer.valid_loader, trainer.model, prefix="val"))
        # run test
        log.update(trainer.evaluate(data_loader.test_loader, trainer.model, prefix="test"))
    else:
        log.update(trainer.evaluate(data_loader, trainer.model, prefix="val"))
    if config.get("topic_evaluation_method", None) is not None:
        log.update(trainer.topic_evaluation(data_loader=data_loader))
    log["Total Time"] = time.time() - start_time
    if trainer.accelerator.is_main_process:  # to avoid duplicated writing
        saved_path = saved_dir / saved_name / saved_filename
        os.makedirs(saved_path.parent, exist_ok=True)
        trainer.save_log(log, saved_path=saved_path)
        logger.info(f"saved log: {saved_path} finished.")


if __name__ == "__main__":
    # setup arguments used to run baseline models
    cmd_args = load_cmd_line()
    timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
    if "nc" in cmd_args["task"].lower():
        config = Configuration()
    else:
        config_file = Path(get_project_root()) / "modules" / "config" / "mind_rs_default.json"
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
    default_saved_name = f'{cmd_args["task"]}/{arch_attr}/' if arch_attr is not None else f'{cmd_args["task"]}/'
    if topic_evaluation_method:
        default_saved_name += f"{topic_evaluation_method}/"
    if entropy_constraint:
        default_saved_name += "entropy_constraint/"
    saved_name = config.get("saved_name", default_saved_name)
    logger = config.get_logger(saved_name)
    # acquires test values for a given arch attribute
    test_values = config.get("values", TEST_CONFIGS.get(arch_attr, None))
    seeds = [int(s) for s in config.get("seeds", TEST_CONFIGS.get("seeds"))]
    if arch_attr is None or arch_attr is False or test_values is None:
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
