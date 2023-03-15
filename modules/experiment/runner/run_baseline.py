import os
import ast
import time
import mlflow
from datetime import datetime

from pathlib import Path
from itertools import product

from accelerate import Accelerator

from modules.config.configuration import Configuration
from modules.config.default_config import TEST_CONFIGS
from modules.config.config_utils import set_seed, load_cmd_line
from modules.experiment.quick_run import run
from modules.utils import get_project_root, init_data_loader, log_params, log_metrics, get_experiment_id


def evaluate_run():
    experiment_id = get_experiment_id(experiment_name=config.get("experiment_name", "default"))
    with mlflow.start_run(run_name=f"{config['arch_type']}-{jobid}", experiment_id=experiment_id) as runner:
        start_time = time.time()
        set_seed(config["seed"])
        config.set("run_id", runner.info.run_id)
        if Accelerator().is_main_process:
            log_params(config.final_configs)
        data_loader = init_data_loader(config)
        trainer = run(config, data_loader=data_loader)
        trainer.resume_checkpoint()  # load the best model
        log["#Voc"] = len(data_loader.word_dict)
        if "nc" in cmd_args["task"].lower():
            # run validation
            log.update(trainer.evaluate(data_loader.valid_loader, trainer.model, prefix="val"))
            # run test
            log.update(trainer.evaluate(data_loader.test_loader, trainer.model, prefix="test"))
        else:
            log.update(trainer.evaluate(data_loader.valid_set, trainer.model, prefix="val"))
            log.update(trainer.evaluate(data_loader.test_set, trainer.model, prefix="test"))
        if config.get("topic_evaluation_method", None) is not None:
            log.update(trainer.topic_evaluation(trainer.model, word_dict=data_loader.word_dict))
        if trainer.writer is not None:
            trainer.writer.flush()
        log["Total Time"] = time.time() - start_time
        if trainer.accelerator.is_main_process:  # to avoid duplicated writing
            log_metrics(log)
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
    jobid = config.get("jobid", os.environ.get("SLURM_JOB_ID", "0"))
    topic_evaluation_method = config.get("topic_evaluation_method", None)
    saved_filename = config.get("saved_filename", config.get('arch_type'))
    days, times = timestamp.split("_")
    saved_filename = f"{days}/{saved_filename}_{times}.csv"
    with_entropy = config.get("with_entropy", 0)
    default_saved_name = f'{cmd_args["task"]}/{arch_attr}/' if arch_attr is not None else f'{cmd_args["task"]}/'
    if topic_evaluation_method:
        default_saved_name += f"{topic_evaluation_method}/"
    if with_entropy:
        default_saved_name += "with_entropy/"
    else:
        default_saved_name += "without_entropy/"
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
            if value is None or not bool(value):  # check if value is None or False (empty string)
                continue
            try:
                config.set(arch_attr, ast.literal_eval(value))  # convert to int or float if it is a numerical value
            except (ValueError, SyntaxError):
                config.set(arch_attr, value)
            log = {"arch_type": config.arch_type, "seed": config.seed, arch_attr: value}
            config.set("seed", seed)
            config.set("identifier", f"{arch_attr}_{value}_{config.seed}")
            evaluate_run()
