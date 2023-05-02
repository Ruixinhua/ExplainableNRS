import os
import time
from datetime import datetime

from pathlib import Path

from modules.config.configuration import Configuration
from modules.config.config_utils import set_seed, load_cmd_line
from modules.experiment.quick_run import setup_trainer
from modules.utils import init_data_loader, load_word_dict

if __name__ == "__main__":
    # setup arguments used to run baseline models
    cmd_args = load_cmd_line()
    timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
    saved_dir_name = cmd_args["saved_dir_name"] if "saved_dir_name" in cmd_args else "performance"
    saved_dir = Path(cmd_args.get("saved_dir", None)) / saved_dir_name  # init saved directory
    topic_evaluation_method = cmd_args.get("topic_evaluation_method", None)
    saved_name = cmd_args.get("saved_name", "evaluate")  # must specify a saved name
    evaluate_dir = cmd_args.get("evaluate_dir", None)
    evaluate_paths = []
    for ed in os.scandir(cmd_args.get("evaluate_dir")):
        if Path(ed).is_dir():
            evaluate_paths.append(ed.path)
    if len(evaluate_paths) == 0:  # only one checkpoint directory
        evaluate_paths = [evaluate_dir]
    for path in evaluate_paths:
        saved_filename = f"{Path(path).name}.csv"
        config = Configuration(config_file=Path(path, "config.json"))
        logger = config.get_logger(saved_name, verbosity=2)
        start_time = time.time()
        data_loader = init_data_loader(config)
        set_seed(config["seed"])
        trainer = setup_trainer(config, data_loader=data_loader)
        trainer.resume_checkpoint(path)  # load the best model
        log = {
            "arch_type": config.arch_type, "#Voc": len(load_word_dict(**config.final_configs)), "seed": config.seed,
            "head_num": config.head_num, "max_length": config.max_length, "head_dim": config.head_dim,
        }
        if "nc" in config.task.lower():
            # run validation
            log.update(trainer.evaluate(trainer.valid_loader, trainer.model, prefix="val"))
            # run test
            log.update(trainer.evaluate(data_loader.test_loader, trainer.model, prefix="test"))
        else:
            log.update(trainer.evaluate(data_loader.valid_set, trainer.model, prefix="val"))
            log.update(trainer.evaluate(data_loader.test_set, trainer.model, prefix="test"))
        if config.get("topic_evaluation_method", None) is not None:
            log.update(trainer.topic_evaluation(trainer.model))
        log["Total Time"] = time.time() - start_time
        if trainer.accelerator.is_main_process:  # to avoid duplicated writing
            saved_path = saved_dir / saved_name / saved_filename
            os.makedirs(saved_path.parent, exist_ok=True)
            trainer.save_log(log, saved_path=saved_path)
            logger.info(f"saved log: {saved_path} finished.")
