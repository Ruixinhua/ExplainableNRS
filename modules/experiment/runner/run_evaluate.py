import os
import time
from datetime import datetime

from pathlib import Path

from modules.config.configuration import Configuration, DEFAULT_CONFIGS
from modules.config.config_utils import set_seed, load_cmd_line
from modules.experiment.quick_run import setup_trainer
from modules.utils import init_data_loader


if __name__ == "__main__":
    # setup arguments used to run baseline models
    cmd_args = load_cmd_line()
    timestamp = datetime.now().strftime(r'%m%d_%H%M%S')
    saved_dir_name = cmd_args["saved_dir_name"] if "saved_dir_name" in cmd_args else "performance"
    saved_dir = Path(cmd_args.get("saved_dir", DEFAULT_CONFIGS["saved_dir"])) / saved_dir_name  # init saved directory
    os.makedirs(saved_dir, exist_ok=True)  # create empty directory
    topic_evaluation_method = cmd_args.get("topic_evaluation_method", None)
    saved_filename = cmd_args.get("saved_filename", None)
    if saved_filename is not None:
        saved_filename = f"{saved_filename}_{timestamp}.csv"
    else:
        saved_filename = f"BATM_{timestamp}.csv"
    saved_name = cmd_args.get("saved_name", "evaluate")  # must specify a saved name
    for ed in os.scandir(cmd_args.get("evaluate_dir")):
        if "model_best" not in ed.name or not Path(ed).is_dir():
            continue
        config = Configuration(config_file=Path(ed, "config.json"))
        logger = config.get_logger(saved_name, verbosity=2)
        start_time = time.time()
        data_loader = init_data_loader(config)
        set_seed(config["seed"])
        trainer = setup_trainer(config, data_loader=data_loader)
        trainer.resume_checkpoint(ed.path)  # load the best model
        log = {
            "arch_type": config.arch_type, "#Voc": len(data_loader.word_dict), "head_num": config.head_num,
            "head_dim": config.head_dim, "max_length": config.max_length, "seed": config.seed,
        }
        if "nc" in config.task.lower():
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
